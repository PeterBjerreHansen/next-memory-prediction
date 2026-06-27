from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .artifacts import append_jsonl, artifacts_for, write_json
from .checkpoint import config_from_checkpoint, load_checkpoint, restore_checkpoint
from .config import (
    ExperimentConfig,
    transition_target_for_variant,
)
from .countdown import (
    CountdownTokenizer,
    check_countdown_solution,
    check_countdown_solution_nextlat_compat,
    countdown_solution_token_count,
)
from .data import (
    load_corpora,
    load_generalization_corpus,
    make_tokenizer,
    sequential_batches,
)
from .diagnostics import (
    effective_rank,
    masked_states,
    mean_adjacent_cosine,
    mean_cross_pass_cosine,
    safe_perplexity,
)
from .factory import build_model, count_parameters
from .models import MemoryTapeOutput
from .objectives import compute_loss, temporal_transition_kl_mask
from .runtime import autocast_context, resolve_device


@torch.no_grad()
def evaluate_batches(
    *,
    config: ExperimentConfig,
    model,
    predictor,
    batches,
    tokenizer: CountdownTokenizer,
    device: torch.device,
    accuracy_prefix: str = "val",
    include_accuracy: bool = False,
) -> dict[str, Any]:
    model.eval()
    if predictor is not None:
        predictor.eval()
    totals: dict[str, float] = defaultdict(float)
    pass_totals: list[float] | None = None
    ntp_token_count = 0
    transition_count = 0
    transition_kl_count = 0
    transition_ce_count = 0
    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        tokens = batch.tokens
        transition_config = config.objective.transition
        with autocast_context(device, config.training.precision):
            output = model(tokens)
            losses = compute_loss(
                variant=config.model.variant,
                model=model,
                output=output,
                tokens=tokens,
                target_mask=batch.target_mask,
                pad_token_id=tokenizer.pad_id,
                eos_token_id=tokenizer.eos_id,
                predictor=predictor,
                lambda_transition=transition_config.lambda_transition,
                lambda_kl=getattr(transition_config, "lambda_kl", 1.0),
                lambda_ce=getattr(transition_config, "lambda_ce", 0.0),
                transition_horizon=getattr(transition_config, "horizon", 1),
                transition_target=transition_target_for_variant(
                    config.model.variant,
                    transition_config,
                ),
                ntp_pass_weights=config.objective.ntp_pass_weights,
            )
        metrics = losses.detached_metrics()
        batch_ntp_tokens = int(
            (batch.target_mask & (tokens[:, 1:] != tokenizer.pad_id)).sum().item()
        )
        totals["weighted_ntp_loss"] += (
            float(metrics["weighted_ntp_loss"]) * batch_ntp_tokens
        )
        totals["final_pass_nll"] += (
            float(metrics["final_pass_nll"]) * batch_ntp_tokens
        )
        ntp_token_count += batch_ntp_tokens
        if metrics["transition_prediction_loss"] is not None:
            next_tokens = tokens[:, 1:]
            batch_transitions = int(
                (
                    (next_tokens != tokenizer.eos_id)
                    & (next_tokens != tokenizer.pad_id)
                )
                .sum()
                .item()
            )
            totals["transition_prediction_loss"] += float(
                metrics["transition_prediction_loss"]
            ) * batch_transitions
            transition_count += batch_transitions
        if metrics["transition_kl_loss"] is not None:
            kl_tokens = int(
                temporal_transition_kl_mask(
                    tokens,
                    batch.target_mask,
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                )
                .sum()
                .item()
            )
            totals["transition_kl_loss"] += float(
                metrics["transition_kl_loss"]
            ) * kl_tokens
            transition_kl_count += kl_tokens
        if metrics["transition_ce_loss"] is not None:
            ce_tokens = int(
                temporal_transition_kl_mask(
                    tokens,
                    batch.target_mask,
                    eos_token_id=tokenizer.eos_id,
                    pad_token_id=tokenizer.pad_id,
                )
                .sum()
                .item()
            )
            totals["transition_ce_loss"] += float(
                metrics["transition_ce_loss"]
            ) * ce_tokens
            transition_ce_count += ce_tokens
        current_passes = list(map(float, metrics["pass_nlls"]))
        if pass_totals is None:
            pass_totals = [0.0] * len(current_passes)
        for index, value in enumerate(current_passes):
            pass_totals[index] += value * batch_ntp_tokens
    if ntp_token_count == 0:
        raise ValueError("validation batches contain no next-token targets")
    weighted_ntp = totals["weighted_ntp_loss"] / ntp_token_count
    final_pass_nll = totals["final_pass_nll"] / ntp_token_count
    result: dict[str, Any] = {
        "loss": weighted_ntp,
        "weighted_ntp_loss": weighted_ntp,
        "final_pass_nll": final_pass_nll,
        "pass_nlls": [
            value / ntp_token_count for value in pass_totals or []
        ],
        "ntp_pass_weights": metrics["ntp_pass_weights"],
        "ntp_tokens": ntp_token_count,
    }
    result["perplexity"] = safe_perplexity(result["final_pass_nll"])
    if predictor is not None:
        transition_config = config.objective.transition
        transition_prediction_loss = (
            totals["transition_prediction_loss"] / transition_count
            if transition_count
            else 0.0
        )
        result["transition_prediction_loss"] = transition_prediction_loss
        result["transition_target"] = transition_target_for_variant(
            config.model.variant,
            transition_config,
        )
        result["transition_count"] = transition_count
        result["transition_horizon"] = getattr(transition_config, "horizon", 1)
        result["lambda_transition"] = transition_config.lambda_transition
        lambda_kl = getattr(transition_config, "lambda_kl", 1.0)
        lambda_ce = getattr(transition_config, "lambda_ce", 0.0)
        result["lambda_kl"] = lambda_kl
        result["lambda_ce"] = lambda_ce
        result["loss"] += (
            transition_config.lambda_transition * transition_prediction_loss
        )
        if transition_kl_count:
            transition_kl_loss = (
                totals["transition_kl_loss"] / transition_kl_count
                if transition_kl_count
                else 0.0
            )
            result["transition_kl_loss"] = transition_kl_loss
            result["transition_kl_count"] = transition_kl_count
            result["loss"] += lambda_kl * transition_kl_loss
        if transition_ce_count:
            transition_ce_loss = totals["transition_ce_loss"] / transition_ce_count
            result["transition_ce_loss"] = transition_ce_loss
            result["transition_ce_count"] = transition_ce_count
            result["loss"] += lambda_ce * transition_ce_loss
    if (
        include_accuracy
        and getattr(tokenizer, "task", None) == "countdown"
        and hasattr(config, "data")
    ):
        result.update(
            countdown_accuracy_for_batches(
                config=config,
                model=model,
                batches=batches,
                tokenizer=tokenizer,
                device=device,
                prefix=accuracy_prefix,
            )
        )
    return result


def _countdown_accuracy_batches(config, corpus, tokenizer):
    return sequential_batches(
        corpus,
        tokenizer,
        batch_size=config.training.micro_batch_size,
        block_size=config.model.block_size,
        num_pause_tokens=config.data.num_pause_tokens,
        num_batches=config.evaluation.accuracy_batches,
    )


@torch.no_grad()
def countdown_accuracy_for_batches(
    *,
    config: ExperimentConfig,
    model,
    batches,
    tokenizer: CountdownTokenizer,
    device: torch.device,
    prefix: str,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    compat_correct = 0
    valid_equations = [0] * config.data.countdown_num_equations
    compat_valid_equations = [0] * config.data.countdown_num_equations
    max_new_tokens = countdown_solution_token_count(
        config.data.countdown_num_equations
    )
    for cpu_batch in batches:
        batch = cpu_batch.to(device)
        for row in range(batch.tokens.size(0)):
            prompt_length = int(batch.prompt_lengths[row].item())
            full_length = int(batch.lengths[row].item())
            if prompt_length < 1 or full_length <= prompt_length:
                continue
            prompt = batch.tokens[row : row + 1, :prompt_length]
            numbers = tokenizer.number_tokens(prompt[0].detach().cpu().tolist())
            if len(numbers) < config.data.countdown_input_numbers + 1:
                continue
            input_numbers = numbers[: config.data.countdown_input_numbers]
            target = numbers[config.data.countdown_input_numbers]
            generated = model.generate(
                prompt,
                max_new_tokens,
                do_sample=False,
                inference_mode="recompute",
                eos_token_id=tokenizer.eos_id,
            )
            prediction_tokens = generated[0, prompt_length:].detach().cpu().tolist()
            prediction = tokenizer.decode(prediction_tokens)
            checked = check_countdown_solution(
                input_numbers=input_numbers,
                target=target,
                prediction=prediction,
                num_equations=config.data.countdown_num_equations,
            )
            compat_checked = check_countdown_solution_nextlat_compat(
                input_numbers=input_numbers,
                target=target,
                prediction=prediction,
                num_equations=config.data.countdown_num_equations,
            )
            total += 1
            correct += int(checked.correct)
            compat_correct += int(compat_checked.correct)
            for index, is_valid in enumerate(checked.valid_equations):
                valid_equations[index] += int(is_valid)
            for index, is_valid in enumerate(compat_checked.valid_equations):
                compat_valid_equations[index] += int(is_valid)
    denominator = max(total, 1)
    result: dict[str, Any] = {
        f"{prefix}_accuracy": correct / denominator,
        f"{prefix}_strict_multiset_accuracy": correct / denominator,
        f"{prefix}_nextlat_compat_accuracy": compat_correct / denominator,
        f"{prefix}_sequences": total,
    }
    for index, value in enumerate(valid_equations, start=1):
        result[f"{prefix}_valid_equation_{index}"] = value / denominator
    for index, value in enumerate(compat_valid_equations, start=1):
        result[f"{prefix}_nextlat_compat_valid_equation_{index}"] = (
            value / denominator
        )
    return result


@torch.no_grad()
def representation_diagnostics(
    *,
    config: ExperimentConfig,
    model,
    batches,
    tokenizer: CountdownTokenizer,
    device: torch.device,
) -> dict[str, Any]:
    hidden_rows = []
    memory_rows = []
    hidden_norm_sum = 0.0
    hidden_count = 0
    memory_norm_sum = 0.0
    memory_count = 0
    adjacent_hidden = []
    adjacent_memory = []
    cross_pass = []

    for cpu_batch in batches:
        tokens = cpu_batch.tokens.to(device)
        output = model(tokens)
        hidden = output.hidden_states
        valid_hidden = masked_states(hidden, tokens, tokenizer.pad_id)
        hidden_rows.append(valid_hidden.cpu())
        hidden_norm_sum += float(valid_hidden.norm(dim=-1).sum().cpu())
        hidden_count += valid_hidden.size(0)
        adjacent_hidden.append(
            mean_adjacent_cosine(hidden, tokens, pad_id=tokenizer.pad_id)
        )
        if isinstance(output, MemoryTapeOutput):
            memory = output.memory_states
            valid_memory = masked_states(memory, tokens, tokenizer.pad_id)
            memory_rows.append(valid_memory.cpu())
            memory_norm_sum += float(valid_memory.norm(dim=-1).sum().cpu())
            memory_count += valid_memory.size(0)
            adjacent_memory.append(
                mean_adjacent_cosine(memory, tokens, pad_id=tokenizer.pad_id)
            )
            cross_pass.append(
                mean_cross_pass_cosine(
                    output.memory_states_per_pass[-2],
                    output.memory_states_per_pass[-1],
                    tokens,
                    pad_id=tokenizer.pad_id,
                )
            )

    all_hidden = torch.cat(hidden_rows, dim=0)
    result: dict[str, Any] = {
        "hidden_effective_rank": effective_rank(all_hidden),
        "hidden_mean_norm": hidden_norm_sum / max(hidden_count, 1),
        "hidden_adjacent_cosine": sum(adjacent_hidden) / len(adjacent_hidden),
    }
    if memory_rows:
        all_memory = torch.cat(memory_rows, dim=0)
        result.update(
            {
                "memory_effective_rank": effective_rank(all_memory),
                "memory_mean_norm": memory_norm_sum / max(memory_count, 1),
                "memory_adjacent_cosine": sum(adjacent_memory)
                / len(adjacent_memory),
                "adjacent_pass_memory_cosine": sum(cross_pass) / len(cross_pass),
            }
        )
    return result


@torch.no_grad()
def generation_diagnostics(
    *,
    config: ExperimentConfig,
    model,
    val_corpus,
    tokenizer: CountdownTokenizer,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = []
    correct = 0
    compat_correct = 0
    max_new_tokens = countdown_solution_token_count(
        config.data.countdown_num_equations
    )

    for index in range(min(config.evaluation.generation_prompts, len(val_corpus))):
        row = val_corpus[index]
        encoded, prompt_length = tokenizer.tokenize(
            row,
            num_pause_tokens=config.data.num_pause_tokens,
        )
        prompt_ids = encoded[:prompt_length]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        generated = model.generate(
            prompt.clone(),
            max_new_tokens,
            do_sample=False,
            inference_mode="recompute",
            eos_token_id=tokenizer.eos_id,
        )
        prediction_tokens = generated[0, prompt_length:].detach().cpu().tolist()
        prediction = tokenizer.decode(prediction_tokens)
        numbers = tokenizer.number_tokens(prompt_ids)
        input_numbers = numbers[: config.data.countdown_input_numbers]
        target = numbers[config.data.countdown_input_numbers]
        checked = check_countdown_solution(
            input_numbers=input_numbers,
            target=target,
            prediction=prediction,
            num_equations=config.data.countdown_num_equations,
        )
        compat_checked = check_countdown_solution_nextlat_compat(
            input_numbers=input_numbers,
            target=target,
            prediction=prediction,
            num_equations=config.data.countdown_num_equations,
        )
        correct += int(checked.correct)
        compat_correct += int(compat_checked.correct)
        samples.append(
            {
                "index": index,
                "prompt": row.split("|", 1)[0] + "|",
                "expected": row.split("|", 1)[1],
                "prediction": prediction,
                "correct": checked.correct,
                "nextlat_compat_correct": compat_checked.correct,
                "valid_equations": list(checked.valid_equations),
                "nextlat_compat_valid_equations": list(
                    compat_checked.valid_equations
                ),
            }
        )

    count = len(samples)
    metrics = {
        "sample_accuracy": (correct / count) if count else None,
        "nextlat_compat_sample_accuracy": (
            compat_correct / count if count else None
        ),
        "samples": count,
    }
    return metrics, samples


def load_run(
    run_dir: str | Path,
    *,
    checkpoint_name: str = "best.pt",
    device_override: str | None = None,
):
    artifacts = artifacts_for(run_dir)
    checkpoint_path = artifacts.run_dir / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = artifacts.latest_checkpoint
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = config_from_checkpoint(checkpoint)
    if device_override is not None:
        config.training.device = device_override
    device = resolve_device(config.training.device)
    tokenizer = make_tokenizer(config.data)
    model, predictor = build_model(config, vocab_size=tokenizer.vocab_size)
    restore_checkpoint(
        checkpoint,
        model=model,
        predictor=predictor,
        restore_rng=False,
    )
    model.to(device).eval()
    if predictor is not None:
        predictor.to(device).eval()
    return artifacts, config, tokenizer, model, predictor, checkpoint, device


def evaluate_run(
    run_dir: str | Path,
    *,
    checkpoint_name: str = "best.pt",
    device_override: str | None = None,
) -> dict[str, Any]:
    (
        artifacts,
        config,
        tokenizer,
        model,
        predictor,
        checkpoint,
        device,
    ) = load_run(
        run_dir,
        checkpoint_name=checkpoint_name,
        device_override=device_override,
    )
    _, val_corpus = load_corpora(config.data)
    loss_batches = sequential_batches(
        val_corpus,
        tokenizer,
        batch_size=config.training.micro_batch_size,
        block_size=config.model.block_size,
        num_pause_tokens=config.data.num_pause_tokens,
        num_batches=config.training.eval_batches,
    )
    loss_metrics = evaluate_batches(
        config=config,
        model=model,
        predictor=predictor,
        batches=loss_batches,
        tokenizer=tokenizer,
        device=device,
        include_accuracy=False,
    )
    accuracy_batches = _countdown_accuracy_batches(config, val_corpus, tokenizer)
    accuracy_metrics = countdown_accuracy_for_batches(
        config=config,
        model=model,
        batches=accuracy_batches,
        tokenizer=tokenizer,
        device=device,
        prefix="val",
    )
    loss_metrics.update(accuracy_metrics)
    diagnostic_batches = sequential_batches(
        val_corpus,
        tokenizer,
        batch_size=config.training.micro_batch_size,
        block_size=config.model.block_size,
        num_pause_tokens=config.data.num_pause_tokens,
        num_batches=config.evaluation.diagnostic_batches,
    )
    diagnostic_loss_metrics = evaluate_batches(
        config=config,
        model=model,
        predictor=predictor,
        batches=diagnostic_batches,
        tokenizer=tokenizer,
        device=device,
        include_accuracy=False,
    )
    representations = representation_diagnostics(
        config=config,
        model=model,
        batches=diagnostic_batches,
        tokenizer=tokenizer,
        device=device,
    )
    generation, samples = generation_diagnostics(
        config=config,
        model=model,
        val_corpus=val_corpus,
        tokenizer=tokenizer,
        device=device,
    )
    generalization = None
    generalization_corpus = load_generalization_corpus(config.data)
    if generalization_corpus is not None:
        generalization_batches = _countdown_accuracy_batches(
            config,
            generalization_corpus,
            tokenizer,
        )
        generalization = countdown_accuracy_for_batches(
            config=config,
            model=model,
            batches=generalization_batches,
            tokenizer=tokenizer,
            device=device,
            prefix="generalization",
        )
    result = {
        "step": int(checkpoint["step"]),
        "checkpoint": checkpoint_name,
        "variant": config.model.variant,
        "transition_target": transition_target_for_variant(
            config.model.variant,
            config.objective.transition,
        ),
        "transition_horizon": config.objective.transition.horizon,
        "lambda_transition": config.objective.transition.lambda_transition,
        "lambda_kl": config.objective.transition.lambda_kl,
        "lambda_ce": config.objective.transition.lambda_ce,
        "parameters": count_parameters(model, predictor),
        "protocol": {
            "config_source": "checkpoint",
            "loss_source": "training.eval_batches",
            "loss_batches": config.training.eval_batches,
            "accuracy_source": "evaluation.accuracy_batches",
            "accuracy_batches": config.evaluation.accuracy_batches,
            "accuracy_sequences": loss_metrics.get("val_sequences"),
            "diagnostic_source": "evaluation.diagnostic_batches",
            "diagnostic_batches": config.evaluation.diagnostic_batches,
            "checkpoint_selection_metric": config.evaluation.checkpoint_metric,
            "checkpoint_selection_mode": config.evaluation.checkpoint_mode,
        },
        "loss": loss_metrics,
        "diagnostic_loss": diagnostic_loss_metrics,
        "representations": representations,
        "generation": generation,
    }
    if generalization is not None:
        result["generalization"] = generalization
    write_json(artifacts.evaluation_path, result)
    if artifacts.samples_path.exists():
        artifacts.samples_path.unlink()
    for sample in samples:
        append_jsonl(artifacts.samples_path, sample)
    return result
