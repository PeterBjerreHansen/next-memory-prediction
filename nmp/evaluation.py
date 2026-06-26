from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .artifacts import append_jsonl, artifacts_for, write_json
from .checkpoint import load_checkpoint, restore_checkpoint
from .config import (
    ExperimentConfig,
    load_config,
    transition_target_for_variant,
)
from .data import (
    TinyStoriesTokenizer,
    load_corpora,
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
from .models import MemoryTapeOutput, MemoryTapeTransformer
from .objectives import compute_loss
from .runtime import autocast_context, resolve_device


@torch.no_grad()
def evaluate_batches(
    *,
    config: ExperimentConfig,
    model,
    predictor,
    batches,
    tokenizer: TinyStoriesTokenizer,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    if predictor is not None:
        predictor.eval()
    totals: dict[str, float] = defaultdict(float)
    pass_totals: list[float] | None = None
    ntp_token_count = 0
    transition_count = 0
    for cpu_batch in batches:
        tokens = cpu_batch.tokens.to(device)
        with autocast_context(device, config.training.precision):
            output = model(tokens)
            losses = compute_loss(
                variant=config.model.variant,
                model=model,
                output=output,
                tokens=tokens,
                pad_token_id=tokenizer.pad_id,
                eos_token_id=tokenizer.eos_id,
                predictor=predictor,
                lambda_transition=config.objective.lambda_transition,
                ntp_pass_weights=config.objective.ntp_pass_weights,
            )
        metrics = losses.detached_metrics()
        batch_ntp_tokens = int(
            (tokens[:, 1:] != tokenizer.pad_id).sum().item()
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
        transition_prediction_loss = (
            totals["transition_prediction_loss"] / transition_count
            if transition_count
            else 0.0
        )
        result["transition_prediction_loss"] = transition_prediction_loss
        result["transition_target"] = transition_target_for_variant(
            config.model.variant
        )
        result["transition_count"] = transition_count
        result["lambda_transition"] = config.objective.lambda_transition
        result["loss"] += (
            config.objective.lambda_transition * transition_prediction_loss
        )
    return result


@torch.no_grad()
def representation_diagnostics(
    *,
    config: ExperimentConfig,
    model,
    batches,
    tokenizer: TinyStoriesTokenizer,
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
    tokenizer: TinyStoriesTokenizer,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = []
    agreement_counts = [0] * config.evaluation.generation_tokens
    agreement_totals = [0] * config.evaluation.generation_tokens

    for index in range(min(config.evaluation.generation_prompts, len(val_corpus))):
        encoded = tokenizer.encode(val_corpus[index])
        if not encoded:
            continue
        prompt_ids = encoded[: config.evaluation.prompt_tokens]
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        recompute = model.generate(
            prompt.clone(),
            config.evaluation.generation_tokens,
            do_sample=False,
            inference_mode="recompute",
        )
        modes = {"recompute": recompute}
        if isinstance(model, MemoryTapeTransformer):
            final_pass = model.generate(
                prompt.clone(),
                config.evaluation.generation_tokens,
                do_sample=False,
                inference_mode="final_pass",
            )
            modes["final_pass"] = final_pass
            recompute_suffix = recompute[0, len(prompt_ids) :]
            final_suffix = final_pass[0, len(prompt_ids) :]
            length = min(recompute_suffix.numel(), final_suffix.numel())
            for position in range(length):
                agreement_totals[position] += 1
                agreement_counts[position] += int(
                    recompute_suffix[position] == final_suffix[position]
                )

        row = {
            "index": index,
            "prompt": tokenizer.decode(prompt_ids),
        }
        for mode, generated in modes.items():
            row[mode] = tokenizer.decode(generated[0].tolist())
        samples.append(row)

    agreement_by_position = [
        (
            agreement_counts[index] / agreement_totals[index]
            if agreement_totals[index]
            else None
        )
        for index in range(len(agreement_counts))
    ]
    valid = [value for value in agreement_by_position if value is not None]
    metrics = {
        "recompute_final_pass_agreement": (
            None if not valid else sum(valid) / len(valid)
        ),
        "agreement_by_generated_position": agreement_by_position,
        "disagreement_by_generated_position": [
            None if value is None else 1.0 - value
            for value in agreement_by_position
        ],
    }
    return metrics, samples


def load_run(
    run_dir: str | Path,
    *,
    checkpoint_name: str = "best.pt",
    device_override: str | None = None,
):
    artifacts = artifacts_for(run_dir)
    config = load_config(artifacts.config_path)
    if device_override is not None:
        config.training.device = device_override
    device = resolve_device(config.training.device)
    tokenizer = TinyStoriesTokenizer()
    model, predictor = build_model(config, vocab_size=tokenizer.vocab_size)
    checkpoint_path = artifacts.run_dir / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = artifacts.latest_checkpoint
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    restore_checkpoint(
        checkpoint,
        model=model,
        predictor=predictor,
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
    batches = sequential_batches(
        val_corpus,
        tokenizer,
        batch_size=config.training.micro_batch_size,
        block_size=config.model.block_size,
        num_batches=config.evaluation.diagnostic_batches,
    )
    loss_metrics = evaluate_batches(
        config=config,
        model=model,
        predictor=predictor,
        batches=batches,
        tokenizer=tokenizer,
        device=device,
    )
    representations = representation_diagnostics(
        config=config,
        model=model,
        batches=batches,
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
    result = {
        "step": int(checkpoint["step"]),
        "checkpoint": checkpoint_name,
        "variant": config.model.variant,
        "transition_target": transition_target_for_variant(
            config.model.variant
        ),
        "lambda_transition": config.objective.lambda_transition,
        "parameters": count_parameters(model, predictor),
        "loss": loss_metrics,
        "representations": representations,
        "generation": generation,
    }
    write_json(artifacts.evaluation_path, result)
    if artifacts.samples_path.exists():
        artifacts.samples_path.unlink()
    for sample in samples:
        append_jsonl(artifacts.samples_path, sample)
    return result
