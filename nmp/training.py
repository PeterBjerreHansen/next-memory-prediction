from __future__ import annotations

from pathlib import Path
import time

import torch

from .artifacts import append_jsonl, prepare_run
from .checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from .config import ExperimentConfig, transition_target_for_variant
from .data import (
    StatefulBatchStream,
    load_corpora,
    make_tokenizer,
    sequential_batches,
)
from .evaluation import countdown_accuracy_for_batches, evaluate_batches
from .factory import build_model, count_parameters, trainable_parameters
from .objectives import compute_loss
from .runtime import (
    autocast_context,
    make_grad_scaler,
    resolve_device,
    set_seed,
    synchronize,
)


def _checkpoint_metric_requires_accuracy(metric: str) -> bool:
    return metric.startswith("val_")


def _optimizer_to(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def train_experiment(
    config: ExperimentConfig,
    *,
    run_dir: str | Path,
    resume_from: str | Path | None = None,
) -> Path:
    config.validate()
    device = resolve_device(config.training.device)
    set_seed(config.seed)
    tokenizer = make_tokenizer(config.data)
    train_corpus, val_corpus = load_corpora(config.data)
    stream = StatefulBatchStream(
        train_corpus,
        tokenizer,
        batch_size=config.training.micro_batch_size,
        block_size=config.model.block_size,
        num_pause_tokens=config.data.num_pause_tokens,
        seed=config.seed + 101,
    )
    model, predictor = build_model(config, vocab_size=tokenizer.vocab_size)
    model.to(device)
    if predictor is not None:
        predictor.to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameters(model, predictor),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )
    scaler = make_grad_scaler(device, config.training.precision)
    artifacts = prepare_run(run_dir, config, fresh=resume_from is None)

    step = 0
    best_final_pass_nll = float("inf")
    best_selection_metric = (
        -float("inf")
        if config.evaluation.checkpoint_mode == "max"
        else float("inf")
    )
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, map_location=device)
        restore_checkpoint(
            checkpoint,
            model=model,
            predictor=predictor,
            optimizer=optimizer,
            scaler=scaler,
            sampler=stream,
        )
        _optimizer_to(optimizer, device)
        step = int(checkpoint["step"])
        best_final_pass_nll = float(checkpoint["best_final_pass_nll"])
        saved_best = checkpoint.get("best_selection_metric")
        if saved_best is not None:
            best_selection_metric = float(saved_best)

    runtime_model = torch.compile(model) if config.training.compile else model
    runtime_predictor = (
        torch.compile(predictor)
        if config.training.compile and predictor is not None
        else predictor
    )
    parameter_counts = count_parameters(model, predictor)
    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "run_start" if step == 0 else "run_resume",
            "step": step,
            "variant": config.model.variant,
            "transition_target": transition_target_for_variant(
                config.model.variant,
                config.objective.transition,
            ),
            "transition_horizon": config.objective.transition.horizon,
            "lambda_transition": config.objective.transition.lambda_transition,
            "lambda_kl": config.objective.transition.lambda_kl,
            "lambda_ce": config.objective.transition.lambda_ce,
            "ntp_pass_weights": config.objective.ntp_pass_weights,
            "device": str(device),
            "effective_batch_size": config.training.effective_batch_size,
            "parameters": parameter_counts,
        },
    )

    validation_batches = sequential_batches(
        val_corpus,
        tokenizer,
        batch_size=config.training.micro_batch_size,
        block_size=config.model.block_size,
        num_pause_tokens=config.data.num_pause_tokens,
        num_batches=config.training.eval_batches,
    )
    accuracy_diagnostic_batches = None
    if config.evaluation.training_accuracy_interval is not None:
        accuracy_diagnostic_batches = sequential_batches(
            val_corpus,
            tokenizer,
            batch_size=config.training.micro_batch_size,
            block_size=config.model.block_size,
            num_pause_tokens=config.data.num_pause_tokens,
            num_batches=config.evaluation.training_accuracy_batches,
        )
    train_window_start = time.perf_counter()
    run_start = train_window_start
    train_window_tokens = 0
    train_window_steps = 0

    while step < config.training.train_steps:
        runtime_model.train()
        if runtime_predictor is not None:
            runtime_predictor.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = None

        for _ in range(config.training.gradient_accumulation_steps):
            batch = stream.next_batch().to(device)
            train_window_tokens += int(
                (
                    batch.target_mask
                    & (batch.tokens[:, 1:] != tokenizer.pad_id)
                )
                .sum()
                .item()
            )
            with autocast_context(device, config.training.precision):
                output = runtime_model(batch.tokens)
                losses = compute_loss(
                    variant=config.model.variant,
                    model=model,
                    output=output,
                    tokens=batch.tokens,
                    target_mask=batch.target_mask,
                    pad_token_id=tokenizer.pad_id,
                    eos_token_id=tokenizer.eos_id,
                    predictor=runtime_predictor,
                    lambda_transition=config.objective.transition.lambda_transition,
                    lambda_kl=config.objective.transition.lambda_kl,
                    lambda_ce=config.objective.transition.lambda_ce,
                    transition_horizon=config.objective.transition.horizon,
                    transition_target=transition_target_for_variant(
                        config.model.variant,
                        config.objective.transition,
                    ),
                    ntp_pass_weights=config.objective.ntp_pass_weights,
                )
                scaled_loss = (
                    losses.total / config.training.gradient_accumulation_steps
                )
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            metrics = losses.detached_metrics()
            if accumulated is None:
                accumulated = metrics
            else:
                for key in ("loss", "weighted_ntp_loss", "final_pass_nll"):
                    accumulated[key] += metrics[key]
                accumulated["pass_nlls"] = [
                    left + right
                    for left, right in zip(
                        accumulated["pass_nlls"],
                        metrics["pass_nlls"],
                    )
                ]
                if metrics["transition_prediction_loss"] is not None:
                    accumulated["transition_prediction_loss"] += metrics[
                        "transition_prediction_loss"
                    ]
                if metrics["transition_kl_loss"] is not None:
                    accumulated["transition_kl_loss"] += metrics[
                        "transition_kl_loss"
                    ]
                if metrics["transition_ce_loss"] is not None:
                    accumulated["transition_ce_loss"] += metrics[
                        "transition_ce_loss"
                    ]

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model, predictor),
            config.training.grad_clip,
        )
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

        step += 1
        train_window_steps += 1
        divisor = config.training.gradient_accumulation_steps
        for key in ("loss", "weighted_ntp_loss", "final_pass_nll"):
            accumulated[key] /= divisor
        accumulated["pass_nlls"] = [
            value / divisor for value in accumulated["pass_nlls"]
        ]
        if accumulated["transition_prediction_loss"] is not None:
            accumulated["transition_prediction_loss"] /= divisor
        if accumulated["transition_kl_loss"] is not None:
            accumulated["transition_kl_loss"] /= divisor
        if accumulated["transition_ce_loss"] is not None:
            accumulated["transition_ce_loss"] /= divisor

        if step % config.training.log_interval == 0 or step == 1:
            synchronize(device)
            elapsed = time.perf_counter() - train_window_start
            append_jsonl(
                artifacts.metrics_path,
                {
                    "event": "train",
                    "step": step,
                    **accumulated,
                    "tokens_per_second": (
                        train_window_tokens / elapsed if elapsed > 0 else 0.0
                    ),
                    "seconds_per_step": (
                        elapsed / train_window_steps
                        if train_window_steps > 0
                        else 0.0
                    ),
                },
            )
            train_window_start = time.perf_counter()
            train_window_tokens = 0
            train_window_steps = 0

        should_eval = (
            step % config.training.eval_interval == 0
            or step == config.training.train_steps
        )
        if should_eval:
            metrics = evaluate_batches(
                config=config,
                model=model,
                predictor=predictor,
                batches=validation_batches,
                tokenizer=tokenizer,
                device=device,
                include_accuracy=_checkpoint_metric_requires_accuracy(
                    config.evaluation.checkpoint_metric
                ),
            )
            append_jsonl(
                artifacts.metrics_path,
                {"event": "validation", "step": step, **metrics},
            )
            if metrics["final_pass_nll"] < best_final_pass_nll:
                best_final_pass_nll = metrics["final_pass_nll"]
            selection_score = float(metrics[config.evaluation.checkpoint_metric])
            improved = (
                selection_score > best_selection_metric
                if config.evaluation.checkpoint_mode == "max"
                else selection_score < best_selection_metric
            )
            if improved:
                best_selection_metric = selection_score
            save_checkpoint(
                artifacts.latest_checkpoint,
                config=config,
                model=model,
                predictor=predictor,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                best_final_pass_nll=best_final_pass_nll,
                sampler_state=stream.state_dict(),
                best_selection_metric=best_selection_metric,
                selection_metric=config.evaluation.checkpoint_metric,
                selection_mode=config.evaluation.checkpoint_mode,
            )
            if improved:
                save_checkpoint(
                    artifacts.best_checkpoint,
                    config=config,
                    model=model,
                    predictor=predictor,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    best_final_pass_nll=best_final_pass_nll,
                    sampler_state=stream.state_dict(),
                    best_selection_metric=best_selection_metric,
                    selection_metric=config.evaluation.checkpoint_metric,
                    selection_mode=config.evaluation.checkpoint_mode,
                )
        elif step % config.training.checkpoint_interval == 0:
            save_checkpoint(
                artifacts.latest_checkpoint,
                config=config,
                model=model,
                predictor=predictor,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                best_final_pass_nll=best_final_pass_nll,
                sampler_state=stream.state_dict(),
                best_selection_metric=best_selection_metric,
                selection_metric=config.evaluation.checkpoint_metric,
                selection_mode=config.evaluation.checkpoint_mode,
            )

        accuracy_interval = config.evaluation.training_accuracy_interval
        should_accuracy_diagnostic = (
            accuracy_interval is not None
            and accuracy_diagnostic_batches is not None
            and (step % accuracy_interval == 0 or step == config.training.train_steps)
            and not (
                should_eval
                and _checkpoint_metric_requires_accuracy(
                    config.evaluation.checkpoint_metric
                )
            )
        )
        if should_accuracy_diagnostic:
            accuracy_metrics = countdown_accuracy_for_batches(
                config=config,
                model=model,
                batches=accuracy_diagnostic_batches,
                tokenizer=tokenizer,
                device=device,
                prefix="val",
            )
            append_jsonl(
                artifacts.metrics_path,
                {
                    "event": "accuracy_diagnostic",
                    "step": step,
                    "accuracy_batches": config.evaluation.training_accuracy_batches,
                    **accuracy_metrics,
                },
            )

    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "run_end",
            "step": step,
            "best_final_pass_nll": best_final_pass_nll,
            "best_selection_metric": best_selection_metric,
            "selection_metric": config.evaluation.checkpoint_metric,
            "selection_mode": config.evaluation.checkpoint_mode,
            "wall_time_seconds": time.perf_counter() - run_start,
        },
    )
    return artifacts.run_dir
