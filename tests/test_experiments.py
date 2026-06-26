from __future__ import annotations

import json
from pathlib import Path

from nmp.artifacts import append_jsonl, artifacts_for, write_json
from nmp.config import ExperimentConfig, save_config
from nmp.experiments import (
    expected_run_specs,
    run_directory,
    summarize_round1,
)


def _synthetic_score(variant: str, lambda_transition: float | None, seed: int):
    if variant == "transformer_ntp":
        return 3.0 + seed * 0.01
    if variant == "memory_tape_ntp":
        return 2.8 + seed * 0.01
    if variant == "memory_tape_nmp":
        values = {
            0.1: (1.0, 10.0, 10.0),
            0.3: (2.0, 2.0, 2.0),
            1.0: (3.0, 3.0, 3.0),
            3.0: (4.0, 4.0, 4.0),
        }
        return values[lambda_transition][seed]
    values = {
        0.1: (1.0, 5.0, 5.0),
        0.3: (2.5, 2.5, 2.5),
        1.0: (1.5, 1.5, 1.5),
        3.0: (3.0, 3.0, 3.0),
    }
    return values[lambda_transition][seed]


def _write_synthetic_run(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
    lambda_transition: float | None,
    best_nll: float,
) -> None:
    config = ExperimentConfig.from_dict(
        {
            "name": "synthetic",
            "seed": seed,
            "model": {
                "variant": variant,
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "memory": {"n_pass": 2},
            },
            "objective": {
                "transition": {
                    "lambda_transition": (
                        1.0
                        if lambda_transition is None
                        else lambda_transition
                    )
                }
            },
            "training": {"train_steps": 1, "micro_batch_size": 1},
        }
    )
    artifacts = artifacts_for(run_dir)
    artifacts.plots_dir.mkdir(parents=True, exist_ok=True)
    save_config(artifacts.config_path, config)
    artifacts.best_checkpoint.write_bytes(b"synthetic")
    (artifacts.plots_dir / "training.png").write_bytes(b"png")
    (artifacts.plots_dir / "probes.png").write_bytes(b"png")
    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "train",
            "step": 1,
            "final_pass_nll": best_nll + 1.0,
            "tokens_per_second": 100.0 + seed,
        },
    )
    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "validation",
            "step": 1,
            "final_pass_nll": best_nll,
            "loss": 0.0 if lambda_transition == 0.1 else 100.0,
        },
    )
    append_jsonl(
        artifacts.metrics_path,
        {"event": "run_end", "step": 1, "wall_time_seconds": 10.0 + seed},
    )
    transition = variant in {
        "memory_tape_nmp",
        "memory_tape_hidden_transition",
    }
    pass_nlls = [best_nll + 0.05]
    if variant != "transformer_ntp":
        pass_nlls = [best_nll + 0.1, best_nll + 0.05]
    write_json(
        artifacts.evaluation_path,
        {
            "protocol": {
                "config_source": "checkpoint",
                "loss_source": "training.eval_batches",
                "loss_batches": 20,
                "diagnostic_source": "evaluation.diagnostic_batches",
                "diagnostic_batches": 8,
                "checkpoint_selection_metric": "final_pass_nll",
            },
            "loss": {
                "final_pass_nll": best_nll + 0.05,
                "perplexity": 20.0 + best_nll,
                "pass_nlls": pass_nlls,
                "transition_prediction_loss": 0.2 if transition else None,
            },
            "diagnostic_loss": {
                "final_pass_nll": best_nll + 9.0,
                "perplexity": 200.0 + best_nll,
                "pass_nlls": [value + 9.0 for value in pass_nlls],
                "transition_prediction_loss": 9.0 if transition else None,
            },
            "parameters": {
                "model": 100,
                "training_only": 10 if transition else 0,
                "total_training": 110 if transition else 100,
            },
            "generation": {
                "recompute_final_pass_agreement": (
                    None if variant == "transformer_ntp" else 0.75
                )
            },
            "representations": {"hidden_mean_norm": 1.0},
        },
    )
    append_jsonl(
        artifacts.probe_metrics_path,
        {
            "event": "probe_validation",
            "source": "hidden",
            "offset": 1,
            "cross_entropy": best_nll,
            "accuracy": 0.1,
            "tokens": 20,
        },
    )
    if variant != "transformer_ntp":
        append_jsonl(
            artifacts.probe_metrics_path,
            {
                "event": "probe_validation",
                "source": "memory",
                "offset": 1,
                "cross_entropy": best_nll + 0.1,
                "accuracy": 0.09,
                "tokens": 20,
            },
        )


def test_development_summary_selects_mean_best_checkpoint_nll(tmp_path: Path):
    runs_root = tmp_path / "runs"
    for spec in expected_run_specs("development"):
        _write_synthetic_run(
            run_directory(runs_root, "development", spec),
            variant=spec.variant,
            seed=spec.seed,
            lambda_transition=spec.lambda_transition,
            best_nll=_synthetic_score(
                spec.variant,
                spec.lambda_transition,
                spec.seed,
            ),
        )

    output_dir = tmp_path / "summary"
    result = summarize_round1(
        runs_root=runs_root,
        scale="development",
        output_dir=output_dir,
    )

    assert result["completed_runs"] == 30
    assert result["selected_lambdas"] == {
        "memory_tape_nmp": 0.3,
        "memory_tape_hidden_transition": 1.0,
    }
    assert len(result["all_condition_summary"]) == 10
    assert len(result["condition_summary"]) == 4
    assert len(result["paired_comparisons"]) == 4
    for filename in (
        "summary.json",
        "selected_lambdas.json",
        "runs.csv",
        "summary.md",
        "comparison.png",
    ):
        assert (output_dir / filename).exists()
    payload = json.loads((output_dir / "summary.json").read_text())
    assert payload["selection_criterion"].startswith("lowest mean")
    transformer_run = next(
        row for row in payload["runs"] if row["variant"] == "transformer_ntp"
    )
    assert transformer_run["final_pass_nll"] == 3.0
    assert transformer_run["evaluation_final_pass_nll"] == 3.05
    assert transformer_run["diagnostic_final_pass_nll"] == 12.0


def test_reference_manifest_has_twelve_runs():
    specs = expected_run_specs(
        "reference",
        selected_lambdas={
            "memory_tape_nmp": 0.3,
            "memory_tape_hidden_transition": 1.0,
        },
    )
    assert len(specs) == 12
