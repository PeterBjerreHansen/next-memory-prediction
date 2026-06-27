from __future__ import annotations

import json
from pathlib import Path

from nmp.artifacts import append_jsonl, artifacts_for, write_json
from nmp.config import ExperimentConfig, save_config
from nmp.experiment_plan import (
    expand_plan,
    load_experiment_plan,
    write_expanded_runs,
)
from nmp.experiments import summarize_experiment


def _synthetic_score(variant: str, lambda_transition: float | None, seed: int):
    if variant == "transformer_ntp":
        return 0.25 + seed * 0.01
    if variant == "memory_tape_ntp":
        return 0.35 + seed * 0.01
    if variant == "memory_tape_nmp":
        values = {
            0.1: (0.20, 0.10, 0.10),
            0.3: (0.80, 0.80, 0.80),
            1.0: (0.50, 0.50, 0.50),
            3.0: (0.30, 0.30, 0.30),
        }
        return values[lambda_transition][seed]
    if variant == "memory_tape_hidden_transition_kl":
        values = {
            0.1: (0.30, 0.30, 0.30),
            0.3: (0.85, 0.85, 0.85),
            1.0: (0.60, 0.60, 0.60),
            3.0: (0.40, 0.40, 0.40),
        }
        return values[lambda_transition][seed]
    values = {
        0.1: (0.40, 0.40, 0.40),
        0.3: (0.50, 0.50, 0.50),
        1.0: (0.90, 0.90, 0.90),
        3.0: (0.20, 0.20, 0.20),
    }
    return values[lambda_transition][seed]


def _synthetic_nll(variant: str, lambda_transition: float | None, seed: int):
    if variant == "transformer_ntp":
        return 3.0 + seed * 0.01
    if variant == "memory_tape_ntp":
        return 2.8 + seed * 0.01
    if variant == "memory_tape_nmp":
        values = {0.1: 3.2, 0.3: 2.0, 1.0: 2.5, 3.0: 2.9}
        return values[lambda_transition] + seed * 0.01
    if variant == "memory_tape_hidden_transition_kl":
        values = {0.1: 2.7, 0.3: 1.9, 1.0: 2.2, 3.0: 2.6}
        return values[lambda_transition] + seed * 0.01
    values = {0.1: 2.4, 0.3: 2.2, 1.0: 1.8, 3.0: 2.7}
    return values[lambda_transition] + seed * 0.01


def _write_synthetic_run(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
    lambda_transition: float | None,
    best_score: float,
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
                    ),
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
    compat_score = min(1.0, best_score + 0.05)
    append_jsonl(
        artifacts.metrics_path,
        {
            "event": "validation",
            "step": 1,
            "final_pass_nll": best_nll,
            "val_accuracy": best_score,
            "val_strict_multiset_accuracy": best_score,
            "val_nextlat_compat_accuracy": compat_score,
            "val_valid_equation_1": best_score,
            "val_valid_equation_2": best_score,
            "val_valid_equation_3": best_score,
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
        "memory_tape_hidden_transition_kl",
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
                "accuracy_source": "evaluation.accuracy_batches",
                "accuracy_batches": None,
                "accuracy_sequences": 10_000,
                "diagnostic_source": "evaluation.diagnostic_batches",
                "diagnostic_batches": 8,
                "checkpoint_selection_metric": "final_pass_nll",
                "checkpoint_selection_mode": "min",
            },
            "loss": {
                "final_pass_nll": best_nll + 0.05,
                "val_accuracy": best_score,
                "val_strict_multiset_accuracy": best_score,
                "val_nextlat_compat_accuracy": compat_score,
                "val_valid_equation_1": best_score,
                "val_valid_equation_2": best_score,
                "val_valid_equation_3": best_score,
                "perplexity": 20.0 + best_nll,
                "pass_nlls": pass_nlls,
                "transition_prediction_loss": 0.2 if transition else None,
                "transition_kl_loss": (
                    0.05 if variant == "memory_tape_hidden_transition_kl" else None
                ),
            },
            "diagnostic_loss": {
                "final_pass_nll": best_nll + 9.0,
                "perplexity": 200.0 + best_nll,
                "pass_nlls": [value + 9.0 for value in pass_nlls],
                "transition_prediction_loss": 9.0 if transition else None,
                "transition_kl_loss": (
                    0.5 if variant == "memory_tape_hidden_transition_kl" else None
                ),
            },
            "parameters": {
                "model": 100,
                "training_only": 10 if transition else 0,
                "total_training": 110 if transition else 100,
            },
            "generalization": {
                "generalization_accuracy": best_score / 2,
                "generalization_strict_multiset_accuracy": best_score / 2,
                "generalization_nextlat_compat_accuracy": compat_score / 2,
            },
            "generation": {
                "sample_accuracy": best_score,
                "nextlat_compat_sample_accuracy": compat_score,
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
    plan = load_experiment_plan("configs/experiments/round1_development.yaml")
    expanded = expand_plan(plan, runs_root=runs_root)
    expanded_path = tmp_path / "expanded_runs.jsonl"
    write_expanded_runs(expanded_path, expanded)
    for run in expanded:
        spec = run.spec
        _write_synthetic_run(
            spec.run_dir,
            variant=spec.variant,
            seed=spec.seed,
            lambda_transition=spec.lambda_transition,
            best_score=_synthetic_score(
                spec.variant,
                spec.lambda_transition,
                spec.seed,
            ),
            best_nll=_synthetic_nll(
                spec.variant,
                spec.lambda_transition,
                spec.seed,
            ),
        )

    output_dir = tmp_path / "summary"
    result = summarize_experiment(
        expanded_runs=expanded_path,
        output_dir=output_dir,
        selection_metric="final_pass_nll",
        selection_mode="min",
    )

    assert result["completed_runs"] == 14
    assert result["selected_lambdas"] == {
        "memory_tape_nmp": 0.3,
        "memory_tape_hidden_transition": 1.0,
        "memory_tape_hidden_transition_kl": 0.3,
    }
    assert len(result["all_condition_summary"]) == 14
    assert len(result["condition_summary"]) == 5
    assert len(result["paired_comparisons"]) == 5
    for filename in (
        "summary.json",
        "selected_lambdas.json",
        "runs.csv",
        "summary.md",
        "comparison.png",
    ):
        assert (output_dir / filename).exists()
    payload = json.loads((output_dir / "summary.json").read_text())
    assert payload["selection_criterion"].startswith("min mean")
    transformer_run = next(
        row for row in payload["runs"] if row["variant"] == "transformer_ntp"
    )
    assert transformer_run["final_pass_nll"] == 3.0
    assert transformer_run["evaluation_final_pass_nll"] == 3.05
    assert transformer_run["diagnostic_final_pass_nll"] == 12.0
    assert transformer_run["val_accuracy"] == 0.25
    assert transformer_run["val_strict_multiset_accuracy"] == 0.25
    assert transformer_run["val_nextlat_compat_accuracy"] == 0.30


def test_reference_manifest_has_four_runs_for_seed_zero():
    plan = load_experiment_plan(
        "configs/experiments/round1_reference_template.yaml"
    )
    expanded = expand_plan(
        plan,
        runs_root="runs",
        selected_lambdas={
            "memory_tape_nmp": 0.3,
            "memory_tape_hidden_transition": 1.0,
            "memory_tape_hidden_transition_kl": 0.3,
        },
    )
    assert len(expanded) == 5


def test_summary_can_leave_lambda_sweeps_unselected(tmp_path: Path):
    runs_root = tmp_path / "runs"
    plan = load_experiment_plan("configs/experiments/round1_development.yaml")
    expanded = expand_plan(plan, runs_root=runs_root)
    expanded_path = tmp_path / "expanded_runs.jsonl"
    write_expanded_runs(expanded_path, expanded)
    for run in expanded:
        spec = run.spec
        _write_synthetic_run(
            spec.run_dir,
            variant=spec.variant,
            seed=spec.seed,
            lambda_transition=spec.lambda_transition,
            best_score=_synthetic_score(
                spec.variant,
                spec.lambda_transition,
                spec.seed,
            ),
            best_nll=_synthetic_nll(
                spec.variant,
                spec.lambda_transition,
                spec.seed,
            ),
        )

    result = summarize_experiment(
        expanded_runs=expanded_path,
        output_dir=tmp_path / "summary",
        selection_metric="final_pass_nll",
        selection_mode="min",
        select_lambda_per_variant=False,
    )

    assert result["selected_lambdas"] == {}
    assert len(result["condition_summary"]) == 14
    assert result["paired_comparisons"] == []
