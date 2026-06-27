from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import artifacts_for, read_jsonl, write_json
from .config import (
    TRANSITION_VARIANTS,
    load_config,
    transition_target_for_variant,
)
from .experiment_plan import ExpandedRunSpec, load_expanded_run_specs


BASELINE_VARIANTS = ("transformer_ntp", "memory_tape_ntp")
DEFAULT_VARIANT_ORDER = (*BASELINE_VARIANTS, *TRANSITION_VARIANTS)


def load_selected_lambdas(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("selected_lambdas", payload)
    return {
        variant: float(values[variant])
        for variant in TRANSITION_VARIANTS
    }


def _variant_sort_key(variant: str) -> tuple[int, str]:
    if variant in DEFAULT_VARIANT_ORDER:
        return DEFAULT_VARIANT_ORDER.index(variant), variant
    return len(DEFAULT_VARIANT_ORDER), variant


def _ordered_variant_sort_key(
    variant: str,
    variant_order: list[str] | None,
) -> tuple[int, str]:
    if variant_order is not None and variant in variant_order:
        return variant_order.index(variant), variant
    return _variant_sort_key(variant)


def _require_run_artifacts(run_dir: Path) -> None:
    artifacts = artifacts_for(run_dir)
    required = (
        artifacts.config_path,
        artifacts.metrics_path,
        artifacts.best_checkpoint,
        artifacts.evaluation_path,
        artifacts.probe_metrics_path,
        artifacts.plots_dir / "training.png",
        artifacts.plots_dir / "probes.png",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"incomplete run {run_dir}; missing: " + ", ".join(missing)
        )


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


def _read_evaluation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_validation_row(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
) -> dict[str, Any]:
    validation = [
        row
        for row in rows
        if row.get("event") == "validation"
        and row.get(metric) is not None
    ]
    if not validation and metric != "final_pass_nll":
        return _best_validation_row(rows, metric="final_pass_nll", mode="min")
    if not validation:
        raise ValueError("run contains no validation metrics")
    key = lambda row: float(row[metric])
    return max(validation, key=key) if mode == "max" else min(validation, key=key)


def _metric_with_fallback(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    value = primary.get(key)
    if value is not None:
        return value
    return fallback.get(key, default)


def load_run_record(
    spec: ExpandedRunSpec,
    *,
    selection_metric: str = "final_pass_nll",
    selection_mode: str = "min",
) -> dict[str, Any]:
    run_dir = Path(spec.run_dir)
    _require_run_artifacts(run_dir)
    artifacts = artifacts_for(run_dir)
    config = load_config(artifacts.config_path)
    if config.model.variant != spec.variant or config.seed != spec.seed:
        raise ValueError(
            f"run identity mismatch in {run_dir}: "
            f"expected {spec.variant}/seed {spec.seed}, "
            f"found {config.model.variant}/seed {config.seed}"
        )
    if spec.lambda_transition is not None and (
        float(config.objective.transition.lambda_transition)
        != float(spec.lambda_transition)
    ):
        raise ValueError(f"transition weight mismatch in {run_dir}")

    metrics = read_jsonl(artifacts.metrics_path)
    best_validation = _best_validation_row(
        metrics,
        metric=selection_metric,
        mode=selection_mode,
    )
    evaluation = _read_evaluation(artifacts.evaluation_path)
    evaluation_loss = evaluation["loss"]
    diagnostic_loss = evaluation.get("diagnostic_loss")
    train_rows = [row for row in metrics if row.get("event") == "train"]
    wall_time_seconds = sum(
        float(row["wall_time_seconds"])
        for row in metrics
        if row.get("event") == "run_end"
        and row.get("wall_time_seconds") is not None
    )
    probes = [
        row
        for row in read_jsonl(artifacts.probe_metrics_path)
        if row.get("event") == "probe_validation"
    ]
    transition_loss = best_validation.get("transition_prediction_loss")
    if transition_loss is None:
        transition_loss = evaluation_loss.get("transition_prediction_loss")
    transition_kl_loss = best_validation.get("transition_kl_loss")
    if transition_kl_loss is None:
        transition_kl_loss = evaluation_loss.get("transition_kl_loss")
    transition_ce_loss = best_validation.get("transition_ce_loss")
    if transition_ce_loss is None:
        transition_ce_loss = evaluation_loss.get("transition_ce_loss")
    parameters = evaluation.get("parameters", {})
    generation = evaluation.get("generation", {})
    generalization = evaluation.get("generalization", {})
    final_pass_nll = float(best_validation["final_pass_nll"])
    perplexity = _metric_with_fallback(
        best_validation,
        evaluation_loss,
        "perplexity",
    )
    pass_nlls = _metric_with_fallback(
        best_validation,
        evaluation_loss,
        "pass_nlls",
        [],
    )
    diagnostic_final_pass_nll = (
        None
        if diagnostic_loss is None
        else diagnostic_loss.get("final_pass_nll")
    )
    if diagnostic_final_pass_nll is None and "protocol" not in evaluation:
        diagnostic_final_pass_nll = evaluation_loss.get("final_pass_nll")

    return {
        "experiment": spec.experiment,
        "variant": spec.variant,
        "seed": spec.seed,
        "lambda_transition": spec.lambda_transition,
        "lambda_kl": config.objective.transition.lambda_kl,
        "lambda_ce": config.objective.transition.lambda_ce,
        "transition_horizon": config.objective.transition.horizon,
        "transition_target": transition_target_for_variant(
            spec.variant,
            config.objective.transition,
        ),
        "run_dir": str(run_dir),
        "best_checkpoint_final_pass_nll": final_pass_nll,
        "best_checkpoint_selection_metric": selection_metric,
        "best_checkpoint_selection_mode": selection_mode,
        "best_checkpoint_selection_score": float(
            best_validation.get(selection_metric, final_pass_nll)
        ),
        "best_checkpoint_step": int(best_validation["step"]),
        "final_pass_nll": final_pass_nll,
        "val_accuracy": _metric_with_fallback(
            best_validation,
            evaluation_loss,
            "val_accuracy",
        ),
        "val_strict_multiset_accuracy": _metric_with_fallback(
            best_validation,
            evaluation_loss,
            "val_strict_multiset_accuracy",
            _metric_with_fallback(best_validation, evaluation_loss, "val_accuracy"),
        ),
        "val_nextlat_compat_accuracy": _metric_with_fallback(
            best_validation,
            evaluation_loss,
            "val_nextlat_compat_accuracy",
        ),
        "val_valid_equation_1": _metric_with_fallback(
            best_validation,
            evaluation_loss,
            "val_valid_equation_1",
        ),
        "val_valid_equation_2": _metric_with_fallback(
            best_validation,
            evaluation_loss,
            "val_valid_equation_2",
        ),
        "val_valid_equation_3": _metric_with_fallback(
            best_validation,
            evaluation_loss,
            "val_valid_equation_3",
        ),
        "generalization_accuracy": generalization.get("generalization_accuracy"),
        "generalization_strict_multiset_accuracy": generalization.get(
            "generalization_strict_multiset_accuracy",
            generalization.get("generalization_accuracy"),
        ),
        "generalization_nextlat_compat_accuracy": generalization.get(
            "generalization_nextlat_compat_accuracy"
        ),
        "perplexity": None if perplexity is None else float(perplexity),
        "pass_nlls": list(map(float, pass_nlls)),
        "ntp_pass_weights": _metric_with_fallback(
            best_validation,
            evaluation_loss,
            "ntp_pass_weights",
        ),
        "evaluation_final_pass_nll": float(evaluation_loss["final_pass_nll"]),
        "diagnostic_final_pass_nll": (
            None
            if diagnostic_final_pass_nll is None
            else float(diagnostic_final_pass_nll)
        ),
        "loss_batches": evaluation.get("protocol", {}).get("loss_batches"),
        "diagnostic_batches": evaluation.get("protocol", {}).get(
            "diagnostic_batches"
        ),
        "transition_prediction_loss": (
            None if transition_loss is None else float(transition_loss)
        ),
        "transition_kl_loss": (
            None if transition_kl_loss is None else float(transition_kl_loss)
        ),
        "transition_ce_loss": (
            None if transition_ce_loss is None else float(transition_ce_loss)
        ),
        "parameters_model": int(parameters.get("model", 0)),
        "parameters_training_only": int(parameters.get("training_only", 0)),
        "parameters_total_training": int(parameters.get("total_training", 0)),
        "mean_tokens_per_second": _mean_or_none(
            row.get("tokens_per_second") for row in train_rows
        ),
        "wall_time_seconds": wall_time_seconds,
        "generation_sample_accuracy": generation.get("sample_accuracy"),
        "generation_nextlat_compat_sample_accuracy": generation.get(
            "nextlat_compat_sample_accuracy"
        ),
        "representations": evaluation.get("representations", {}),
        "probes": probes,
    }


def _condition_key(record: dict[str, Any]) -> tuple[str, float | None]:
    return record["variant"], record["lambda_transition"]


def select_development_lambdas(
    records: list[dict[str, Any]],
    *,
    selection_metric: str,
    selection_mode: str,
) -> dict[str, float]:
    selected: dict[str, float] = {}
    variants = sorted(
        {
            record["variant"]
            for record in records
            if record["lambda_transition"] is not None
        },
        key=_variant_sort_key,
    )
    for variant in variants:
        candidates = []
        seed_sets = []
        for lambda_transition in sorted(
            {
                float(record["lambda_transition"])
                for record in records
                if record["variant"] == variant
                and record["lambda_transition"] is not None
            }
        ):
            rows = [
                record
                for record in records
                if record["variant"] == variant
                and record["lambda_transition"] == lambda_transition
            ]
            seed_sets.append({int(row["seed"]) for row in rows})
            mean_score = statistics.mean(
                float(
                    row[selection_metric]
                    if row.get(selection_metric) is not None
                    else row["best_checkpoint_selection_score"]
                )
                for row in rows
            )
            candidates.append((mean_score, lambda_transition))
        if len({tuple(sorted(seeds)) for seeds in seed_sets}) > 1:
            raise ValueError(f"{variant} lambda candidates have unequal seeds")
        selected[variant] = (
            max(candidates)[1] if selection_mode == "max" else min(candidates)[1]
        )
    return selected


def selected_condition_records(
    records: list[dict[str, Any]],
    selected_lambdas: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["lambda_transition"] is None
        or record["variant"] not in selected_lambdas
        or record["lambda_transition"] == selected_lambdas[record["variant"]]
    ]


def has_unselected_lambda_sweeps(
    records: list[dict[str, Any]],
    selected_lambdas: dict[str, float],
) -> bool:
    if selected_lambdas:
        return False
    by_variant: dict[str, set[float]] = {}
    for record in records:
        if record["lambda_transition"] is not None:
            by_variant.setdefault(record["variant"], set()).add(
                float(record["lambda_transition"])
            )
    return any(len(values) > 1 for values in by_variant.values())


def _scalar_statistics(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_conditions(
    records: list[dict[str, Any]],
    *,
    variant_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_condition_key(record), []).append(record)
    rows = []
    scalar_fields = (
        "best_checkpoint_selection_score",
        "val_accuracy",
        "val_strict_multiset_accuracy",
        "val_nextlat_compat_accuracy",
        "val_valid_equation_1",
        "val_valid_equation_2",
        "val_valid_equation_3",
        "generalization_accuracy",
        "generalization_strict_multiset_accuracy",
        "generalization_nextlat_compat_accuracy",
        "best_checkpoint_final_pass_nll",
        "final_pass_nll",
        "evaluation_final_pass_nll",
        "diagnostic_final_pass_nll",
        "perplexity",
        "transition_prediction_loss",
        "transition_kl_loss",
        "transition_ce_loss",
        "parameters_model",
        "parameters_training_only",
        "parameters_total_training",
        "mean_tokens_per_second",
        "wall_time_seconds",
        "generation_sample_accuracy",
        "generation_nextlat_compat_sample_accuracy",
    )
    for (variant, lambda_transition), condition in sorted(
        grouped.items(),
        key=lambda item: (
            _ordered_variant_sort_key(item[0][0], variant_order),
            -1.0 if item[0][1] is None else item[0][1],
        ),
    ):
        summary: dict[str, Any] = {
            "variant": variant,
            "lambda_transition": lambda_transition,
            "lambda_kl": condition[0].get("lambda_kl"),
            "lambda_ce": condition[0].get("lambda_ce"),
            "transition_horizon": condition[0].get("transition_horizon"),
            "transition_target": condition[0].get(
                "transition_target",
                transition_target_for_variant(variant),
            ),
            "ntp_pass_weights": condition[0].get("ntp_pass_weights"),
            "seeds": sorted(record["seed"] for record in condition),
        }
        for field in scalar_fields:
            values = [
                float(record[field])
                for record in condition
                if record[field] is not None
            ]
            summary[field] = _scalar_statistics(values) if values else None
        pass_count = len(condition[0]["pass_nlls"])
        summary["pass_nlls"] = [
            _scalar_statistics(
                [record["pass_nlls"][index] for record in condition]
            )
            for index in range(pass_count)
        ]
        representation_keys = sorted(
            {
                key
                for record in condition
                for key, value in record["representations"].items()
                if isinstance(value, (int, float))
            }
        )
        summary["representations"] = {
            key: _scalar_statistics(
                [
                    float(record["representations"][key])
                    for record in condition
                    if record["representations"].get(key) is not None
                ]
            )
            for key in representation_keys
        }
        rows.append(summary)
    return rows


def summarize_probes(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None, str, int], list[dict[str, Any]]] = {}
    for record in records:
        for probe in record["probes"]:
            key = (
                record["variant"],
                record["lambda_transition"],
                str(probe["source"]),
                int(probe["offset"]),
            )
            grouped.setdefault(key, []).append(probe)
    rows = []
    for (variant, lambda_transition, source, offset), probes in sorted(
        grouped.items()
    ):
        rows.append(
            {
                "variant": variant,
                "lambda_transition": lambda_transition,
                "source": source,
                "offset": offset,
                "cross_entropy": _scalar_statistics(
                    [float(probe["cross_entropy"]) for probe in probes]
                ),
                "accuracy": _scalar_statistics(
                    [float(probe["accuracy"]) for probe in probes]
                ),
            }
        )
    return rows


def paired_comparisons(
    records: list[dict[str, Any]],
    selected_lambdas: dict[str, float],
) -> list[dict[str, Any]]:
    conditions = {
        (record["variant"], record["seed"]): record
        for record in selected_condition_records(records, selected_lambdas)
    }
    comparisons = (
        ("architecture", "transformer_ntp", "memory_tape_ntp"),
        ("memory_transition", "memory_tape_ntp", "memory_tape_nmp"),
        ("hidden_transition", "memory_tape_ntp", "memory_tape_hidden_transition"),
        (
            "explicit_memory_vs_hidden",
            "memory_tape_hidden_transition",
            "memory_tape_nmp",
        ),
        (
            "hidden_kl_distillation",
            "memory_tape_hidden_transition",
            "memory_tape_hidden_transition_kl",
        ),
    )
    results = []
    available_variants = {record["variant"] for record in conditions.values()}
    for name, source, target in comparisons:
        if source not in available_variants or target not in available_variants:
            continue
        source_seeds = {
            seed for variant, seed in conditions if variant == source
        }
        target_seeds = {
            seed for variant, seed in conditions if variant == target
        }
        paired_seeds = sorted(source_seeds & target_seeds)
        if not paired_seeds:
            continue
        seed_rows = []
        for seed in paired_seeds:
            left = conditions[(source, seed)]
            right = conditions[(target, seed)]
            seed_rows.append(
                {
                    "seed": seed,
                    "val_accuracy_delta": (
                        right["val_accuracy"] - left["val_accuracy"]
                    ),
                    "best_checkpoint_final_pass_nll_delta": (
                        right["best_checkpoint_final_pass_nll"]
                        - left["best_checkpoint_final_pass_nll"]
                    ),
                    "final_pass_nll_delta": (
                        right["final_pass_nll"] - left["final_pass_nll"]
                    ),
                    "perplexity_delta": (
                        right["perplexity"] - left["perplexity"]
                    ),
                }
            )
        results.append(
            {
                "name": name,
                "source": source,
                "target": target,
                "definition": (
                    "target minus source; positive accuracy favors target, "
                    "negative loss favors target"
                ),
                "by_seed": seed_rows,
                "val_accuracy_delta": _scalar_statistics(
                    [row["val_accuracy_delta"] for row in seed_rows]
                ),
                "best_checkpoint_final_pass_nll_delta": _scalar_statistics(
                    [
                        row["best_checkpoint_final_pass_nll_delta"]
                        for row in seed_rows
                    ]
                ),
                "final_pass_nll_delta": _scalar_statistics(
                    [row["final_pass_nll_delta"] for row in seed_rows]
                ),
                "perplexity_delta": _scalar_statistics(
                    [row["perplexity_delta"] for row in seed_rows]
                ),
            }
        )
    return results


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = list(records[0]) if records else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: _csv_value(value) for key, value in record.items()}
            )


def write_summary_markdown(
    path: Path,
    *,
    scale: str,
    selected_lambdas: dict[str, float],
    all_condition_summary: list[dict[str, Any]],
    condition_summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    probe_summary: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {scale.replace('_', ' ').title()} Summary",
        "",
        "Validation metrics are reported; no separate held-out test set is used.",
        "Primary loss columns use the checkpoint-selection validation estimate,",
        "while diagnostic loss columns use the smaller diagnostics batch count.",
        "",
        "## Selected transition weights",
        "",
    ]
    for variant in TRANSITION_VARIANTS:
        if variant in selected_lambdas:
            lines.append(f"- `{variant}`: `{selected_lambdas[variant]:g}`")
    lambdas_by_variant: dict[str, set[float]] = {}
    for row in all_condition_summary:
        if row["lambda_transition"] is not None:
            lambdas_by_variant.setdefault(row["variant"], set()).add(
                float(row["lambda_transition"])
            )
    if any(len(values) > 1 for values in lambdas_by_variant.values()):
        lines.extend(
            [
                "",
                "## Transition-weight sweep",
                "",
                "| Variant | λ | Mean best-checkpoint selection score |",
                "|---|---:|---:|",
            ]
        )
        for row in all_condition_summary:
            if row["lambda_transition"] is None:
                continue
            score = row["best_checkpoint_selection_score"]
            lines.append(
                f"| `{row['variant']}` | {row['lambda_transition']:g} | "
                f"{score['mean']:.4f} ± {score['std']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "| Variant | Target | λ | KL λ | CE λ | NTP weights | Strict val | NextLat compat | Generalization | Final-pass NLL | Diagnostic NLL | Transition loss | KL loss | CE loss |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in condition_summary:
        lambda_text = (
            "—"
            if row["lambda_transition"] is None
            else f"{row['lambda_transition']:g}"
        )
        transition = row["transition_prediction_loss"]
        transition_text = (
            "—"
            if transition is None
            else f"{transition['mean']:.4f} ± {transition['std']:.4f}"
        )
        transition_kl = row["transition_kl_loss"]
        transition_kl_text = (
            "—"
            if transition_kl is None
            else f"{transition_kl['mean']:.4f} ± {transition_kl['std']:.4f}"
        )
        transition_ce = row["transition_ce_loss"]
        transition_ce_text = (
            "—"
            if transition_ce is None
            else f"{transition_ce['mean']:.4f} ± {transition_ce['std']:.4f}"
        )
        weights = row.get("ntp_pass_weights")
        weights_text = (
            "—"
            if weights is None
            else ", ".join(f"{float(weight):g}" for weight in weights)
        )
        diagnostic = row.get("diagnostic_final_pass_nll")
        diagnostic_text = (
            "—"
            if diagnostic is None
            else f"{diagnostic['mean']:.4f} ± {diagnostic['std']:.4f}"
        )
        accuracy = row.get("val_accuracy")
        accuracy_text = (
            "—"
            if accuracy is None
            else f"{accuracy['mean']:.3f} ± {accuracy['std']:.3f}"
        )
        generalization = row.get("generalization_accuracy")
        generalization_text = (
            "—"
            if generalization is None
            else f"{generalization['mean']:.3f} ± {generalization['std']:.3f}"
        )
        compat = row.get("val_nextlat_compat_accuracy")
        compat_text = (
            "—"
            if compat is None
            else f"{compat['mean']:.3f} ± {compat['std']:.3f}"
        )
        lambda_kl = row.get("lambda_kl")
        lambda_kl_text = "—" if lambda_kl is None else f"{lambda_kl:g}"
        lambda_ce = row.get("lambda_ce")
        lambda_ce_text = "—" if lambda_ce is None else f"{lambda_ce:g}"
        target_text = row.get("transition_target") or "—"
        lines.append(
            f"| `{row['variant']}` | {target_text} | {lambda_text} | "
            f"{lambda_kl_text} | {lambda_ce_text} | {weights_text} | "
            f"{accuracy_text} | "
            f"{compat_text} | "
            f"{generalization_text} | "
            f"{row['final_pass_nll']['mean']:.4f} ± "
            f"{row['final_pass_nll']['std']:.4f} | "
            f"{diagnostic_text} | "
            f"{transition_text} | {transition_kl_text} | {transition_ce_text} |"
        )
    lines.extend(
        [
            "",
            "## Compute and generation",
            "",
            "| Variant | Model params | Training-only params | Tokens/s | "
            "Sample accuracy | NextLat sample |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in condition_summary:
        throughput = row["mean_tokens_per_second"]
        agreement = row["generation_sample_accuracy"]
        compat_agreement = row["generation_nextlat_compat_sample_accuracy"]
        lines.append(
            f"| `{row['variant']}` | "
            f"{row['parameters_model']['mean']:.0f} | "
            f"{row['parameters_training_only']['mean']:.0f} | "
            f"{throughput['mean']:.1f} ± {throughput['std']:.1f} | "
            + (
                "—"
                if agreement is None
                else f"{agreement['mean']:.3f} ± {agreement['std']:.3f}"
            )
            + " | "
            + (
                "—"
                if compat_agreement is None
                else (
                    f"{compat_agreement['mean']:.3f} ± "
                    f"{compat_agreement['std']:.3f}"
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-pass validation NLL",
            "",
            "| Variant | Pass NLLs |",
            "|---|---|",
        ]
    )
    for row in condition_summary:
        pass_text = ", ".join(
            f"{value['mean']:.4f} ± {value['std']:.4f}"
            for value in row["pass_nlls"]
        )
        lines.append(f"| `{row['variant']}` | {pass_text} |")
    representation_keys = sorted(
        {
            key
            for row in condition_summary
            for key in row["representations"]
        }
    )
    lines.extend(
        [
            "",
            "## Representation diagnostics",
            "",
            "| Variant | Diagnostic | Mean ± std |",
            "|---|---|---:|",
        ]
    )
    for row in condition_summary:
        for key in representation_keys:
            value = row["representations"].get(key)
            if value is not None:
                lines.append(
                    f"| `{row['variant']}` | {key} | "
                    f"{value['mean']:.4f} ± {value['std']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Paired seed deltas",
            "",
            "Deltas are target minus source; positive accuracy and negative loss favor the target.",
            "",
            "| Comparison | Source → target | Val accuracy Δ | Final-pass NLL Δ |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison in comparisons:
        nll = comparison["final_pass_nll_delta"]
        accuracy = comparison["val_accuracy_delta"]
        lines.append(
            f"| {comparison['name']} | `{comparison['source']}` → "
            f"`{comparison['target']}` | "
            f"{accuracy['mean']:.4f} ± {accuracy['std']:.4f} | "
            f"{nll['mean']:.4f} ± {nll['std']:.4f} | "
        )
    lines.extend(
        [
            "",
            "## Linear probes",
            "",
            "| Variant | λ | Source | Offset | Cross-entropy | Accuracy |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in probe_summary:
        lambda_text = (
            "—"
            if row["lambda_transition"] is None
            else f"{row['lambda_transition']:g}"
        )
        lines.append(
            f"| `{row['variant']}` | {lambda_text} | {row['source']} | "
            f"{row['offset']} | {row['cross_entropy']['mean']:.4f} ± "
            f"{row['cross_entropy']['std']:.4f} | "
            f"{row['accuracy']['mean']:.4f} ± "
            f"{row['accuracy']['std']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(
    path: Path,
    condition_summary: list[dict[str, Any]],
) -> None:
    labels = [row["variant"] for row in condition_summary]
    positions = list(range(len(labels)))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, field, title in (
        (axes[0], "val_accuracy", "Countdown validation accuracy"),
        (axes[1], "final_pass_nll", "Final-pass validation NLL"),
    ):
        rows = [row for row in condition_summary if row.get(field) is not None]
        if not rows:
            continue
        positions = list(range(len(rows)))
        axis.errorbar(
            positions,
            [row[field]["mean"] for row in rows],
            yerr=[row[field]["std"] for row in rows],
            fmt="o",
            capsize=4,
        )
        axis.set_xticks(
            positions,
            [row["variant"] for row in rows],
            rotation=20,
            ha="right",
        )
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _variant_order_from_specs(specs: list[ExpandedRunSpec]) -> list[str]:
    order: list[str] = []
    for spec in specs:
        if spec.variant not in order:
            order.append(spec.variant)
    return order


def _selected_lambdas_from_expanded_records(
    records: list[dict[str, Any]],
    *,
    selection_metric: str,
    selection_mode: str,
    select_lambda_per_variant: bool,
) -> dict[str, float]:
    if not select_lambda_per_variant:
        return {}
    by_variant: dict[str, set[float]] = {}
    for record in records:
        value = record["lambda_transition"]
        if value is not None:
            by_variant.setdefault(record["variant"], set()).add(float(value))
    if any(len(values) > 1 for values in by_variant.values()):
        return select_development_lambdas(
            records,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
        )
    return {
        variant: next(iter(values))
        for variant, values in by_variant.items()
        if values
    }


def summarize_experiment(
    *,
    expanded_runs: str | Path,
    output_dir: str | Path,
    selection_file: str | Path | None = None,
    selection_metric: str = "final_pass_nll",
    selection_mode: str = "min",
    select_lambda_per_variant: bool = True,
) -> dict[str, Any]:
    specs = load_expanded_run_specs(expanded_runs)
    if not specs:
        raise ValueError("expanded run manifest is empty")
    records = [
        load_run_record(
            spec,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
        )
        for spec in specs
    ]
    selected_lambdas = _selected_lambdas_from_expanded_records(
        records,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        select_lambda_per_variant=select_lambda_per_variant,
    )
    if selection_file is not None and not selected_lambdas:
        selected_lambdas = load_selected_lambdas(selection_file)
    selected_records = selected_condition_records(records, selected_lambdas)
    variant_order = _variant_order_from_specs(specs)
    all_condition_summary = summarize_conditions(
        records,
        variant_order=variant_order,
    )
    condition_summary = summarize_conditions(
        selected_records,
        variant_order=variant_order,
    )
    probe_summary = summarize_probes(selected_records)
    comparisons = (
        []
        if has_unselected_lambda_sweeps(records, selected_lambdas)
        else paired_comparisons(records, selected_lambdas)
    )
    experiment_name = specs[0].experiment
    result = {
        "experiment": experiment_name,
        "expanded_runs": str(expanded_runs),
        "expected_runs": len(specs),
        "completed_runs": len(records),
        "selection_criterion": (
            f"{selection_mode} mean best-checkpoint {selection_metric} across seeds"
        ),
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "select_lambda_per_variant": select_lambda_per_variant,
        "selected_lambdas": selected_lambdas,
        "runs": records,
        "all_condition_summary": all_condition_summary,
        "condition_summary": condition_summary,
        "paired_comparisons": comparisons,
        "probe_summary": probe_summary,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", result)
    if selected_lambdas:
        write_json(
            output_dir / "selected_lambdas.json",
            {
                "selection_criterion": result["selection_criterion"],
                "selected_lambdas": selected_lambdas,
            },
        )
    write_records_csv(output_dir / "runs.csv", records)
    write_summary_markdown(
        output_dir / "summary.md",
        scale=experiment_name,
        selected_lambdas=selected_lambdas,
        all_condition_summary=all_condition_summary,
        condition_summary=condition_summary,
        comparisons=comparisons,
        probe_summary=probe_summary,
    )
    plot_comparison(output_dir / "comparison.png", condition_summary)
    return result
