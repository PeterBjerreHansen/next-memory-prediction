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
from .config import active_objective_metadata, condition_label, load_config
from .experiment_plan import ExpandedRunSpec, load_expanded_run_specs


DEFAULT_CONDITION_ORDER = (
    "transformer_ntp",
    "memory_tape_ntp",
    "memory_tape_nmp",
    "memory_tape_hidden_transition",
    "memory_tape_hidden_transition_kl",
)


def load_selected_lambdas(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("selected_lambdas", payload)
    return {str(condition): float(value) for condition, value in values.items()}


def _condition_sort_key(condition: str) -> tuple[int, str]:
    if condition in DEFAULT_CONDITION_ORDER:
        return DEFAULT_CONDITION_ORDER.index(condition), condition
    return len(DEFAULT_CONDITION_ORDER), condition


def _ordered_condition_sort_key(
    condition: str,
    condition_order: list[str] | None,
) -> tuple[int, str]:
    if condition_order is not None and condition in condition_order:
        return condition_order.index(condition), condition
    return _condition_sort_key(condition)


def _require_run_artifacts(run_dir: Path) -> None:
    artifacts = artifacts_for(run_dir)
    required = (
        artifacts.config_path,
        artifacts.metrics_path,
        artifacts.best_checkpoint,
        artifacts.evaluation_path,
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
        if row.get("event") == "validation" and row.get(metric) is not None
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


def _copy_prefixed_metrics(
    destination: dict[str, Any],
    primary: dict[str, Any],
    fallback: dict[str, Any],
    prefixes: tuple[str, ...],
) -> None:
    keys = {
        key
        for source in (primary, fallback)
        for key in source
        if any(key.startswith(prefix) for prefix in prefixes)
    }
    for key in sorted(keys):
        destination[key] = _metric_with_fallback(primary, fallback, key)


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
    label = condition_label(config)
    if (
        label != spec.variant
        or config.model.architecture != spec.architecture
        or config.objective.transition != spec.transition
        or config.seed != spec.seed
    ):
        raise ValueError(
            f"run identity mismatch in {run_dir}: expected "
            f"{spec.variant}/{spec.architecture}/{spec.transition}/seed {spec.seed}, "
            f"found {label}/{config.model.architecture}/"
            f"{config.objective.transition}/seed {config.seed}"
        )
    if spec.lambda_transition is not None and (
        float(config.objective.lambda_transition) != float(spec.lambda_transition)
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
    transition_loss = best_validation.get("transition_prediction_loss")
    if transition_loss is None:
        transition_loss = evaluation_loss.get("transition_prediction_loss")
    transition_kl_loss = best_validation.get("transition_kl_loss")
    if transition_kl_loss is None:
        transition_kl_loss = evaluation_loss.get("transition_kl_loss")
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
        None if diagnostic_loss is None else diagnostic_loss.get("final_pass_nll")
    )
    objective_metadata = active_objective_metadata(config.objective)

    record: dict[str, Any] = {
        "experiment": spec.experiment,
        "variant": spec.variant,
        "condition": spec.condition,
        "architecture": spec.architecture,
        "transition": spec.transition,
        "seed": spec.seed,
        "lambda_transition": objective_metadata["lambda_transition"],
        "lambda_kl": objective_metadata["lambda_kl"],
        "transition_target": objective_metadata["transition_target"],
        "run_dir": str(run_dir),
        "best_checkpoint_final_pass_nll": final_pass_nll,
        "best_checkpoint_selection_metric": selection_metric,
        "best_checkpoint_selection_mode": selection_mode,
        "best_checkpoint_selection_score": float(
            best_validation.get(selection_metric, final_pass_nll)
        ),
        "best_checkpoint_step": int(best_validation["step"]),
        "final_pass_nll": final_pass_nll,
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
    }
    _copy_prefixed_metrics(
        record,
        best_validation,
        evaluation_loss,
        prefixes=("val_",),
    )
    for key, value in generalization.items():
        record[key] = value
    return record


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
        key=_condition_sort_key,
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


def _scalar_summary_fields(records: list[dict[str, Any]]) -> list[str]:
    excluded = {
        "experiment",
        "variant",
        "condition",
        "architecture",
        "transition",
        "transition_target",
        "run_dir",
        "seed",
        "lambda_transition",
        "lambda_kl",
        "pass_nlls",
        "ntp_pass_weights",
        "representations",
    }
    fields = sorted(
        {
            key
            for record in records
            for key, value in record.items()
            if key not in excluded and isinstance(value, (int, float))
        }
    )
    preferred = [
        "best_checkpoint_selection_score",
        "val_accuracy",
        "val_strict_multiset_accuracy",
        "val_nextlat_compat_accuracy",
        "final_pass_nll",
        "evaluation_final_pass_nll",
        "diagnostic_final_pass_nll",
        "perplexity",
        "transition_prediction_loss",
        "transition_kl_loss",
        "parameters_model",
        "parameters_training_only",
        "parameters_total_training",
        "mean_tokens_per_second",
        "wall_time_seconds",
        "generation_sample_accuracy",
        "generation_nextlat_compat_sample_accuracy",
    ]
    ordered = [field for field in preferred if field in fields]
    ordered.extend(field for field in fields if field not in ordered)
    return ordered


def summarize_conditions(
    records: list[dict[str, Any]],
    *,
    condition_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_condition_key(record), []).append(record)
    scalar_fields = _scalar_summary_fields(records)
    rows = []
    for (variant, lambda_transition), condition in sorted(
        grouped.items(),
        key=lambda item: (
            _ordered_condition_sort_key(item[0][0], condition_order),
            -1.0 if item[0][1] is None else item[0][1],
        ),
    ):
        summary: dict[str, Any] = {
            "variant": variant,
            "architecture": condition[0]["architecture"],
            "transition": condition[0]["transition"],
            "lambda_transition": lambda_transition,
            "lambda_kl": condition[0].get("lambda_kl"),
            "transition_target": condition[0].get("transition_target"),
            "ntp_pass_weights": condition[0].get("ntp_pass_weights"),
            "seeds": sorted(record["seed"] for record in condition),
        }
        for field in scalar_fields:
            values = [
                float(record[field])
                for record in condition
                if record.get(field) is not None
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
                    "final_pass_nll_delta": (
                        right["final_pass_nll"] - left["final_pass_nll"]
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
                "final_pass_nll_delta": _scalar_statistics(
                    [row["final_pass_nll_delta"] for row in seed_rows]
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
            writer.writerow({key: _csv_value(value) for key, value in record.items()})


def _mean_text(value: dict[str, float] | None, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value['mean']:.{digits}f} +/- {value['std']:.{digits}f}"


def write_summary_markdown(
    path: Path,
    *,
    scale: str,
    selected_lambdas: dict[str, float],
    all_condition_summary: list[dict[str, Any]],
    condition_summary: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {scale.replace('_', ' ').title()} Summary",
        "",
        "Validation metrics are reported; no separate held-out test set is used.",
        "",
        "## Selected transition weights",
        "",
    ]
    if selected_lambdas:
        for condition, value in sorted(selected_lambdas.items()):
            lines.append(f"- `{condition}`: `{value:g}`")
    else:
        lines.append("- None")

    if any(row["lambda_transition"] is not None for row in all_condition_summary):
        lines.extend(
            [
                "",
                "## Transition-weight sweep",
                "",
                "| Condition | lambda | Selection score |",
                "|---|---:|---:|",
            ]
        )
        for row in all_condition_summary:
            if row["lambda_transition"] is None:
                continue
            score = row["best_checkpoint_selection_score"]
            lines.append(
                f"| `{row['variant']}` | {row['lambda_transition']:g} | "
                f"{_mean_text(score)} |"
            )

    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "| Condition | Architecture | Transition | lambda | Strict val | "
            "NextLat compat | Final-pass NLL | Transition loss | KL loss |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in condition_summary:
        lambda_text = (
            "-"
            if row["lambda_transition"] is None
            else f"{row['lambda_transition']:g}"
        )
        lines.append(
            f"| `{row['variant']}` | {row['architecture']} | "
            f"{row['transition']} | {lambda_text} | "
            f"{_mean_text(row.get('val_strict_multiset_accuracy'), digits=3)} | "
            f"{_mean_text(row.get('val_nextlat_compat_accuracy'), digits=3)} | "
            f"{_mean_text(row.get('final_pass_nll'))} | "
            f"{_mean_text(row.get('transition_prediction_loss'))} | "
            f"{_mean_text(row.get('transition_kl_loss'))} |"
        )

    equation_keys = sorted(
        {
            key
            for row in condition_summary
            for key in row
            if key.startswith("val_valid_equation_")
        }
    )
    if equation_keys:
        lines.extend(
            [
                "",
                "## Equation Validity",
                "",
                "| Condition | " + " | ".join(equation_keys) + " |",
                "|---|" + "---:|" * len(equation_keys),
            ]
        )
        for row in condition_summary:
            values = " | ".join(
                _mean_text(row.get(key), digits=3) for key in equation_keys
            )
            lines.append(f"| `{row['variant']}` | {values} |")

    if comparisons:
        lines.extend(
            [
                "",
                "## Paired seed deltas",
                "",
                "Deltas are target minus source; positive accuracy and negative loss favor the target.",
                "",
                "| Comparison | Source -> target | Val accuracy delta | Final-pass NLL delta |",
                "|---|---|---:|---:|",
            ]
        )
        for comparison in comparisons:
            lines.append(
                f"| {comparison['name']} | `{comparison['source']}` -> "
                f"`{comparison['target']}` | "
                f"{_mean_text(comparison['val_accuracy_delta'])} | "
                f"{_mean_text(comparison['final_pass_nll_delta'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison(
    path: Path,
    condition_summary: list[dict[str, Any]],
) -> None:
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


def _condition_order_from_specs(specs: list[ExpandedRunSpec]) -> list[str]:
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
    condition_order = _condition_order_from_specs(specs)
    all_condition_summary = summarize_conditions(
        records,
        condition_order=condition_order,
    )
    condition_summary = summarize_conditions(
        selected_records,
        condition_order=condition_order,
    )
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
    )
    plot_comparison(output_dir / "comparison.png", condition_summary)
    return result
