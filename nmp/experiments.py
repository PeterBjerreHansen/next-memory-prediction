from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import artifacts_for, read_jsonl, write_json
from .config import TRANSITION_VARIANTS, load_config, transition_target_for_variant


ROUND1_SEEDS = (0, 1, 2)
ROUND1_LAMBDAS = (0.1, 0.3, 1.0, 3.0)
BASELINE_VARIANTS = ("transformer_ntp", "memory_tape_ntp")
ROUND1_VARIANTS = (*BASELINE_VARIANTS, *TRANSITION_VARIANTS)


@dataclass(frozen=True)
class RunSpec:
    variant: str
    seed: int
    lambda_transition: float | None = None


def format_lambda(value: float) -> str:
    return str(float(value))


def run_directory(
    runs_root: str | Path,
    scale: str,
    spec: RunSpec,
) -> Path:
    path = Path(runs_root) / scale / spec.variant
    if spec.lambda_transition is not None:
        path /= f"lambda_{format_lambda(spec.lambda_transition)}"
    return path / f"seed_{spec.seed}"


def expected_run_specs(
    scale: str,
    *,
    selected_lambdas: dict[str, float] | None = None,
) -> list[RunSpec]:
    if scale not in {"development", "reference"}:
        raise ValueError("scale must be development or reference")
    specs = [
        RunSpec(variant=variant, seed=seed)
        for variant in BASELINE_VARIANTS
        for seed in ROUND1_SEEDS
    ]
    if scale == "development":
        specs.extend(
            RunSpec(
                variant=variant,
                seed=seed,
                lambda_transition=lambda_transition,
            )
            for variant in TRANSITION_VARIANTS
            for lambda_transition in ROUND1_LAMBDAS
            for seed in ROUND1_SEEDS
        )
        return specs
    if selected_lambdas is None:
        raise ValueError("reference summaries require selected transition weights")
    missing = [
        variant for variant in TRANSITION_VARIANTS if variant not in selected_lambdas
    ]
    if missing:
        raise ValueError(
            "selection file is missing variants: " + ", ".join(missing)
        )
    specs.extend(
        RunSpec(
            variant=variant,
            seed=seed,
            lambda_transition=float(selected_lambdas[variant]),
        )
        for variant in TRANSITION_VARIANTS
        for seed in ROUND1_SEEDS
    )
    return specs


def load_selected_lambdas(path: str | Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("selected_lambdas", payload)
    return {variant: float(values[variant]) for variant in TRANSITION_VARIANTS}


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


def _best_validation_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    validation = [
        row
        for row in rows
        if row.get("event") == "validation"
        and row.get("final_pass_nll") is not None
    ]
    if not validation:
        raise ValueError("run contains no validation metrics")
    return min(validation, key=lambda row: float(row["final_pass_nll"]))


def load_run_record(
    run_dir: str | Path,
    *,
    scale: str,
    spec: RunSpec,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
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
        float(config.objective.lambda_transition)
        != float(spec.lambda_transition)
    ):
        raise ValueError(f"transition weight mismatch in {run_dir}")

    metrics = read_jsonl(artifacts.metrics_path)
    best_validation = _best_validation_row(metrics)
    evaluation = _read_evaluation(artifacts.evaluation_path)
    evaluation_loss = evaluation["loss"]
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
    transition_loss = evaluation_loss.get("transition_prediction_loss")
    if transition_loss is None:
        transition_loss = evaluation_loss.get("memory_prediction_loss")
    parameters = evaluation.get("parameters", {})
    generation = evaluation.get("generation", {})

    return {
        "scale": scale,
        "variant": spec.variant,
        "seed": spec.seed,
        "lambda_transition": (
            None
            if spec.variant in BASELINE_VARIANTS
            else float(config.objective.lambda_transition)
        ),
        "transition_target": transition_target_for_variant(spec.variant),
        "run_dir": str(run_dir),
        "best_checkpoint_final_pass_nll": float(
            best_validation["final_pass_nll"]
        ),
        "best_checkpoint_step": int(best_validation["step"]),
        "final_pass_nll": float(evaluation_loss["final_pass_nll"]),
        "perplexity": float(evaluation_loss["perplexity"]),
        "pass_nlls": list(map(float, evaluation_loss["pass_nlls"])),
        "transition_prediction_loss": (
            None if transition_loss is None else float(transition_loss)
        ),
        "parameters_model": int(parameters.get("model", 0)),
        "parameters_training_only": int(parameters.get("training_only", 0)),
        "parameters_total_training": int(parameters.get("total_training", 0)),
        "mean_tokens_per_second": _mean_or_none(
            row.get("tokens_per_second") for row in train_rows
        ),
        "wall_time_seconds": wall_time_seconds,
        "generation_agreement": generation.get(
            "recompute_final_pass_agreement"
        ),
        "representations": evaluation.get("representations", {}),
        "probes": probes,
    }


def _condition_key(record: dict[str, Any]) -> tuple[str, float | None]:
    return record["variant"], record["lambda_transition"]


def select_development_lambdas(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for variant in TRANSITION_VARIANTS:
        candidates = []
        for lambda_transition in ROUND1_LAMBDAS:
            rows = [
                record
                for record in records
                if record["variant"] == variant
                and record["lambda_transition"] == lambda_transition
            ]
            if len(rows) != len(ROUND1_SEEDS):
                raise ValueError(
                    f"{variant} lambda={lambda_transition:g} has "
                    f"{len(rows)} seeds, expected {len(ROUND1_SEEDS)}"
                )
            mean_nll = statistics.mean(
                row["best_checkpoint_final_pass_nll"] for row in rows
            )
            candidates.append((mean_nll, lambda_transition))
        selected[variant] = min(candidates)[1]
    return selected


def selected_condition_records(
    records: list[dict[str, Any]],
    selected_lambdas: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["variant"] in BASELINE_VARIANTS
        or record["lambda_transition"] == selected_lambdas[record["variant"]]
    ]


def _scalar_statistics(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_conditions(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_condition_key(record), []).append(record)
    rows = []
    scalar_fields = (
        "best_checkpoint_final_pass_nll",
        "final_pass_nll",
        "perplexity",
        "transition_prediction_loss",
        "parameters_model",
        "parameters_training_only",
        "parameters_total_training",
        "mean_tokens_per_second",
        "wall_time_seconds",
        "generation_agreement",
    )
    for (variant, lambda_transition), condition in sorted(
        grouped.items(),
        key=lambda item: (
            ROUND1_VARIANTS.index(item[0][0]),
            -1.0 if item[0][1] is None else item[0][1],
        ),
    ):
        summary: dict[str, Any] = {
            "variant": variant,
            "lambda_transition": lambda_transition,
            "transition_target": transition_target_for_variant(variant),
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
        (
            "hidden_transition",
            "memory_tape_ntp",
            "memory_tape_hidden_transition",
        ),
        (
            "explicit_memory_vs_hidden",
            "memory_tape_hidden_transition",
            "memory_tape_nmp",
        ),
    )
    results = []
    for name, source, target in comparisons:
        seed_rows = []
        for seed in ROUND1_SEEDS:
            left = conditions[(source, seed)]
            right = conditions[(target, seed)]
            seed_rows.append(
                {
                    "seed": seed,
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
                "definition": "target minus source; negative favors target",
                "by_seed": seed_rows,
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
        f"# Round 1 {scale.title()} Summary",
        "",
        "Validation metrics are reported; no separate held-out test set is used.",
        "",
        "## Selected transition weights",
        "",
    ]
    for variant in TRANSITION_VARIANTS:
        lines.append(f"- `{variant}`: `{selected_lambdas[variant]:g}`")
    if scale == "development":
        lines.extend(
            [
                "",
                "## Transition-weight sweep",
                "",
                "| Variant | λ | Mean best-checkpoint final-pass NLL |",
                "|---|---:|---:|",
            ]
        )
        for row in all_condition_summary:
            if row["lambda_transition"] is None:
                continue
            score = row["best_checkpoint_final_pass_nll"]
            lines.append(
                f"| `{row['variant']}` | {row['lambda_transition']:g} | "
                f"{score['mean']:.4f} ± {score['std']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "| Variant | λ | Final-pass NLL | Perplexity | Transition loss |",
            "|---|---:|---:|---:|---:|",
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
        lines.append(
            f"| `{row['variant']}` | {lambda_text} | "
            f"{row['final_pass_nll']['mean']:.4f} ± "
            f"{row['final_pass_nll']['std']:.4f} | "
            f"{row['perplexity']['mean']:.3f} ± "
            f"{row['perplexity']['std']:.3f} | {transition_text} |"
        )
    lines.extend(
        [
            "",
            "## Compute and generation",
            "",
            "| Variant | Model params | Training-only params | Tokens/s | "
            "Generation agreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in condition_summary:
        throughput = row["mean_tokens_per_second"]
        agreement = row["generation_agreement"]
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
            "Deltas are target minus source; negative values favor the target.",
            "",
            "| Comparison | Source → target | Final-pass NLL Δ | Perplexity Δ |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison in comparisons:
        nll = comparison["final_pass_nll_delta"]
        perplexity = comparison["perplexity_delta"]
        lines.append(
            f"| {comparison['name']} | `{comparison['source']}` → "
            f"`{comparison['target']}` | "
            f"{nll['mean']:.4f} ± {nll['std']:.4f} | "
            f"{perplexity['mean']:.3f} ± {perplexity['std']:.3f} |"
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
        (axes[0], "final_pass_nll", "Final-pass validation NLL"),
        (axes[1], "perplexity", "Validation perplexity"),
    ):
        axis.errorbar(
            positions,
            [row[field]["mean"] for row in condition_summary],
            yerr=[row[field]["std"] for row in condition_summary],
            fmt="o",
            capsize=4,
        )
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def summarize_round1(
    *,
    runs_root: str | Path,
    scale: str,
    output_dir: str | Path,
    selection_file: str | Path | None = None,
) -> dict[str, Any]:
    if scale == "development":
        selected_for_manifest = None
    else:
        if selection_file is None:
            raise ValueError("reference summary requires --selection-file")
        selected_for_manifest = load_selected_lambdas(selection_file)
    specs = expected_run_specs(
        scale,
        selected_lambdas=selected_for_manifest,
    )
    records = [
        load_run_record(
            run_directory(runs_root, scale, spec),
            scale=scale,
            spec=spec,
        )
        for spec in specs
    ]
    selected_lambdas = (
        select_development_lambdas(records)
        if scale == "development"
        else selected_for_manifest
    )
    assert selected_lambdas is not None
    selected_records = selected_condition_records(records, selected_lambdas)
    all_condition_summary = summarize_conditions(records)
    condition_summary = summarize_conditions(selected_records)
    probe_summary = summarize_probes(selected_records)
    comparisons = paired_comparisons(records, selected_lambdas)
    result = {
        "scale": scale,
        "expected_runs": len(specs),
        "completed_runs": len(records),
        "selection_criterion": (
            "lowest mean best-checkpoint final-pass validation NLL across seeds"
        ),
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
        scale=scale,
        selected_lambdas=selected_lambdas,
        all_condition_summary=all_condition_summary,
        condition_summary=condition_summary,
        comparisons=comparisons,
        probe_summary=probe_summary,
    )
    plot_comparison(output_dir / "comparison.png", condition_summary)
    return result
