from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import artifacts_for, read_jsonl


def transition_loss_from_row(row: dict) -> float | None:
    return row.get("transition_prediction_loss")


def plot_training(run_dir: str | Path) -> Path | None:
    artifacts = artifacts_for(run_dir)
    rows = read_jsonl(artifacts.metrics_path)
    train = [row for row in rows if row.get("event") == "train"]
    validation = [row for row in rows if row.get("event") == "validation"]
    accuracy_diagnostics = [
        row
        for row in rows
        if row.get("event") in {"validation", "accuracy_diagnostic"}
        and row.get("val_accuracy") is not None
    ]
    if not train and not validation and not accuracy_diagnostics:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    if train:
        axes[0].plot(
            [row["step"] for row in train],
            [row["final_pass_nll"] for row in train],
            label="train final-pass NLL",
        )
        transition_rows = [
            row for row in train if transition_loss_from_row(row) is not None
        ]
        if transition_rows:
            axes[1].plot(
                [row["step"] for row in transition_rows],
                [transition_loss_from_row(row) for row in transition_rows],
                label="transition prediction",
            )
        kl_rows = [row for row in train if row.get("transition_kl_loss") is not None]
        if kl_rows:
            axes[1].plot(
                [row["step"] for row in kl_rows],
                [row["transition_kl_loss"] for row in kl_rows],
                label="transition KL",
            )
    if validation:
        axes[0].plot(
            [row["step"] for row in validation],
            [row["final_pass_nll"] for row in validation],
            marker="o",
            label="validation final-pass NLL",
        )
        if not accuracy_diagnostics:
            axes[1].plot(
                [row["step"] for row in validation],
                [row["perplexity"] for row in validation],
                marker="o",
                label="validation perplexity",
            )
    if accuracy_diagnostics:
        axes[1].plot(
            [row["step"] for row in accuracy_diagnostics],
            [row["val_accuracy"] for row in accuracy_diagnostics],
            marker="o",
            label="strict validation accuracy",
        )
        compat_rows = [
            row
            for row in accuracy_diagnostics
            if row.get("val_nextlat_compat_accuracy") is not None
        ]
        if compat_rows:
            axes[1].plot(
                [row["step"] for row in compat_rows],
                [row["val_nextlat_compat_accuracy"] for row in compat_rows],
                marker="o",
                label="NextLat-compatible accuracy",
            )
    axes[0].set_title("Token objective")
    axes[1].set_title("Countdown / auxiliary")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output = artifacts.plots_dir / "training.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_probes(run_dir: str | Path) -> Path | None:
    artifacts = artifacts_for(run_dir)
    rows = [
        row
        for row in read_jsonl(artifacts.probe_metrics_path)
        if row.get("event") == "probe_validation"
    ]
    if not rows:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    sources = sorted({row["source"] for row in rows})
    for source in sources:
        selected = sorted(
            (row for row in rows if row["source"] == source),
            key=lambda row: row["offset"],
        )
        offsets = [row["offset"] for row in selected]
        axes[0].plot(
            offsets,
            [row["cross_entropy"] for row in selected],
            marker="o",
            label=source,
        )
        axes[1].plot(
            offsets,
            [row["accuracy"] for row in selected],
            marker="o",
            label=source,
        )
    axes[0].set_title("Probe cross-entropy")
    axes[1].set_title("Probe accuracy")
    for axis in axes:
        axis.set_xlabel("future-token offset")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output = artifacts.plots_dir / "probes.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_run(run_dir: str | Path) -> list[Path]:
    return [
        path
        for path in (plot_training(run_dir), plot_probes(run_dir))
        if path is not None
    ]
