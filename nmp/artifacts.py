from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, save_config
from .provenance import provenance_manifest


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config_path: Path
    provenance_path: Path
    metrics_path: Path
    latest_checkpoint: Path
    best_checkpoint: Path
    samples_path: Path
    evaluation_path: Path
    probe_metrics_path: Path
    probe_checkpoint_path: Path
    plots_dir: Path


def artifacts_for(run_dir: str | Path) -> RunArtifacts:
    root = Path(run_dir).resolve()
    return RunArtifacts(
        run_dir=root,
        config_path=root / "config.yaml",
        provenance_path=root / "provenance.json",
        metrics_path=root / "metrics.jsonl",
        latest_checkpoint=root / "latest.pt",
        best_checkpoint=root / "best.pt",
        samples_path=root / "samples.jsonl",
        evaluation_path=root / "evaluation.json",
        probe_metrics_path=root / "probe_metrics.jsonl",
        probe_checkpoint_path=root / "probes.pt",
        plots_dir=root / "plots",
    )


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def prepare_run(
    run_dir: str | Path,
    config: ExperimentConfig,
    *,
    fresh: bool,
) -> RunArtifacts:
    artifacts = artifacts_for(run_dir)
    artifacts.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts.plots_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for path in (
            artifacts.metrics_path,
            artifacts.samples_path,
            artifacts.probe_metrics_path,
            artifacts.latest_checkpoint,
            artifacts.best_checkpoint,
            artifacts.evaluation_path,
            artifacts.probe_checkpoint_path,
        ):
            if path.exists():
                path.unlink()
        for path in artifacts.plots_dir.glob("*.png"):
            path.unlink()
    save_config(artifacts.config_path, config)
    write_json(artifacts.provenance_path, provenance_manifest())
    return artifacts


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
