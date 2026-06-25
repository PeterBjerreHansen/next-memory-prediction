from __future__ import annotations

from pathlib import Path

import pytest

from nmp.artifacts import artifacts_for
from nmp.cli.probe import main as probe_main
from nmp.cli.train import main as train_main


@pytest.mark.parametrize(
    "variant",
    ["transformer_ntp", "memory_tape_ntp", "memory_tape_nmp"],
)
def test_offline_smoke_workflow(variant, local_story_files, tmp_path: Path):
    train_file, val_file = local_story_files
    run_dir = tmp_path / variant
    train_main(
        [
            "--config",
            "configs/smoke.yaml",
            "--variant",
            variant,
            "--run-dir",
            str(run_dir),
            "--train-file",
            str(train_file),
            "--val-file",
            str(val_file),
            "--device",
            "cpu",
            "--steps",
            "1",
        ]
    )
    probe_main(
        [
            "--run-dir",
            str(run_dir),
            "--device",
            "cpu",
            "--steps",
            "1",
        ]
    )
    artifacts = artifacts_for(run_dir)
    for path in (
        artifacts.config_path,
        artifacts.provenance_path,
        artifacts.metrics_path,
        artifacts.latest_checkpoint,
        artifacts.best_checkpoint,
        artifacts.samples_path,
        artifacts.evaluation_path,
        artifacts.probe_metrics_path,
        artifacts.probe_checkpoint_path,
        artifacts.plots_dir / "training.png",
        artifacts.plots_dir / "probes.png",
    ):
        assert path.exists(), path
