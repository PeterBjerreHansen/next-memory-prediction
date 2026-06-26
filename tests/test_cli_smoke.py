from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nmp.artifacts import artifacts_for, read_jsonl
from nmp.checkpoint import load_checkpoint
from nmp.cli.probe import main as probe_main
from nmp.cli.train import main as train_main


@pytest.mark.parametrize(
    "variant",
    [
        "transformer_ntp",
        "memory_tape_ntp",
        "memory_tape_nmp",
        "memory_tape_hidden_transition",
        "memory_tape_hidden_transition_kl",
    ],
)
def test_offline_smoke_workflow(variant, local_countdown_files, tmp_path: Path):
    train_file, val_file = local_countdown_files
    run_dir = tmp_path / variant
    train_main(
        [
            "--config",
            "configs/scales/smoke.yaml",
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
    train_main(
        [
            "--resume-from",
            str(run_dir / "latest.pt"),
            "--device",
            "cpu",
            "--steps",
            "2",
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
    assert load_checkpoint(artifacts.latest_checkpoint)["step"] == 2
    payloads = read_jsonl(artifacts.metrics_path)
    payloads.append(json.loads(artifacts.evaluation_path.read_text()))

    def assert_finite(value):
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, dict):
            for child in value.values():
                assert_finite(child)
        elif isinstance(value, list):
            for child in value:
                assert_finite(child)

    assert_finite(payloads)
