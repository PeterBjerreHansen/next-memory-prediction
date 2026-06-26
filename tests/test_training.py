from __future__ import annotations

from pathlib import Path

import pytest
import torch

from conftest import make_config
from nmp.artifacts import artifacts_for, read_jsonl
from nmp.checkpoint import load_checkpoint
from nmp.training import train_experiment


@pytest.mark.parametrize(
    "variant",
    [
        "memory_tape_nmp",
        "memory_tape_hidden_transition",
        "memory_tape_hidden_transition_kl",
    ],
)
def test_checkpoint_resume_is_exact(
    variant,
    local_countdown_files,
    tmp_path: Path,
):
    train_file, val_file = local_countdown_files
    full_config = make_config(
        variant,
        train_file,
        val_file,
        train_steps=4,
    )
    full_dir = tmp_path / "full"
    train_experiment(full_config, run_dir=full_dir)

    partial_config = make_config(
        variant,
        train_file,
        val_file,
        train_steps=2,
    )
    resumed_dir = tmp_path / "resumed"
    train_experiment(partial_config, run_dir=resumed_dir)
    resume_config = make_config(
        variant,
        train_file,
        val_file,
        train_steps=4,
    )
    train_experiment(
        resume_config,
        run_dir=resumed_dir,
        resume_from=resumed_dir / "latest.pt",
    )

    full = load_checkpoint(full_dir / "latest.pt")
    resumed = load_checkpoint(resumed_dir / "latest.pt")
    assert full["step"] == resumed["step"] == 4
    for name, value in full["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][name]), name
    for name, value in full["predictor_state_dict"].items():
        assert torch.equal(value, resumed["predictor_state_dict"][name]), name

    rows = read_jsonl(artifacts_for(resumed_dir).metrics_path)
    assert any(row["event"] == "run_resume" for row in rows)
    assert [row["step"] for row in rows if row["event"] == "validation"] == [
        1,
        2,
        3,
        4,
    ]
