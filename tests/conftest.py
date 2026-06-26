from __future__ import annotations

from pathlib import Path

import pytest

from nmp.config import ExperimentConfig


@pytest.fixture
def local_countdown_files() -> tuple[Path, Path]:
    root = Path(__file__).parent / "fixtures"
    train = root / "countdown_train.txt"
    val = root / "countdown_val.txt"
    return train, val


def make_config(
    variant: str,
    train_file: Path,
    val_file: Path,
    *,
    train_steps: int = 2,
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "name": f"test-{variant}",
            "seed": 11,
            "model": {
                "variant": variant,
                "block_size": 40,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
                "memory": {"n_pass": 2},
            },
            "data": {
                "source": "local",
                "train_file": str(train_file),
                "val_file": str(val_file),
                "countdown_max_intermediate": 10000,
                "countdown_input_numbers": 4,
                "countdown_num_equations": 3,
                "num_pause_tokens": 8,
            },
            "objective": {
                "transition": {
                    "lambda_transition": 1.0,
                    "projection_factor": 1.3,
                },
            },
            "training": {
                "train_steps": train_steps,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "eval_interval": 1,
                "eval_batches": 1,
                "log_interval": 1,
                "checkpoint_interval": 1,
                "device": "cpu",
                "precision": "float32",
                "compile": False,
            },
            "evaluation": {
                "generation_prompts": 1,
                "prompt_tokens": 4,
                "generation_tokens": 2,
                "diagnostic_batches": 1,
                "probe_steps": 1,
                "probe_batch_size": 2,
                "probe_offsets": [1, 2],
                "checkpoint_metric": "val_accuracy",
                "checkpoint_mode": "max",
            },
        }
    )
