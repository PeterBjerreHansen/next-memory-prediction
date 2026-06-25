from __future__ import annotations

from pathlib import Path

import pytest

from nmp.config import ExperimentConfig


STORIES = [
    "Once upon a time, a little fox found a red ball in the garden.",
    "Mia and Tom built a small boat and sailed across the blue pond.",
    "The kind bear helped a bird put its nest back in the tree.",
    "Lucy lost her toy, but her friend Sam found it under the bed.",
    "A green frog jumped over a rock and made everyone laugh.",
    "The puppy was scared of the rain until Lily gave it a warm blanket.",
    "Ben planted a seed and watched a bright flower grow.",
    "The tiny mouse shared its cake with all the animals.",
]


@pytest.fixture
def local_story_files(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("\n".join(STORIES * 2) + "\n", encoding="utf-8")
    val.write_text("\n".join(reversed(STORIES)) + "\n", encoding="utf-8")
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
                "block_size": 24,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 16,
                "n_pass": 2,
                "memory_tape_gate": "scalar",
            },
            "data": {
                "source": "local",
                "train_file": str(train_file),
                "val_file": str(val_file),
            },
            "objective": {
                "lambda_memory": 1.0,
                "memory_horizon": 1,
                "dynamics_projection_factor": 1.3,
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
            },
        }
    )

