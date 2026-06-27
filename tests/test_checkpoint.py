from __future__ import annotations

import torch

from nmp.checkpoint import config_from_checkpoint, restore_checkpoint


def test_eval_style_restore_does_not_require_rng_state():
    model = torch.nn.Linear(2, 2)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": None,
    }

    restore_checkpoint(
        checkpoint,
        model=model,
        predictor=None,
        restore_rng=False,
    )


def test_checkpoint_config_uses_current_schema_exactly():
    checkpoint = {
        "config": {
            "name": "current",
            "seed": 0,
            "model": {
                "variant": "memory_tape_nmp",
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "memory": {"n_pass": 3},
            },
            "objective": {
                "transition": {
                    "horizon": 1,
                    "lambda_transition": 0.3,
                    "target": "memory",
                    "projection_factor": 1.7,
                },
            },
            "training": {
                "train_steps": 1,
                "micro_batch_size": 1,
            },
        }
    }

    config = config_from_checkpoint(checkpoint)

    assert config.objective.transition.lambda_transition == 0.3
    assert config.objective.transition.horizon == 1
    assert config.objective.transition.projection_factor == 1.7
    assert config.model.memory.n_pass == 3
