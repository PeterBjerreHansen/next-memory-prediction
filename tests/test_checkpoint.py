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


def test_checkpoint_config_migrates_retired_objective_fields():
    checkpoint = {
        "config": {
            "name": "old",
            "seed": 0,
            "model": {
                "variant": "transformer_ntp",
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
            },
            "objective": {
                "lambda_transition": 0.3,
                "memory_horizon": 1,
            },
            "training": {
                "train_steps": 1,
                "micro_batch_size": 1,
            },
        }
    }

    config = config_from_checkpoint(checkpoint)

    assert config.objective.lambda_transition == 0.3
    assert "memory_horizon" not in config.to_dict()["objective"]
