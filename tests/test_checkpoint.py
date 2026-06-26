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


def test_checkpoint_config_drops_unknown_nested_fields():
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
                "n_pass": 3,
                "retired_model_field": "scalar",
            },
            "objective": {
                "transition_horizon": 1,
                "lambda_transition": 0.3,
                "lambda_ce": 0.2,
                "dynamics_projection_factor": 1.7,
                "retired_objective_field": 1,
            },
            "training": {
                "train_steps": 1,
                "micro_batch_size": 1,
                "retired_training_field": True,
            },
        }
    }

    config = config_from_checkpoint(checkpoint)

    assert config.objective.transition.lambda_transition == 0.3
    assert config.objective.transition.lambda_ce == 0.2
    assert config.objective.transition.horizon == 1
    assert config.objective.transition.projection_factor == 1.7
    assert config.model.memory.n_pass == 3
    resolved = config.to_dict()
    assert "n_pass" not in resolved["model"]
    assert "retired_model_field" not in resolved["model"]
    assert "lambda_transition" not in resolved["objective"]
    assert "lambda_ce" not in resolved["objective"]
    assert "transition_horizon" not in resolved["objective"]
    assert "dynamics_projection_factor" not in resolved["objective"]
    assert "retired_objective_field" not in resolved["objective"]
    assert "retired_training_field" not in resolved["training"]
