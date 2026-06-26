from __future__ import annotations

import torch

from nmp.checkpoint import restore_checkpoint


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
