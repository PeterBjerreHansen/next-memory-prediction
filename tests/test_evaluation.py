from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nmp.data import TextBatch
from nmp.evaluation import evaluate_batches
from nmp.models import TransformerOutput
from nmp.objectives import next_token_loss


class UnequalBatchLossModel(nn.Module):
    def forward(self, tokens: torch.Tensor) -> TransformerOutput:
        logits = torch.zeros((*tokens.shape, 5), device=tokens.device)
        if int(tokens[0, 0]) == 2:
            logits[:, 0, 1] = 4.0
        hidden = torch.zeros((*tokens.shape, 3), device=tokens.device)
        return TransformerOutput(logits=logits, hidden_states=hidden)


def test_validation_nll_is_weighted_by_valid_target_tokens():
    model = UnequalBatchLossModel()
    batches = [
        TextBatch(
            tokens=torch.tensor([[2, 1, 0, 0]]),
            lengths=torch.tensor([2]),
        ),
        TextBatch(
            tokens=torch.tensor([[3, 4, 2, 1]]),
            lengths=torch.tensor([4]),
        ),
    ]
    config = SimpleNamespace(
        model=SimpleNamespace(variant="transformer_ntp"),
        training=SimpleNamespace(precision="float32"),
        objective=SimpleNamespace(lambda_memory=1.0),
    )
    tokenizer = SimpleNamespace(pad_id=0, eos_id=1)

    first_loss = next_token_loss(
        model(batches[0].tokens).logits,
        batches[0].tokens,
        pad_token_id=0,
    )
    second_loss = next_token_loss(
        model(batches[1].tokens).logits,
        batches[1].tokens,
        pad_token_id=0,
    )
    expected = (first_loss + 3 * second_loss) / 4

    metrics = evaluate_batches(
        config=config,
        model=model,
        predictor=None,
        batches=batches,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    assert metrics["ntp_tokens"] == 4
    assert metrics["final_pass_nll"] == pytest.approx(float(expected))
    assert metrics["loss"] == pytest.approx(float(expected))
    assert metrics["final_pass_nll"] != pytest.approx(
        float((first_loss + second_loss) / 2)
    )
