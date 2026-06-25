from __future__ import annotations

import torch
from torch.nn import functional as F

from nmp.models import (
    MemoryDynamicsPredictor,
    MemoryTapeConfig,
    MemoryTapeTransformer,
)
from nmp.objectives import (
    compute_loss,
    next_token_loss,
    temporal_memory_prediction_loss,
)


def make_model():
    return MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=5,
            vocab_size=19,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_pass=2,
        )
    )


def test_next_token_loss_masks_padding_but_trains_eos():
    logits = torch.randn(1, 4, 7)
    tokens = torch.tensor([[3, 4, 1, 0]])
    actual = next_token_loss(logits, tokens, pad_token_id=0)
    expected = F.cross_entropy(
        logits[:, :2].reshape(-1, 7),
        tokens[:, 1:3].reshape(-1),
    )
    assert torch.allclose(actual, expected)


def test_memory_loss_detaches_target_and_masks_eos_and_padding():
    torch.manual_seed(4)
    model = make_model()
    predictor = MemoryDynamicsPredictor(8)
    memory = torch.randn(1, 4, 8, requires_grad=True)
    tokens = torch.tensor([[2, 3, 1, 0]])
    loss = temporal_memory_prediction_loss(
        model,
        predictor,
        memory,
        tokens,
        eos_token_id=1,
        pad_token_id=0,
    )
    loss.backward()
    assert memory.grad[:, 0].abs().sum() > 0
    assert memory.grad[:, 1:].abs().sum() == 0
    assert predictor.mlp[0].weight.grad.abs().sum() > 0
    assert model.transformer["wte"].weight.grad.abs().sum() > 0


def test_nmp_backward_reaches_model_memory_and_embedding_parameters():
    torch.manual_seed(8)
    model = make_model()
    predictor = MemoryDynamicsPredictor(8)
    tokens = torch.tensor([[2, 3, 4, 1, 0], [5, 6, 7, 1, 0]])
    output = model(tokens)
    losses = compute_loss(
        variant="memory_tape_nmp",
        model=model,
        output=output,
        tokens=tokens,
        pad_token_id=0,
        eos_token_id=1,
        predictor=predictor,
        lambda_memory=1.0,
    )
    losses.total.backward()
    assert model.mem_head.weight.grad.abs().sum() > 0
    assert model.transformer["wte"].weight.grad.abs().sum() > 0
    assert predictor.mlp[0].weight.grad.abs().sum() > 0


def test_lambda_zero_reproduces_memory_tape_ntp_total():
    torch.manual_seed(12)
    model = make_model()
    predictor = MemoryDynamicsPredictor(8)
    tokens = torch.tensor([[2, 3, 4, 1, 0]])
    output = model(tokens)
    ntp = compute_loss(
        variant="memory_tape_ntp",
        model=model,
        output=output,
        tokens=tokens,
        pad_token_id=0,
        eos_token_id=1,
    )
    nmp_zero = compute_loss(
        variant="memory_tape_nmp",
        model=model,
        output=output,
        tokens=tokens,
        pad_token_id=0,
        eos_token_id=1,
        predictor=predictor,
        lambda_memory=0.0,
    )
    assert torch.equal(ntp.total, nmp_zero.total)

