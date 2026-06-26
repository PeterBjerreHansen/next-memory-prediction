from __future__ import annotations

import torch
from torch.nn import functional as F

from nmp.models import (
    LatentTransitionPredictor,
    MemoryTapeConfig,
    MemoryTapeTransformer,
)
from nmp.objectives import (
    compute_loss,
    next_token_loss,
    temporal_transition_prediction_loss,
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
    predictor = LatentTransitionPredictor(8)
    memory = torch.randn(1, 4, 8, requires_grad=True)
    tokens = torch.tensor([[2, 3, 1, 0]])
    loss = temporal_transition_prediction_loss(
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
    predictor = LatentTransitionPredictor(8)
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
        lambda_transition=1.0,
    )
    losses.total.backward()
    assert model.mem_head.weight.grad.abs().sum() > 0
    assert model.transformer["wte"].weight.grad.abs().sum() > 0
    assert predictor.mlp[0].weight.grad.abs().sum() > 0


def test_hidden_transition_uses_final_pass_hidden_states():
    model = make_model()
    tokens = torch.tensor([[2, 3, 4, 1, 0]])
    output = model(tokens)

    class RecordingPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, current_latent, next_token_embeddings):
            self.seen = current_latent
            return current_latent

    predictor = RecordingPredictor()
    losses = compute_loss(
        variant="memory_tape_hidden_transition",
        model=model,
        output=output,
        tokens=tokens,
        pad_token_id=0,
        eos_token_id=1,
        predictor=predictor,
    )
    assert losses.transition_target == "hidden"
    assert torch.equal(predictor.seen, output.hidden_states[:, :-1])


def test_memory_transition_uses_final_pass_memory_states():
    model = make_model()
    tokens = torch.tensor([[2, 3, 4, 1, 0]])
    output = model(tokens)

    class RecordingPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, current_latent, next_token_embeddings):
            self.seen = current_latent
            return current_latent

    predictor = RecordingPredictor()
    losses = compute_loss(
        variant="memory_tape_nmp",
        model=model,
        output=output,
        tokens=tokens,
        pad_token_id=0,
        eos_token_id=1,
        predictor=predictor,
    )
    assert losses.transition_target == "memory"
    assert torch.equal(predictor.seen, output.memory_states[:, :-1])


def test_memory_tape_ntp_pass_weights_are_normalized_and_applied():
    torch.manual_seed(31)
    model = make_model()
    tokens = torch.tensor([[2, 3, 4, 1, 0]])
    output = model(tokens)
    losses = compute_loss(
        variant="memory_tape_ntp",
        model=model,
        output=output,
        tokens=tokens,
        pad_token_id=0,
        eos_token_id=1,
        ntp_pass_weights=[0.0, 2.0],
    )
    assert losses.ntp_pass_weights == (0.0, 1.0)
    assert torch.equal(losses.weighted_ntp, losses.pass_nlls[-1])


def test_hidden_transition_detaches_target_and_reaches_current_and_embedding():
    torch.manual_seed(21)
    model = make_model()
    predictor = LatentTransitionPredictor(8)
    hidden = torch.randn(1, 4, 8, requires_grad=True)
    tokens = torch.tensor([[2, 3, 1, 0]])
    loss = temporal_transition_prediction_loss(
        model,
        predictor,
        hidden,
        tokens,
        eos_token_id=1,
        pad_token_id=0,
    )
    loss.backward()
    assert hidden.grad[:, 0].abs().sum() > 0
    assert hidden.grad[:, 1:].abs().sum() == 0
    assert predictor.mlp[0].weight.grad.isfinite().all()
    assert model.transformer["wte"].weight.grad.isfinite().all()


@torch.no_grad()
def test_memory_variants_share_model_initialization_and_predictor_shape(
    local_story_files,
):
    from conftest import make_config
    from nmp.factory import build_model, count_parameters

    train_file, val_file = local_story_files
    built = {}
    for variant in (
        "memory_tape_ntp",
        "memory_tape_nmp",
        "memory_tape_hidden_transition",
    ):
        torch.manual_seed(77)
        built[variant] = build_model(
            make_config(variant, train_file, val_file),
            vocab_size=19,
        )
    baseline_state = built["memory_tape_ntp"][0].state_dict()
    for variant in ("memory_tape_nmp", "memory_tape_hidden_transition"):
        for name, value in baseline_state.items():
            assert torch.equal(value, built[variant][0].state_dict()[name])
    memory_predictor = built["memory_tape_nmp"][1]
    hidden_predictor = built["memory_tape_hidden_transition"][1]
    for name, value in memory_predictor.state_dict().items():
        assert torch.equal(value, hidden_predictor.state_dict()[name])
    assert count_parameters(*built["memory_tape_nmp"])["training_only"] == (
        count_parameters(*built["memory_tape_hidden_transition"])[
            "training_only"
        ]
    )


@torch.no_grad()
def test_lambda_zero_reproduces_memory_tape_ntp_total():
    torch.manual_seed(12)
    model = make_model()
    predictor = LatentTransitionPredictor(8)
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
    for variant in ("memory_tape_nmp", "memory_tape_hidden_transition"):
        transition_zero = compute_loss(
            variant=variant,
            model=model,
            output=output,
            tokens=tokens,
            pad_token_id=0,
            eos_token_id=1,
            predictor=predictor,
            lambda_transition=0.0,
        )
        assert torch.equal(ntp.total, transition_zero.total)
