from __future__ import annotations

import inspect
import types

import pytest
import torch

from nmp.models import (
    CausalTransformer,
    MemoryTapeConfig,
    MemoryTapeOutput,
    MemoryTapeTransformer,
    TransformerConfig,
)


def test_model_output_shapes_and_backward():
    torch.manual_seed(3)
    baseline = CausalTransformer(
        TransformerConfig(
            block_size=8,
            vocab_size=31,
            n_layer=1,
            n_head=2,
            n_embd=16,
        )
    )
    ids = torch.randint(0, 31, (2, 8))
    output = baseline(ids)
    assert output.logits.shape == (2, 8, 31)
    assert output.hidden_states.shape == (2, 8, 16)
    output.logits.square().mean().backward()
    assert baseline.transformer["wte"].weight.grad is not None

    memory_model = MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=8,
            vocab_size=31,
            n_layer=1,
            n_head=2,
            n_embd=16,
            n_pass=3,
        )
    )
    memory_output = memory_model(ids)
    assert len(memory_output.logits_per_pass) == 3
    assert memory_output.memory_states.shape == (2, 8, 16)
    memory_output.logits.square().mean().backward()
    assert memory_model.mem_head.weight.grad is not None


def test_first_pass_exactly_matches_trunk_matched_transformer():
    seed = 31
    base_config = TransformerConfig(
        block_size=8,
        vocab_size=31,
        n_layer=2,
        n_head=2,
        n_embd=16,
    )
    memory_config = MemoryTapeConfig(
        **base_config.to_dict(),
        n_pass=3,
        memory_tape_gate="scalar",
    )
    torch.manual_seed(seed)
    baseline = CausalTransformer(base_config)
    torch.manual_seed(seed)
    memory_model = MemoryTapeTransformer(memory_config)
    ids = torch.randint(0, 31, (2, 8))
    assert torch.equal(
        baseline(ids).logits,
        memory_model(ids).logits_per_pass[0],
    )


@pytest.mark.parametrize("model_kind", ["transformer", "memory"])
def test_future_tokens_cannot_change_earlier_outputs(model_kind):
    torch.manual_seed(5)
    if model_kind == "transformer":
        model = CausalTransformer(
            TransformerConfig(
                block_size=8,
                vocab_size=31,
                n_layer=2,
                n_head=2,
                n_embd=16,
            )
        )
    else:
        model = MemoryTapeTransformer(
            MemoryTapeConfig(
                block_size=8,
                vocab_size=31,
                n_layer=2,
                n_head=2,
                n_embd=16,
                n_pass=3,
            )
        )
    first = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    second = first.clone()
    second[:, 5:] = torch.tensor([[9, 10, 11]])
    left = model(first)
    right = model(second)
    assert torch.allclose(left.logits[:, :5], right.logits[:, :5])
    assert torch.allclose(left.hidden_states[:, :5], right.hidden_states[:, :5])
    if model_kind == "memory":
        assert torch.allclose(
            left.memory_states[:, :5],
            right.memory_states[:, :5],
        )


def test_shifted_previous_pass_memory_is_used():
    model = MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=5,
            vocab_size=17,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_pass=2,
        )
    )
    seen = []
    original = model._run_full_pass

    def recording(self, token_stream, memory_tape):
        seen.append(memory_tape.detach().clone())
        return original(token_stream, memory_tape)

    model._run_full_pass = types.MethodType(recording, model)
    output = model(torch.randint(0, 17, (1, 5)))
    assert torch.count_nonzero(seen[0]) == 0
    assert torch.allclose(seen[1][:, 0], torch.zeros_like(seen[1][:, 0]))
    assert torch.allclose(
        seen[1][:, 1:],
        output.memory_states_per_pass[0][:, :-1],
    )


def test_scalar_gate_starts_at_point_two():
    model = MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=4,
            vocab_size=17,
            n_layer=2,
            n_head=1,
            n_embd=8,
            n_pass=2,
            memory_tape_gate="scalar",
        )
    )
    stats = model.memory_gate_stats()
    assert stats["effective"] == pytest.approx([0.2, 0.2])


def test_first_generated_token_matches_between_inference_modes():
    torch.manual_seed(17)
    model = MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=8,
            vocab_size=31,
            n_layer=2,
            n_head=2,
            n_embd=16,
            n_pass=3,
        )
    ).eval()
    prompt = torch.randint(0, 31, (2, 6))
    exact = model.generate(
        prompt.clone(),
        1,
        do_sample=False,
        inference_mode="recompute",
    )
    approximate = model.generate(
        prompt.clone(),
        1,
        do_sample=False,
        inference_mode="final_pass",
    )
    assert torch.equal(exact, approximate)


def test_final_pass_generation_has_no_memory_source_flag():
    parameters = inspect.signature(MemoryTapeTransformer.generate).parameters
    assert "cache_source" not in parameters


def test_final_pass_generation_reuses_last_pass_memory():
    model = MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=8,
            vocab_size=11,
            n_layer=1,
            n_head=1,
            n_embd=4,
            n_pass=2,
        )
    ).eval()
    seen_memory = []

    def fake_forward(self, ids):
        shape = (ids.size(0), ids.size(1), self.config.n_embd)
        logits = torch.zeros((*ids.shape, self.config.vocab_size))
        logits[..., 2] = 1.0
        hidden = torch.zeros(shape)
        return MemoryTapeOutput(
            logits_per_pass=(logits, logits),
            hidden_states_per_pass=(hidden, hidden),
            memory_states_per_pass=(
                torch.full(shape, 3.0),
                torch.full(shape, 7.0),
            ),
        )

    def recording_pass(self, token_stream, memory_tape):
        seen_memory.append(memory_tape.detach().clone())
        return token_stream

    model.forward = types.MethodType(fake_forward, model)
    model._run_full_pass = types.MethodType(recording_pass, model)
    model.generate(
        torch.tensor([[4, 5]]),
        2,
        do_sample=False,
        inference_mode="final_pass",
    )

    assert len(seen_memory) == 1
    assert torch.all(seen_memory[0][:, 1:] == 7.0)
