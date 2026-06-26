from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from conftest import make_config
from nmp.artifacts import artifacts_for
from nmp.countdown import (
    check_countdown_solution,
    check_countdown_solution_nextlat_compat,
)
from nmp.data import SequenceBatch
from nmp.evaluation import evaluate_batches
from nmp.evaluation import evaluate_run
from nmp.models import (
    LatentTransitionPredictor,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    TransformerOutput,
)
from nmp.objectives import compute_loss, next_token_loss
from nmp.training import train_experiment


def test_countdown_solution_checker_accepts_valid_duplicate_operands():
    checked = check_countdown_solution(
        input_numbers=[2, 3, 4, 5],
        target=14,
        prediction="2+3=5,5+5=10,10+4=14",
        num_equations=3,
    )
    assert checked.correct
    assert checked.valid_equations == (True, True, True)


def test_nextlat_compat_checker_allows_reusing_consumed_numbers():
    prediction = "2+3=5,5+5=10,10+5=15"
    strict = check_countdown_solution(
        input_numbers=[2, 3, 4, 5],
        target=15,
        prediction=prediction,
        num_equations=3,
    )
    compat = check_countdown_solution_nextlat_compat(
        input_numbers=[2, 3, 4, 5],
        target=15,
        prediction=prediction,
        num_equations=3,
    )

    assert not strict.correct
    assert strict.valid_equations == (True, True, False)
    assert compat.correct
    assert compat.valid_equations == (True, True, True)


def test_nextlat_compat_checker_uses_set_multiplicity():
    prediction = "2+3=5,5+5=10,10+4=14"
    strict = check_countdown_solution(
        input_numbers=[2, 3, 4, 6],
        target=14,
        prediction=prediction,
        num_equations=3,
    )
    compat = check_countdown_solution_nextlat_compat(
        input_numbers=[2, 3, 4, 6],
        target=14,
        prediction=prediction,
        num_equations=3,
    )

    assert not strict.correct
    assert compat.correct


@pytest.mark.parametrize(
    "prediction",
    [
        "2+3=5,5+5=10",
        "2+3=6,6+5=11,11+4=15",
        "9+3=12,12+5=17,17+4=21",
        "5/2=2,2+3=5,5+4=9",
        "2+3=5,5+5=10,10+4=15",
    ],
)
def test_countdown_solution_checker_rejects_invalid_outputs(prediction):
    checked = check_countdown_solution(
        input_numbers=[2, 3, 4, 5],
        target=14,
        prediction=prediction,
        num_equations=3,
    )
    assert not checked.correct


def test_nextlat_compat_checker_still_rejects_invalid_division():
    checked = check_countdown_solution_nextlat_compat(
        input_numbers=[2, 3, 4, 5],
        target=9,
        prediction="5/2=2,2+3=5,5+4=9",
        num_equations=3,
    )

    assert not checked.correct


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
        SequenceBatch(
            tokens=torch.tensor([[2, 1, 0, 0]]),
            lengths=torch.tensor([2]),
            target_mask=torch.tensor([[True, False, False]]),
            prompt_lengths=torch.tensor([1]),
        ),
        SequenceBatch(
            tokens=torch.tensor([[3, 4, 2, 1]]),
            lengths=torch.tensor([4]),
            target_mask=torch.tensor([[True, True, True]]),
            prompt_lengths=torch.tensor([1]),
        ),
    ]
    config = SimpleNamespace(
        model=SimpleNamespace(variant="transformer_ntp"),
        training=SimpleNamespace(precision="float32"),
        objective=SimpleNamespace(
            transition=SimpleNamespace(lambda_transition=1.0),
            ntp_pass_weights=None,
        ),
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


def test_validation_transition_loss_is_weighted_by_valid_transitions():
    torch.manual_seed(19)
    model = MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=4,
            vocab_size=9,
            n_layer=1,
            n_head=1,
            n_embd=8,
            n_pass=2,
        )
    )
    predictor = LatentTransitionPredictor(8)
    batches = [
        SequenceBatch(
            tokens=torch.tensor([[2, 3, 1, 0]]),
            lengths=torch.tensor([3]),
            target_mask=torch.tensor([[True, True, False]]),
            prompt_lengths=torch.tensor([1]),
        ),
        SequenceBatch(
            tokens=torch.tensor([[4, 5, 6, 7]]),
            lengths=torch.tensor([4]),
            target_mask=torch.tensor([[True, True, True]]),
            prompt_lengths=torch.tensor([1]),
        ),
    ]
    config = SimpleNamespace(
        model=SimpleNamespace(variant="memory_tape_nmp"),
        training=SimpleNamespace(precision="float32"),
        objective=SimpleNamespace(
            transition=SimpleNamespace(lambda_transition=1.0),
            ntp_pass_weights=None,
        ),
    )
    tokenizer = SimpleNamespace(pad_id=0, eos_id=1)
    batch_losses = []
    for batch in batches:
        output = model(batch.tokens)
        losses = compute_loss(
            variant="memory_tape_nmp",
            model=model,
            output=output,
            tokens=batch.tokens,
            pad_token_id=0,
            eos_token_id=1,
            predictor=predictor,
        )
        batch_losses.append(float(losses.transition_prediction.detach()))
    expected = (batch_losses[0] + 3 * batch_losses[1]) / 4

    metrics = evaluate_batches(
        config=config,
        model=model,
        predictor=predictor,
        batches=batches,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
    )

    assert metrics["transition_count"] == 4
    assert metrics["transition_prediction_loss"] == pytest.approx(expected)


def test_evaluate_run_uses_training_eval_batches_for_reported_loss(
    local_countdown_files,
    tmp_path,
):
    train_file, val_file = local_countdown_files
    config = make_config(
        "transformer_ntp",
        train_file,
        val_file,
        train_steps=1,
    )
    config.training.eval_batches = 2
    config.evaluation.diagnostic_batches = 1
    run_dir = tmp_path / "run"
    train_experiment(config, run_dir=run_dir)

    result = evaluate_run(run_dir, device_override="cpu")
    protocol = result["protocol"]
    loss = result["loss"]
    diagnostic_loss = result["diagnostic_loss"]

    assert protocol["config_source"] == "checkpoint"
    assert protocol["loss_source"] == "training.eval_batches"
    assert protocol["loss_batches"] == 2
    assert protocol["diagnostic_batches"] == 1
    assert loss["ntp_tokens"] > diagnostic_loss["ntp_tokens"]
    assert artifacts_for(run_dir).evaluation_path.exists()
