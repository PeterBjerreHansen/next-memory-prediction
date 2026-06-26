from __future__ import annotations

import torch
import pytest

from nmp.cli.train import parse_args, resolve_config
from nmp.config import ExperimentConfig, load_config
from nmp.countdown import (
    CountdownTokenizer,
    check_countdown_solution,
    generate_countdown_example,
    target_splits,
)
from nmp.data import (
    StatefulBatchStream,
    encode_sequences,
    load_corpora,
    make_tokenizer,
)


def test_countdown_tokenizer_has_atomic_numbers_and_special_ids():
    tokenizer = CountdownTokenizer(max_intermediate=10_000)
    assert tokenizer.encode("1234") == [1234]
    assert tokenizer.pipe_id == 10_001
    assert tokenizer.pad_id != tokenizer.eos_id
    assert tokenizer.vocab_size == tokenizer.pad_id + 1


def test_countdown_encoding_skips_prompt_commas_and_masks_prompt_pause():
    tokenizer = CountdownTokenizer()
    batch = encode_sequences(
        ["2,3,4,5,14|2+3=5,5+5=10,10+4=14"],
        tokenizer=tokenizer,
        block_size=40,
        num_pause_tokens=8,
    )
    prompt_length = int(batch.prompt_lengths[0])
    assert batch.tokens.shape == (1, 40)
    assert batch.tokens[0, :5].tolist() == [2, 3, 4, 5, 14]
    assert batch.tokens[0, 5:prompt_length].tolist() == [tokenizer.pipe_id] * 8
    assert batch.tokens[0, batch.lengths[0] - 1] == tokenizer.eos_id
    assert torch.all(batch.tokens[0, batch.lengths[0] :] == tokenizer.pad_id)
    assert not bool(batch.target_mask[0, : prompt_length - 1].any())
    assert bool(batch.target_mask[0, prompt_length - 1])


def test_generated_countdown_splits_are_deterministic_and_valid():
    train_targets, heldout_targets = target_splits(
        min_target=10,
        max_target=100,
        seed=444,
    )
    assert set(train_targets).isdisjoint(heldout_targets)
    import random

    row = generate_countdown_example(
        random.Random(123),
        target_pool=heldout_targets,
        input_numbers=4,
        max_target=100,
        max_intermediate=10_000,
    )
    prompt, solution = row.split("|")
    numbers = [int(item) for item in prompt.split(",")]
    checked = check_countdown_solution(
        input_numbers=numbers[:4],
        target=numbers[4],
        prediction=solution,
        num_equations=3,
    )
    assert numbers[4] in heldout_targets
    assert checked.correct


def test_local_corpus_and_stream_state_are_reproducible(local_countdown_files):
    train, val = local_countdown_files
    config = ExperimentConfig.from_dict(
        {
            "name": "data",
            "seed": 0,
            "model": {
                "variant": "transformer_ntp",
                "block_size": 40,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
            },
            "data": {
                "source": "local",
                "train_file": str(train),
                "val_file": str(val),
            },
            "training": {
                "train_steps": 1,
                "micro_batch_size": 2,
            },
        }
    )
    train_corpus, _ = load_corpora(config.data)
    tokenizer = make_tokenizer(config.data)
    stream = StatefulBatchStream(
        train_corpus,
        tokenizer,
        batch_size=2,
        block_size=40,
        num_pause_tokens=config.data.num_pause_tokens,
        seed=9,
    )
    _ = stream.next_batch()
    state = stream.state_dict()
    expected = stream.next_batch().tokens
    restored = StatefulBatchStream(
        train_corpus,
        tokenizer,
        batch_size=2,
        block_size=40,
        num_pause_tokens=config.data.num_pause_tokens,
        seed=999,
    )
    restored.load_state_dict(state)
    assert torch.equal(expected, restored.next_batch().tokens)


def test_all_shipped_configs_validate():
    from pathlib import Path

    for path in Path("configs/scales").glob("*.yaml"):
        load_config(path)
    from nmp.experiment_plan import expand_plan, load_experiment_plan

    for path in Path("configs/experiments").glob("*.yaml"):
        plan = load_experiment_plan(path)
        selected_lambdas = None
        if path.name == "round1_reference_template.yaml":
            selected_lambdas = {
                "memory_tape_nmp": 0.3,
                "memory_tape_hidden_transition": 1.0,
                "memory_tape_hidden_transition_kl": 1.0,
            }
        assert expand_plan(plan, selected_lambdas=selected_lambdas)


def test_legacy_nextlat_variant_alias_resolves_canonically():
    config = ExperimentConfig.from_dict(
        {
            "name": "legacy-variant",
            "seed": 0,
            "model": {
                "variant": "memory_tape_nextlat_no_kl",
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "memory": {"n_pass": 2},
            },
            "training": {"train_steps": 1, "micro_batch_size": 1},
        }
    )
    assert config.model.variant == "memory_tape_hidden_transition"
    assert config.model.memory.n_pass == 2


def test_hidden_transition_kl_variant_validates():
    config = ExperimentConfig.from_dict(
        {
            "name": "hidden-kl",
            "seed": 0,
            "model": {
                "variant": "memory_tape_hidden_transition_kl",
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "memory": {"n_pass": 2},
            },
            "objective": {"transition": {"lambda_transition": 0.3, "lambda_kl": 1.0}},
            "training": {"train_steps": 1, "micro_batch_size": 1},
        }
    )
    assert config.model.variant == "memory_tape_hidden_transition_kl"
    assert config.objective.transition.target == "hidden"
    assert config.objective.transition.horizon == 1
    assert config.objective.transition.lambda_kl == 1.0
    assert config.objective.transition.lambda_ce == 0.0


def test_transition_target_must_match_variant():
    with pytest.raises(ValueError, match="transition.target"):
        ExperimentConfig.from_dict(
            {
                "name": "bad-target",
                "seed": 0,
                "model": {
                    "variant": "memory_tape_hidden_transition",
                    "block_size": 8,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 8,
                    "memory": {"n_pass": 2},
                },
                "objective": {"transition": {"target": "memory"}},
                "training": {"train_steps": 1, "micro_batch_size": 1},
            }
        )


def test_ntp_pass_weights_validate_against_memory_tape_pass_count():
    config = ExperimentConfig.from_dict(
        {
            "name": "weighted",
            "seed": 0,
            "model": {
                "variant": "memory_tape_ntp",
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "memory": {"n_pass": 4},
            },
            "objective": {"ntp_pass_weights": [0, 0, 0.5, 0.5]},
            "training": {"train_steps": 1, "micro_batch_size": 1},
        }
    )
    assert config.objective.ntp_pass_weights == [0.0, 0.0, 0.5, 0.5]
    assert config.to_dict()["model"]["memory"]["n_pass"] == 4


def test_legacy_flat_memory_and_transition_fields_migrate_to_nested_config():
    config = ExperimentConfig.from_dict(
        {
            "name": "legacy-flat",
            "seed": 0,
            "model": {
                "variant": "memory_tape_nmp",
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "n_pass": 3,
            },
            "objective": {
                "transition_horizon": 1,
                "lambda_transition": 0.3,
                "lambda_kl": 0.7,
                "lambda_ce": 0.2,
                "transition_target": "memory",
                "dynamics_projection_factor": 1.7,
            },
            "training": {"train_steps": 1, "micro_batch_size": 1},
        }
    )

    resolved = config.to_dict()
    assert config.model.memory.n_pass == 3
    assert config.objective.transition.horizon == 1
    assert config.objective.transition.target == "memory"
    assert config.objective.transition.lambda_transition == 0.3
    assert config.objective.transition.lambda_kl == 0.7
    assert config.objective.transition.lambda_ce == 0.2
    assert config.objective.transition.projection_factor == 1.7
    assert "n_pass" not in resolved["model"]
    assert "transition_horizon" not in resolved["objective"]
    assert "lambda_transition" not in resolved["objective"]
    assert "lambda_kl" not in resolved["objective"]
    assert "lambda_ce" not in resolved["objective"]
    assert "transition_target" not in resolved["objective"]
    assert "dynamics_projection_factor" not in resolved["objective"]


def test_ntp_pass_weights_cli_parses_json_list(tmp_path):
    args = parse_args(
        [
            "--config",
            "configs/scales/development.yaml",
            "--run-dir",
            str(tmp_path / "run"),
            "--variant",
            "memory_tape_ntp",
            "--ntp-pass-weights",
            "[0.0, 0.0, 0.5, 0.5]",
        ]
    )
    config, _ = resolve_config(args)
    assert config.objective.ntp_pass_weights == [0.0, 0.0, 0.5, 0.5]


def test_transition_objective_cli_overrides(tmp_path):
    args = parse_args(
        [
            "--config",
            "configs/scales/development.yaml",
            "--run-dir",
            str(tmp_path / "run"),
            "--variant",
            "memory_tape_hidden_transition_kl",
            "--lambda-transition",
            "0.3",
            "--lambda-kl",
            "0.7",
            "--lambda-ce",
            "0.2",
            "--transition-horizon",
            "1",
            "--transition-target",
            "hidden",
        ]
    )
    config, _ = resolve_config(args)
    transition = config.objective.transition
    assert transition.target == "hidden"
    assert transition.horizon == 1
    assert transition.lambda_transition == 0.3
    assert transition.lambda_kl == 0.7
    assert transition.lambda_ce == 0.2


def test_resume_rejects_config_mutating_overrides(tmp_path):
    args = parse_args(
        [
            "--resume-from",
            str(tmp_path / "latest.pt"),
            "--variant",
            "memory_tape_nmp",
        ]
    )
    with pytest.raises(ValueError, match="only accepts --steps and --device"):
        resolve_config(args)
