from __future__ import annotations

import torch

from nmp.cli.train import parse_args, resolve_config
from nmp.config import ExperimentConfig, load_config
from nmp.data import (
    StatefulBatchStream,
    TinyStoriesTokenizer,
    encode_texts,
    load_corpora,
)


def test_tokenizer_is_pinned_1000_token_asset():
    tokenizer = TinyStoriesTokenizer()
    assert tokenizer.vocab_size == 1000
    assert tokenizer.pad_id == 0
    assert tokenizer.eos_id == 1
    assert tokenizer.encode("Once upon a time.")


def test_one_story_per_row_only_appends_eos_at_true_end():
    tokenizer = TinyStoriesTokenizer()
    short_text = "Once upon a time."
    long_text = "A " * 100
    batch = encode_texts(
        [short_text, long_text],
        tokenizer=tokenizer,
        block_size=8,
    )
    assert batch.tokens.shape == (2, 8)
    assert batch.tokens[0, batch.lengths[0] - 1] == tokenizer.eos_id
    assert batch.lengths[1] == 8
    assert torch.equal(
        batch.tokens[1],
        torch.tensor(tokenizer.encode(long_text)[:8]),
    )
    assert tokenizer.eos_id not in batch.tokens[1]
    assert torch.all(
        batch.tokens[0, batch.lengths[0] :] == tokenizer.pad_id
    )


def test_local_corpus_and_stream_state_are_reproducible(local_story_files):
    train, val = local_story_files
    config = ExperimentConfig.from_dict(
        {
            "name": "data",
            "seed": 0,
            "model": {
                "variant": "transformer_ntp",
                "block_size": 16,
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
    tokenizer = TinyStoriesTokenizer()
    stream = StatefulBatchStream(
        train_corpus,
        tokenizer,
        batch_size=2,
        block_size=16,
        seed=9,
    )
    _ = stream.next_batch()
    state = stream.state_dict()
    expected = stream.next_batch().tokens
    restored = StatefulBatchStream(
        train_corpus,
        tokenizer,
        batch_size=2,
        block_size=16,
        seed=999,
    )
    restored.load_state_dict(state)
    assert torch.equal(expected, restored.next_batch().tokens)


def test_all_shipped_configs_validate():
    from pathlib import Path

    for path in Path("configs").glob("*.yaml"):
        load_config(path)


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
                "n_pass": 2,
            },
            "training": {"train_steps": 1, "micro_batch_size": 1},
        }
    )
    assert config.model.variant == "memory_tape_hidden_transition"


def test_non_scalar_memory_gate_modes_are_rejected():
    import pytest

    with pytest.raises(ValueError, match="fixed to scalar"):
        ExperimentConfig.from_dict(
            {
                "name": "bad-gate",
                "seed": 0,
                "model": {
                    "variant": "memory_tape_ntp",
                    "block_size": 8,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 8,
                    "n_pass": 2,
                    "memory_tape_gate": "tanh",
                },
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
                "n_pass": 4,
            },
            "objective": {"ntp_pass_weights": [0, 0, 0.5, 0.5]},
            "training": {"train_steps": 1, "micro_batch_size": 1},
        }
    )
    assert config.objective.ntp_pass_weights == [0.0, 0.0, 0.5, 0.5]


def test_ntp_pass_weights_cli_parses_json_list(tmp_path):
    args = parse_args(
        [
            "--config",
            "configs/development.yaml",
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
