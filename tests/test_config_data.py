from __future__ import annotations

import torch

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


def test_one_story_per_row_truncates_appends_eos_and_pads():
    tokenizer = TinyStoriesTokenizer()
    batch = encode_texts(
        ["Once upon a time.", "A " * 100],
        tokenizer=tokenizer,
        block_size=8,
    )
    assert batch.tokens.shape == (2, 8)
    assert batch.tokens[0, batch.lengths[0] - 1] == tokenizer.eos_id
    assert batch.tokens[1, -1] == tokenizer.eos_id
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
