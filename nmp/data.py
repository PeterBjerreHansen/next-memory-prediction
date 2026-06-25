from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from tokenizers import Tokenizer

from .config import DataConfig
from .tokenizer_asset import materialize_tokenizer


@dataclass(frozen=True)
class TextBatch:
    tokens: torch.Tensor
    lengths: torch.Tensor

    def to(self, device: str | torch.device) -> "TextBatch":
        return TextBatch(tokens=self.tokens.to(device), lengths=self.lengths.to(device))


class TinyStoriesTokenizer:
    def __init__(self, path: str | Path | None = None):
        tokenizer_path = Path(path) if path is not None else materialize_tokenizer()
        self.path = tokenizer_path
        self.backend = Tokenizer.from_file(str(tokenizer_path))
        self.pad_id = self.backend.token_to_id("<|pad|>")
        self.eos_id = self.backend.token_to_id("<|eos|>")
        if self.pad_id is None or self.eos_id is None:
            raise ValueError("tokenizer must define <|pad|> and <|eos|>")

    @property
    def vocab_size(self) -> int:
        return self.backend.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self.backend.decode(list(map(int, ids)), skip_special_tokens=skip_special_tokens)


class TextCorpus:
    def __init__(self, rows: Any, *, text_field: str = "text"):
        self.rows = rows
        self.text_field = text_field

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> str:
        row = self.rows[index]
        if isinstance(row, str):
            return row
        value = row[self.text_field]
        return str(value)


def _load_local_rows(path: str | Path, text_field: str) -> list[str]:
    path = Path(path)
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    rows.append(str(payload[text_field]))
        else:
            rows.extend(line.rstrip("\n") for line in handle if line.strip())
    if not rows:
        raise ValueError(f"no text rows found in {path}")
    return rows


def load_corpora(config: DataConfig) -> tuple[TextCorpus, TextCorpus]:
    if config.source == "local":
        return (
            TextCorpus(_load_local_rows(config.train_file, config.text_field)),
            TextCorpus(_load_local_rows(config.val_file, config.text_field)),
        )

    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset_name,
        split="train",
        revision=config.dataset_revision,
        cache_dir=config.cache_dir,
    )
    split = dataset.train_test_split(
        test_size=config.validation_size,
        seed=config.split_seed,
        shuffle=True,
    )
    return (
        TextCorpus(split["train"], text_field=config.text_field),
        TextCorpus(split["test"], text_field=config.text_field),
    )


def encode_texts(
    texts: Sequence[str],
    *,
    tokenizer: TinyStoriesTokenizer,
    block_size: int,
) -> TextBatch:
    tokens = torch.full(
        (len(texts), block_size),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    lengths = torch.zeros(len(texts), dtype=torch.long)
    for row, text in enumerate(texts):
        encoded = tokenizer.encode(text)
        if len(encoded) <= block_size - 1:
            ids = [*encoded, tokenizer.eos_id]
        else:
            ids = encoded[:block_size]
        length = len(ids)
        tokens[row, :length] = torch.tensor(ids, dtype=torch.long)
        lengths[row] = length
    return TextBatch(tokens=tokens, lengths=lengths)


class StatefulBatchStream:
    """Random-with-replacement batch stream with an exactly serializable RNG."""

    def __init__(
        self,
        corpus: TextCorpus,
        tokenizer: TinyStoriesTokenizer,
        *,
        batch_size: int,
        block_size: int,
        seed: int,
    ):
        if len(corpus) < 1:
            raise ValueError("corpus must contain at least one row")
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = random.Random(seed)
        self.batches_emitted = 0

    def next_batch(self) -> TextBatch:
        indices = [self.rng.randrange(len(self.corpus)) for _ in range(self.batch_size)]
        self.batches_emitted += 1
        return encode_texts(
            [self.corpus[index] for index in indices],
            tokenizer=self.tokenizer,
            block_size=self.block_size,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.getstate(),
            "batches_emitted": self.batches_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(state["rng_state"])
        self.batches_emitted = int(state.get("batches_emitted", 0))


def sequential_batches(
    corpus: TextCorpus,
    tokenizer: TinyStoriesTokenizer,
    *,
    batch_size: int,
    block_size: int,
    num_batches: int,
) -> list[TextBatch]:
    batches = []
    cursor = 0
    for _ in range(num_batches):
        texts = [corpus[(cursor + offset) % len(corpus)] for offset in range(batch_size)]
        cursor = (cursor + batch_size) % len(corpus)
        batches.append(encode_texts(texts, tokenizer=tokenizer, block_size=block_size))
    return batches
