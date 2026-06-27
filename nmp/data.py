from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch

from .config import DataConfig
from .countdown import (
    CountdownTokenizer,
    generate_countdown_example,
    target_splits,
)


@dataclass(frozen=True)
class SequenceBatch:
    tokens: torch.Tensor
    lengths: torch.Tensor
    target_mask: torch.Tensor
    prompt_lengths: torch.Tensor

    def to(self, device: str | torch.device) -> "SequenceBatch":
        return SequenceBatch(
            tokens=self.tokens.to(device),
            lengths=self.lengths.to(device),
            target_mask=self.target_mask.to(device),
            prompt_lengths=self.prompt_lengths.to(device),
        )


class CountdownCorpus:
    def __init__(self, rows: Sequence[str]):
        if not rows:
            raise ValueError("Countdown corpus must contain at least one row")
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> str:
        return self.rows[index]


def _load_local_rows(path: str | Path) -> list[str]:
    path = Path(path)
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    rows.append(str(payload.get("text", payload.get("sequence"))))
        else:
            rows.extend(line.strip() for line in handle if line.strip())
    if not rows:
        raise ValueError(f"no Countdown rows found in {path}")
    return rows


def _generated_rows(
    *,
    config: DataConfig,
    split: str,
    sample_count: int,
    target_pool: Sequence[int],
) -> list[str]:
    split_offsets = {"train": 0, "val": 1_000_000, "generalization": 2_000_000}
    rng = random.Random(config.split_seed + split_offsets[split])
    return [
        generate_countdown_example(
            rng,
            target_pool=target_pool,
            input_numbers=config.countdown_input_numbers,
            max_target=config.countdown_max_target,
            max_intermediate=config.countdown_max_intermediate,
        )
        for _ in range(sample_count)
    ]


def load_corpora(config: DataConfig) -> tuple[CountdownCorpus, CountdownCorpus]:
    if config.source == "local":
        return (
            CountdownCorpus(_load_local_rows(config.train_file)),
            CountdownCorpus(_load_local_rows(config.val_file)),
        )

    train_targets, heldout_targets = target_splits(
        min_target=config.countdown_min_target,
        max_target=config.countdown_max_target,
        seed=config.split_seed,
    )
    del heldout_targets
    return (
        CountdownCorpus(
            _generated_rows(
                config=config,
                split="train",
                sample_count=config.train_samples,
                target_pool=train_targets,
            )
        ),
        CountdownCorpus(
            _generated_rows(
                config=config,
                split="val",
                sample_count=config.val_samples,
                target_pool=train_targets,
            )
        ),
    )


def load_generalization_corpus(config: DataConfig) -> CountdownCorpus | None:
    if config.source == "local":
        if config.generalization_file is None:
            return None
        return CountdownCorpus(_load_local_rows(config.generalization_file))

    if config.generalization_samples < 1:
        return None
    _, heldout_targets = target_splits(
        min_target=config.countdown_min_target,
        max_target=config.countdown_max_target,
        seed=config.split_seed,
    )
    return CountdownCorpus(
        _generated_rows(
            config=config,
            split="generalization",
            sample_count=config.generalization_samples,
            target_pool=heldout_targets,
        )
    )


def make_tokenizer(config: DataConfig) -> CountdownTokenizer:
    return CountdownTokenizer(max_intermediate=config.countdown_max_intermediate)


def encode_sequences(
    rows: Sequence[str],
    *,
    tokenizer: CountdownTokenizer,
    block_size: int,
    num_pause_tokens: int,
) -> SequenceBatch:
    tokens = torch.full(
        (len(rows), block_size),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    lengths = torch.zeros(len(rows), dtype=torch.long)
    prompt_lengths = torch.zeros(len(rows), dtype=torch.long)
    target_mask = torch.zeros((len(rows), block_size - 1), dtype=torch.bool)

    for row, text in enumerate(rows):
        encoded, prompt_length = tokenizer.tokenize(
            text,
            num_pause_tokens=num_pause_tokens,
        )
        if len(encoded) > block_size:
            raise ValueError(
                f"encoded Countdown sequence length {len(encoded)} exceeds block size "
                f"{block_size}: {text}"
            )
        length = len(encoded)
        tokens[row, :length] = torch.tensor(encoded, dtype=torch.long)
        lengths[row] = length
        prompt_lengths[row] = prompt_length
        target_start = max(prompt_length - 1, 0)
        target_stop = max(length - 1, 0)
        target_mask[row, target_start:target_stop] = True

    return SequenceBatch(
        tokens=tokens,
        lengths=lengths,
        target_mask=target_mask,
        prompt_lengths=prompt_lengths,
    )


class StatefulBatchStream:
    """Random-with-replacement batch stream with an exactly serializable RNG."""

    def __init__(
        self,
        corpus: CountdownCorpus,
        tokenizer: CountdownTokenizer,
        *,
        batch_size: int,
        block_size: int,
        num_pause_tokens: int,
        seed: int,
    ):
        if len(corpus) < 1:
            raise ValueError("corpus must contain at least one row")
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_pause_tokens = num_pause_tokens
        self.rng = random.Random(seed)
        self.batches_emitted = 0

    def next_batch(self) -> SequenceBatch:
        indices = [self.rng.randrange(len(self.corpus)) for _ in range(self.batch_size)]
        self.batches_emitted += 1
        return encode_sequences(
            [self.corpus[index] for index in indices],
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            num_pause_tokens=self.num_pause_tokens,
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
    corpus: CountdownCorpus,
    tokenizer: CountdownTokenizer,
    *,
    batch_size: int,
    block_size: int,
    num_pause_tokens: int,
    num_batches: int | None,
) -> list[SequenceBatch]:
    if num_batches is not None and num_batches < 1:
        raise ValueError("num_batches must be positive or None")
    batches = []
    if num_batches is None:
        for cursor in range(0, len(corpus), batch_size):
            rows = [
                corpus[index]
                for index in range(cursor, min(cursor + batch_size, len(corpus)))
            ]
            batches.append(
                encode_sequences(
                    rows,
                    tokenizer=tokenizer,
                    block_size=block_size,
                    num_pause_tokens=num_pause_tokens,
                )
            )
        return batches

    cursor = 0
    for _ in range(num_batches):
        rows = [corpus[(cursor + offset) % len(corpus)] for offset in range(batch_size)]
        cursor = (cursor + batch_size) % len(corpus)
        batches.append(
            encode_sequences(
                rows,
                tokenizer=tokenizer,
                block_size=block_size,
                num_pause_tokens=num_pause_tokens,
            )
        )
    return batches
