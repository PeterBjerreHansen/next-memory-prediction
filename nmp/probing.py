from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from .artifacts import append_jsonl
from .data import StatefulBatchStream, load_corpora, sequential_batches
from .evaluation import load_run
from .models import MemoryTapeOutput


class LinearProbeBank(nn.Module):
    def __init__(
        self,
        *,
        sources: list[str],
        offsets: list[int],
        n_embd: int,
        vocab_size: int,
    ):
        super().__init__()
        self.sources = tuple(sources)
        self.offsets = tuple(offsets)
        self.heads = nn.ModuleDict(
            {
                self.key(source, offset): nn.Linear(
                    n_embd,
                    vocab_size,
                    bias=True,
                )
                for source in sources
                for offset in offsets
            }
        )

    @staticmethod
    def key(source: str, offset: int) -> str:
        return f"{source}__{offset}"

    def head(self, source: str, offset: int) -> nn.Linear:
        return self.heads[self.key(source, offset)]


@torch.no_grad()
def extract_representations(model, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    output = model(tokens)
    representations = {"hidden": output.hidden_states.detach()}
    if isinstance(output, MemoryTapeOutput):
        representations["memory"] = output.memory_states.detach()
    return representations


def probe_loss(
    bank: LinearProbeBank,
    representations: dict[str, torch.Tensor],
    tokens: torch.Tensor,
    *,
    pad_id: int,
) -> tuple[torch.Tensor, int]:
    losses = []
    examples = 0
    for source, states in representations.items():
        for offset in bank.offsets:
            if offset >= states.size(1):
                continue
            features = states[:, :-offset, :]
            targets = tokens[:, offset:]
            valid = targets != pad_id
            if not bool(valid.any()):
                continue
            logits = bank.head(source, offset)(features[valid])
            losses.append(F.cross_entropy(logits, targets[valid]))
            examples += int(valid.sum().item())
    if not losses:
        return tokens.sum() * 0.0, 0
    return torch.stack(losses).mean(), examples


@torch.no_grad()
def evaluate_probes(
    bank: LinearProbeBank,
    model,
    batches,
    *,
    pad_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    totals: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {"loss": 0.0, "correct": 0.0, "count": 0.0}
    )
    model.eval()
    bank.eval()
    for cpu_batch in batches:
        tokens = cpu_batch.tokens.to(device)
        representations = extract_representations(model, tokens)
        for source, states in representations.items():
            for offset in bank.offsets:
                if offset >= states.size(1):
                    continue
                features = states[:, :-offset, :]
                targets = tokens[:, offset:]
                valid = targets != pad_id
                count = int(valid.sum().item())
                if count == 0:
                    continue
                logits = bank.head(source, offset)(features[valid])
                loss = F.cross_entropy(logits, targets[valid], reduction="sum")
                predictions = logits.argmax(dim=-1)
                entry = totals[(source, offset)]
                entry["loss"] += float(loss.cpu())
                entry["correct"] += float((predictions == targets[valid]).sum().cpu())
                entry["count"] += count
    rows = []
    for (source, offset), values in sorted(totals.items()):
        count = max(values["count"], 1.0)
        rows.append(
            {
                "source": source,
                "offset": offset,
                "cross_entropy": values["loss"] / count,
                "accuracy": values["correct"] / count,
                "tokens": int(values["count"]),
            }
        )
    return rows


def train_probes(
    run_dir: str | Path,
    *,
    steps: int | None = None,
    device_override: str | None = None,
) -> list[dict[str, Any]]:
    (
        artifacts,
        config,
        tokenizer,
        model,
        _predictor,
        checkpoint,
        device,
    ) = load_run(run_dir, device_override=device_override)
    if artifacts.probe_metrics_path.exists():
        artifacts.probe_metrics_path.unlink()
    train_corpus, val_corpus = load_corpora(config.data)
    probe_steps = steps or config.evaluation.probe_steps
    stream = StatefulBatchStream(
        train_corpus,
        tokenizer,
        batch_size=config.evaluation.probe_batch_size,
        block_size=config.model.block_size,
        seed=config.seed + 9001,
    )
    sources = ["hidden"]
    if config.model.variant != "transformer_ntp":
        sources.append("memory")
    offsets = [
        offset
        for offset in config.evaluation.probe_offsets
        if offset < config.model.block_size
    ]
    bank = LinearProbeBank(
        sources=sources,
        offsets=offsets,
        n_embd=config.model.n_embd,
        vocab_size=tokenizer.vocab_size,
    ).to(device)
    optimizer = torch.optim.AdamW(bank.parameters(), lr=3e-4, weight_decay=0.0)
    model.eval()

    for step in range(1, probe_steps + 1):
        batch = stream.next_batch().to(device)
        with torch.no_grad():
            representations = extract_representations(model, batch.tokens)
        optimizer.zero_grad(set_to_none=True)
        loss, examples = probe_loss(
            bank,
            representations,
            batch.tokens,
            pad_id=tokenizer.pad_id,
        )
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(probe_steps // 10, 1) == 0:
            append_jsonl(
                artifacts.probe_metrics_path,
                {
                    "event": "probe_train",
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "examples": examples,
                },
            )

    val_batches = sequential_batches(
        val_corpus,
        tokenizer,
        batch_size=config.evaluation.probe_batch_size,
        block_size=config.model.block_size,
        num_batches=config.evaluation.diagnostic_batches,
    )
    rows = evaluate_probes(
        bank,
        model,
        val_batches,
        pad_id=tokenizer.pad_id,
        device=device,
    )
    for row in rows:
        append_jsonl(
            artifacts.probe_metrics_path,
            {
                "event": "probe_validation",
                "checkpoint_step": int(checkpoint["step"]),
                **row,
            },
        )
    torch.save(
        {
            "checkpoint_step": int(checkpoint["step"]),
            "sources": sources,
            "offsets": offsets,
            "state_dict": bank.state_dict(),
        },
        artifacts.probe_checkpoint_path,
    )
    return rows
