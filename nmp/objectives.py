from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .models import (
    CausalTransformer,
    MemoryDynamicsPredictor,
    MemoryTapeOutput,
    MemoryTapeTransformer,
    TransformerOutput,
)


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    weighted_ntp: torch.Tensor
    final_pass_nll: torch.Tensor
    pass_nlls: tuple[torch.Tensor, ...]
    memory_prediction: torch.Tensor | None

    def detached_metrics(self) -> dict[str, object]:
        return {
            "loss": float(self.total.detach().cpu()),
            "weighted_ntp_loss": float(self.weighted_ntp.detach().cpu()),
            "final_pass_nll": float(self.final_pass_nll.detach().cpu()),
            "pass_nlls": [float(item.detach().cpu()) for item in self.pass_nlls],
            "memory_prediction_loss": (
                None
                if self.memory_prediction is None
                else float(self.memory_prediction.detach().cpu())
            ),
        }


def next_token_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    pad_token_id: int,
) -> torch.Tensor:
    predictions = logits[:, :-1, :]
    targets = tokens[:, 1:]
    return F.cross_entropy(
        predictions.reshape(-1, predictions.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_token_id,
    )


def temporal_memory_prediction_loss(
    model: MemoryTapeTransformer,
    predictor: MemoryDynamicsPredictor,
    memory_states: torch.Tensor,
    tokens: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    current_memory = memory_states[:, :-1, :]
    target_memory = memory_states[:, 1:, :].detach()
    next_tokens = tokens[:, 1:]
    next_token_embeddings = model.token_embeddings(next_tokens)
    predicted_memory = predictor(current_memory, next_token_embeddings)
    valid = (next_tokens != eos_token_id) & (next_tokens != pad_token_id)
    elementwise = F.smooth_l1_loss(
        predicted_memory,
        target_memory,
        reduction="none",
    )
    weights = valid.unsqueeze(-1).to(elementwise.dtype)
    denominator = weights.expand_as(elementwise).sum().clamp_min(1.0)
    return (elementwise * weights).sum() / denominator


def compute_loss(
    *,
    variant: str,
    model: CausalTransformer | MemoryTapeTransformer,
    output: TransformerOutput | MemoryTapeOutput,
    tokens: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int,
    predictor: MemoryDynamicsPredictor | None = None,
    lambda_memory: float = 1.0,
) -> LossBreakdown:
    if variant == "transformer_ntp":
        if not isinstance(output, TransformerOutput):
            raise TypeError("transformer_ntp requires TransformerOutput")
        nll = next_token_loss(output.logits, tokens, pad_token_id=pad_token_id)
        return LossBreakdown(
            total=nll,
            weighted_ntp=nll,
            final_pass_nll=nll,
            pass_nlls=(nll,),
            memory_prediction=None,
        )

    if not isinstance(model, MemoryTapeTransformer) or not isinstance(
        output, MemoryTapeOutput
    ):
        raise TypeError("memory-tape variants require MemoryTapeTransformer output")
    pass_nlls = tuple(
        next_token_loss(logits, tokens, pad_token_id=pad_token_id)
        for logits in output.logits_per_pass
    )
    ntp_loss = torch.stack(pass_nlls).mean()
    memory_loss = None
    total = ntp_loss
    if variant == "memory_tape_nmp":
        if predictor is None:
            raise ValueError("memory_tape_nmp requires a dynamics predictor")
        memory_loss = temporal_memory_prediction_loss(
            model,
            predictor,
            output.memory_states,
            tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        total = total + lambda_memory * memory_loss
    elif variant != "memory_tape_ntp":
        raise ValueError(f"unknown variant: {variant}")

    return LossBreakdown(
        total=total,
        weighted_ntp=ntp_loss,
        final_pass_nll=pass_nlls[-1],
        pass_nlls=pass_nlls,
        memory_prediction=memory_loss,
    )
