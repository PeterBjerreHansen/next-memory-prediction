from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .config import (
    TRANSITION_VARIANTS,
    canonicalize_variant,
    transition_target_for_variant,
)
from .models import (
    CausalTransformer,
    LatentTransitionPredictor,
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
    ntp_pass_weights: tuple[float, ...]
    transition_prediction: torch.Tensor | None
    transition_target: str | None

    def detached_metrics(self) -> dict[str, object]:
        return {
            "loss": float(self.total.detach().cpu()),
            "weighted_ntp_loss": float(self.weighted_ntp.detach().cpu()),
            "final_pass_nll": float(self.final_pass_nll.detach().cpu()),
            "pass_nlls": [float(item.detach().cpu()) for item in self.pass_nlls],
            "ntp_pass_weights": list(self.ntp_pass_weights),
            "transition_prediction_loss": (
                None
                if self.transition_prediction is None
                else float(self.transition_prediction.detach().cpu())
            ),
            "transition_target": self.transition_target,
        }


def normalize_pass_weights(
    pass_count: int,
    weights: list[float] | tuple[float, ...] | None,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if weights is None:
        return torch.full(
            (pass_count,),
            1.0 / pass_count,
            dtype=torch.float32,
            device=device,
        )
    if len(weights) != pass_count:
        raise ValueError("ntp_pass_weights must match the number of passes")
    tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
    if torch.any(~torch.isfinite(tensor)) or torch.any(tensor < 0):
        raise ValueError("ntp_pass_weights must be finite and non-negative")
    total = tensor.sum()
    if float(total.detach().cpu()) <= 0.0:
        raise ValueError("ntp_pass_weights must have positive sum")
    return tensor / total


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


def temporal_transition_prediction_loss(
    model: MemoryTapeTransformer,
    predictor: LatentTransitionPredictor,
    latent_states: torch.Tensor,
    tokens: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    current_latent = latent_states[:, :-1, :]
    target_latent = latent_states[:, 1:, :].detach()
    next_tokens = tokens[:, 1:]
    next_token_embeddings = model.token_embeddings(next_tokens)
    predicted_latent = predictor(current_latent, next_token_embeddings)
    valid = (next_tokens != eos_token_id) & (next_tokens != pad_token_id)
    elementwise = F.smooth_l1_loss(
        predicted_latent,
        target_latent,
        reduction="none",
    )
    weights = valid.unsqueeze(-1).to(elementwise.dtype)
    denominator = weights.expand_as(elementwise).sum().clamp_min(1.0)
    return (elementwise * weights).sum() / denominator


def temporal_memory_prediction_loss(
    model: MemoryTapeTransformer,
    predictor: LatentTransitionPredictor,
    memory_states: torch.Tensor,
    tokens: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Compatibility wrapper for the original memory-only objective."""
    return temporal_transition_prediction_loss(
        model,
        predictor,
        memory_states,
        tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def compute_loss(
    *,
    variant: str,
    model: CausalTransformer | MemoryTapeTransformer,
    output: TransformerOutput | MemoryTapeOutput,
    tokens: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int,
    predictor: LatentTransitionPredictor | None = None,
    lambda_transition: float = 1.0,
    ntp_pass_weights: list[float] | tuple[float, ...] | None = None,
) -> LossBreakdown:
    variant = canonicalize_variant(variant)
    if variant == "transformer_ntp":
        if ntp_pass_weights is not None:
            raise ValueError("transformer_ntp does not use ntp_pass_weights")
        if not isinstance(output, TransformerOutput):
            raise TypeError("transformer_ntp requires TransformerOutput")
        nll = next_token_loss(output.logits, tokens, pad_token_id=pad_token_id)
        return LossBreakdown(
            total=nll,
            weighted_ntp=nll,
            final_pass_nll=nll,
            pass_nlls=(nll,),
            ntp_pass_weights=(1.0,),
            transition_prediction=None,
            transition_target=None,
        )

    if not isinstance(model, MemoryTapeTransformer) or not isinstance(
        output, MemoryTapeOutput
    ):
        raise TypeError("memory-tape variants require MemoryTapeTransformer output")
    pass_nlls = tuple(
        next_token_loss(logits, tokens, pad_token_id=pad_token_id)
        for logits in output.logits_per_pass
    )
    normalized_weights = normalize_pass_weights(
        len(pass_nlls),
        ntp_pass_weights,
        device=pass_nlls[0].device,
    )
    ntp_loss = (
        torch.stack(pass_nlls) * normalized_weights.to(pass_nlls[0].dtype)
    ).sum()
    transition_loss = None
    transition_target = transition_target_for_variant(variant)
    total = ntp_loss
    if variant in TRANSITION_VARIANTS:
        if predictor is None:
            raise ValueError(f"{variant} requires a transition predictor")
        latent_states = (
            output.memory_states
            if transition_target == "memory"
            else output.hidden_states
        )
        transition_loss = temporal_transition_prediction_loss(
            model,
            predictor,
            latent_states,
            tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        total = total + lambda_transition * transition_loss
    elif variant != "memory_tape_ntp":
        raise ValueError(f"unknown variant: {variant}")

    return LossBreakdown(
        total=total,
        weighted_ntp=ntp_loss,
        final_pass_nll=pass_nlls[-1],
        pass_nlls=pass_nlls,
        ntp_pass_weights=tuple(
            float(weight.detach().cpu()) for weight in normalized_weights
        ),
        transition_prediction=transition_loss,
        transition_target=transition_target,
    )
