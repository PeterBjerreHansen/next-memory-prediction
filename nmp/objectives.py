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
    transition_kl: torch.Tensor | None
    transition_ce: torch.Tensor | None
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
            "transition_kl_loss": (
                None
                if self.transition_kl is None
                else float(self.transition_kl.detach().cpu())
            ),
            "transition_ce_loss": (
                None
                if self.transition_ce is None
                else float(self.transition_ce.detach().cpu())
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
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    predictions = logits[:, :-1, :]
    targets = tokens[:, 1:]
    if target_mask is not None:
        if target_mask.shape != targets.shape:
            raise ValueError("target_mask must match next-token targets")
        valid = target_mask & (targets != pad_token_id)
        if not bool(valid.any()):
            return predictions.sum() * 0.0
        targets = targets.masked_fill(~valid, -100)
        ignore_index = -100
    else:
        ignore_index = pad_token_id
    return F.cross_entropy(
        predictions.reshape(-1, predictions.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def temporal_transition_prediction(
    model: MemoryTapeTransformer,
    predictor: LatentTransitionPredictor,
    latent_states: torch.Tensor,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    current_latent = latent_states[:, :-1, :]
    target_latent = latent_states[:, 1:, :].detach()
    next_tokens = tokens[:, 1:]
    next_token_embeddings = model.token_embeddings(next_tokens)
    predicted_latent = predictor(current_latent, next_token_embeddings)
    return predicted_latent, target_latent, next_tokens


def temporal_transition_prediction_loss_from_prediction(
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
    next_tokens: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    valid = (next_tokens != eos_token_id) & (next_tokens != pad_token_id)
    elementwise = F.smooth_l1_loss(
        predicted_latent,
        target_latent,
        reduction="none",
    )
    weights = valid.unsqueeze(-1).to(elementwise.dtype)
    denominator = weights.expand_as(elementwise).sum().clamp_min(1.0)
    return (elementwise * weights).sum() / denominator


def temporal_transition_prediction_loss(
    model: MemoryTapeTransformer,
    predictor: LatentTransitionPredictor,
    latent_states: torch.Tensor,
    tokens: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    predicted_latent, target_latent, next_tokens = temporal_transition_prediction(
        model,
        predictor,
        latent_states,
        tokens,
    )
    return temporal_transition_prediction_loss_from_prediction(
        predicted_latent,
        target_latent,
        next_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


def temporal_transition_kl_mask(
    tokens: torch.Tensor,
    target_mask: torch.Tensor | None,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    if tokens.size(1) < 3:
        return torch.zeros(
            (tokens.size(0), 0),
            dtype=torch.bool,
            device=tokens.device,
        )
    current_predicted_token = tokens[:, 1:-1]
    next_next_token = tokens[:, 2:]
    valid = (
        (current_predicted_token != eos_token_id)
        & (current_predicted_token != pad_token_id)
        & (next_next_token != pad_token_id)
    )
    if target_mask is not None:
        valid = valid & target_mask[:, 1:]
    return valid


def temporal_transition_self_distillation_kl_loss(
    model: MemoryTapeTransformer,
    predicted_latent: torch.Tensor,
    teacher_logits: torch.Tensor,
    tokens: torch.Tensor,
    target_mask: torch.Tensor | None,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    if predicted_latent.size(1) < 2:
        return predicted_latent.sum() * 0.0
    student_inputs = predicted_latent[:, :-1, :]
    teacher = teacher_logits[:, 1:-1, :].detach()
    student = F.linear(student_inputs, model.lm_head.weight.detach())
    valid = temporal_transition_kl_mask(
        tokens,
        target_mask,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    if not bool(valid.any()):
        return student.sum() * 0.0
    log_teacher = F.log_softmax(teacher, dim=-1)
    log_student = F.log_softmax(student, dim=-1)
    kl_per_vocab = F.kl_div(
        log_student,
        log_teacher,
        log_target=True,
        reduction="none",
    )
    kl_per_token = kl_per_vocab.sum(dim=-1)
    weights = valid.to(kl_per_token.dtype)
    return (kl_per_token * weights).sum() / weights.sum().clamp_min(1.0)


def temporal_transition_self_distillation_ce_loss(
    model: MemoryTapeTransformer,
    predicted_latent: torch.Tensor,
    tokens: torch.Tensor,
    target_mask: torch.Tensor | None,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    if predicted_latent.size(1) < 2:
        return predicted_latent.sum() * 0.0
    student_inputs = predicted_latent[:, :-1, :]
    student = F.linear(student_inputs, model.lm_head.weight.detach())
    targets = tokens[:, 2:]
    valid = temporal_transition_kl_mask(
        tokens,
        target_mask,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    if not bool(valid.any()):
        return student.sum() * 0.0
    targets = targets.masked_fill(~valid, -100)
    return F.cross_entropy(
        student.reshape(-1, student.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )


def compute_loss(
    *,
    variant: str,
    model: CausalTransformer | MemoryTapeTransformer,
    output: TransformerOutput | MemoryTapeOutput,
    tokens: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    pad_token_id: int,
    eos_token_id: int,
    predictor: LatentTransitionPredictor | None = None,
    lambda_transition: float = 1.0,
    lambda_kl: float = 1.0,
    lambda_ce: float = 0.0,
    transition_horizon: int = 1,
    transition_target: str | None = None,
    ntp_pass_weights: list[float] | tuple[float, ...] | None = None,
) -> LossBreakdown:
    variant = canonicalize_variant(variant)
    if transition_horizon != 1:
        raise ValueError("transition_horizon must be 1 in this implementation")
    if variant == "transformer_ntp":
        if ntp_pass_weights is not None:
            raise ValueError("transformer_ntp does not use ntp_pass_weights")
        if not isinstance(output, TransformerOutput):
            raise TypeError("transformer_ntp requires TransformerOutput")
        nll = next_token_loss(
            output.logits,
            tokens,
            pad_token_id=pad_token_id,
            target_mask=target_mask,
        )
        return LossBreakdown(
            total=nll,
            weighted_ntp=nll,
            final_pass_nll=nll,
            pass_nlls=(nll,),
            ntp_pass_weights=(1.0,),
            transition_prediction=None,
            transition_kl=None,
            transition_ce=None,
            transition_target=None,
        )

    if not isinstance(model, MemoryTapeTransformer) or not isinstance(
        output, MemoryTapeOutput
    ):
        raise TypeError("memory-tape variants require MemoryTapeTransformer output")
    pass_nlls = tuple(
        next_token_loss(logits, tokens, pad_token_id=pad_token_id)
        if target_mask is None
        else next_token_loss(
            logits,
            tokens,
            pad_token_id=pad_token_id,
            target_mask=target_mask,
        )
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
    transition_kl_loss = None
    transition_ce_loss = None
    transition_target = transition_target or transition_target_for_variant(variant)
    total = ntp_loss
    if variant in TRANSITION_VARIANTS:
        if predictor is None:
            raise ValueError(f"{variant} requires a transition predictor")
        if transition_target not in {"memory", "hidden"}:
            raise ValueError(f"{variant} requires memory or hidden transition target")
        latent_states = (
            output.memory_states
            if transition_target == "memory"
            else output.hidden_states
        )
        predicted_latent, target_latent, next_tokens = temporal_transition_prediction(
            model,
            predictor,
            latent_states,
            tokens,
        )
        transition_loss = temporal_transition_prediction_loss_from_prediction(
            predicted_latent,
            target_latent,
            next_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        total = total + lambda_transition * transition_loss
        if variant == "memory_tape_hidden_transition_kl":
            transition_kl_loss = temporal_transition_self_distillation_kl_loss(
                model,
                predicted_latent,
                output.logits,
                tokens,
                target_mask,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            total = total + lambda_kl * transition_kl_loss
            transition_ce_loss = temporal_transition_self_distillation_ce_loss(
                model,
                predicted_latent,
                tokens,
                target_mask,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
            total = total + lambda_ce * transition_ce_loss
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
        transition_kl=transition_kl_loss,
        transition_ce=transition_ce_loss,
        transition_target=transition_target,
    )
