from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def effective_rank(states: torch.Tensor) -> float:
    if states.numel() == 0:
        return 0.0
    matrix = states.float()
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(matrix)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel() == 0:
        return 0.0
    probabilities = singular_values / singular_values.sum()
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.exp().cpu())


def masked_states(states: torch.Tensor, tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    return states[tokens != pad_id]


def mean_adjacent_cosine(
    states: torch.Tensor,
    tokens: torch.Tensor,
    *,
    pad_id: int,
) -> float:
    valid = (tokens[:, :-1] != pad_id) & (tokens[:, 1:] != pad_id)
    if not bool(valid.any()):
        return 0.0
    similarities = F.cosine_similarity(states[:, :-1], states[:, 1:], dim=-1)
    return float(similarities[valid].mean().cpu())


def mean_cross_pass_cosine(
    earlier: torch.Tensor,
    later: torch.Tensor,
    tokens: torch.Tensor,
    *,
    pad_id: int,
) -> float:
    valid = tokens != pad_id
    if not bool(valid.any()):
        return 0.0
    similarities = F.cosine_similarity(earlier, later, dim=-1)
    return float(similarities[valid].mean().cpu())


def safe_perplexity(nll: float) -> float:
    return math.exp(min(float(nll), 20.0))

