from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig


def _torch_load(path: str | Path, *, map_location="cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path: str | Path,
    *,
    config: ExperimentConfig,
    model,
    predictor,
    optimizer,
    scaler,
    step: int,
    best_final_pass_nll: float,
    sampler_state: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "step": int(step),
        "best_final_pass_nll": float(best_final_pass_nll),
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": (
            None if predictor is None else predictor.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "sampler_state": sampler_state,
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, *, map_location="cpu") -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "latest.pt"
    return _torch_load(path, map_location=map_location)


def restore_checkpoint(
    checkpoint: dict[str, Any],
    *,
    model,
    predictor,
    optimizer=None,
    scaler=None,
    sampler=None,
    restore_rng: bool = True,
) -> None:
    model.load_state_dict(checkpoint["model_state_dict"])
    saved_predictor = checkpoint.get("predictor_state_dict")
    if predictor is not None:
        if saved_predictor is None:
            raise ValueError("checkpoint has no dynamics predictor state")
        predictor.load_state_dict(saved_predictor)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if sampler is not None:
        sampler.load_state_dict(checkpoint["sampler_state"])
    if not restore_rng:
        return
    random.setstate(checkpoint["python_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_state = checkpoint.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
