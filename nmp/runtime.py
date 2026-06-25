from __future__ import annotations

from contextlib import nullcontext
import random

import torch


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    if device.type == "mps" and dtype == torch.bfloat16:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "float16":
        return torch.cuda.amp.GradScaler()
    return None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

