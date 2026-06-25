from __future__ import annotations

from .config import ExperimentConfig
from .models import (
    CausalTransformer,
    MemoryDynamicsPredictor,
    MemoryTapeConfig,
    MemoryTapeTransformer,
    TransformerConfig,
)


def build_model(
    config: ExperimentConfig,
    *,
    vocab_size: int,
) -> tuple[
    CausalTransformer | MemoryTapeTransformer,
    MemoryDynamicsPredictor | None,
]:
    model_config = config.model
    if model_config.variant == "transformer_ntp":
        model = CausalTransformer(
            TransformerConfig(
                block_size=model_config.block_size,
                vocab_size=vocab_size,
                n_layer=model_config.n_layer,
                n_head=model_config.n_head,
                n_embd=model_config.n_embd,
            )
        )
        return model, None

    model = MemoryTapeTransformer(
        MemoryTapeConfig(
            block_size=model_config.block_size,
            vocab_size=vocab_size,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            n_embd=model_config.n_embd,
            n_pass=model_config.n_pass,
            memory_tape_gate=model_config.memory_tape_gate,
        )
    )
    predictor = (
        MemoryDynamicsPredictor(
            model_config.n_embd,
            projection_factor=config.objective.dynamics_projection_factor,
        )
        if model_config.variant == "memory_tape_nmp"
        else None
    )
    return model, predictor


def trainable_parameters(model, predictor=None):
    parameters = list(model.parameters())
    if predictor is not None:
        parameters.extend(predictor.parameters())
    return parameters


def count_parameters(model, predictor=None) -> dict[str, int]:
    model_count = sum(parameter.numel() for parameter in model.parameters())
    predictor_count = (
        0
        if predictor is None
        else sum(parameter.numel() for parameter in predictor.parameters())
    )
    return {
        "model": model_count,
        "training_only": predictor_count,
        "total_training": model_count + predictor_count,
    }
