from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .provenance import DATASET_NAME, DATASET_REVISION


VARIANTS = ("transformer_ntp", "memory_tape_ntp", "memory_tape_nmp")
PRECISIONS = ("float32", "bfloat16", "float16")


@dataclass(kw_only=True)
class ModelConfig:
    variant: str
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    n_pass: int = 4
    memory_tape_gate: str = "scalar"

    def validate(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of: {', '.join(VARIANTS)}")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if min(self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.variant != "transformer_ntp" and self.n_pass < 2:
            raise ValueError("memory-tape variants require n_pass >= 2")
        if self.memory_tape_gate not in {"none", "tanh", "scalar"}:
            raise ValueError("memory_tape_gate must be none, tanh, or scalar")


@dataclass(kw_only=True)
class DataConfig:
    source: str = "huggingface"
    dataset_name: str = DATASET_NAME
    dataset_revision: str = DATASET_REVISION
    text_field: str = "text"
    cache_dir: str | None = None
    train_file: str | None = None
    val_file: str | None = None
    validation_size: int = 10_000
    split_seed: int = 1234

    def validate(self) -> None:
        if self.source not in {"huggingface", "local"}:
            raise ValueError("data.source must be huggingface or local")
        if self.source == "local" and (not self.train_file or not self.val_file):
            raise ValueError("local data requires train_file and val_file")
        if self.validation_size < 1:
            raise ValueError("validation_size must be positive")


@dataclass(kw_only=True)
class ObjectiveConfig:
    lambda_memory: float = 1.0
    memory_horizon: int = 1
    dynamics_projection_factor: float = 1.3

    def validate(self) -> None:
        if self.lambda_memory < 0:
            raise ValueError("lambda_memory must be non-negative")
        if self.memory_horizon != 1:
            raise ValueError("only memory_horizon=1 is implemented")
        if self.dynamics_projection_factor <= 0:
            raise ValueError("dynamics_projection_factor must be positive")


@dataclass(kw_only=True)
class TrainingConfig:
    train_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 1000
    eval_batches: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 1000
    device: str = "auto"
    precision: str = "float32"
    compile: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        positive = {
            "train_steps": self.train_steps,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "eval_interval": self.eval_interval,
            "eval_batches": self.eval_batches,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of: {', '.join(PRECISIONS)}")


@dataclass(kw_only=True)
class EvaluationConfig:
    generation_prompts: int = 4
    prompt_tokens: int = 32
    generation_tokens: int = 32
    diagnostic_batches: int = 8
    probe_steps: int = 1000
    probe_batch_size: int = 64
    probe_offsets: list[int] = field(default_factory=lambda: list(range(1, 21)))

    def validate(self) -> None:
        if min(
            self.generation_prompts,
            self.prompt_tokens,
            self.generation_tokens,
            self.diagnostic_batches,
            self.probe_steps,
            self.probe_batch_size,
        ) < 1:
            raise ValueError("evaluation counts must be positive")
        if not self.probe_offsets or min(self.probe_offsets) < 1:
            raise ValueError("probe_offsets must contain positive integers")


@dataclass(kw_only=True)
class ExperimentConfig:
    name: str
    seed: int
    model: ModelConfig
    data: DataConfig
    objective: ObjectiveConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        self.objective.validate()
        self.training.validate()
        self.evaluation.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        config = cls(
            name=str(payload["name"]),
            seed=int(payload.get("seed", 0)),
            model=ModelConfig(**payload["model"]),
            data=DataConfig(**payload.get("data", {})),
            objective=ObjectiveConfig(**payload.get("objective", {})),
            training=TrainingConfig(**payload["training"]),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
        )
        config.validate()
        return config


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    return ExperimentConfig.from_dict(payload)


def save_config(path: str | Path, config: ExperimentConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)

