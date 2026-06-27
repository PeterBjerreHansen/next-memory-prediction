from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml

ARCHITECTURES = ("transformer", "memory_tape")
TRANSITIONS = ("none", "memory", "hidden", "hidden_kl")
TRANSITION_OBJECTIVES = ("memory", "hidden", "hidden_kl")
PRECISIONS = ("float32", "bfloat16", "float16")


@dataclass(kw_only=True)
class MemoryConfig:
    n_pass: int = 4

    def validate(self) -> None:
        if self.n_pass < 2:
            raise ValueError("memory.n_pass must be at least 2")


@dataclass(kw_only=True)
class ModelConfig:
    architecture: str
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def validate(self) -> None:
        if isinstance(self.memory, dict):
            self.memory = MemoryConfig(**self.memory)
        if self.architecture not in ARCHITECTURES:
            accepted = ", ".join(ARCHITECTURES)
            raise ValueError(f"architecture must be one of: {accepted}")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if min(self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.architecture == "memory_tape":
            self.memory.validate()


@dataclass(kw_only=True)
class DataConfig:
    train_file: str
    val_file: str
    generalization_file: str | None = None
    countdown_max_intermediate: int = 10_000
    countdown_input_numbers: int = 4
    countdown_num_equations: int = 3
    num_pause_tokens: int = 8

    def validate(self) -> None:
        if not self.train_file or not self.val_file:
            raise ValueError("data.train_file and data.val_file are required")
        positive = {
            "countdown_max_intermediate": self.countdown_max_intermediate,
            "countdown_input_numbers": self.countdown_input_numbers,
            "countdown_num_equations": self.countdown_num_equations,
            "num_pause_tokens": self.num_pause_tokens,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.countdown_num_equations != self.countdown_input_numbers - 1:
            raise ValueError(
                "countdown_num_equations must equal countdown_input_numbers - 1"
            )


@dataclass(kw_only=True)
class ObjectiveConfig:
    transition: str = "none"
    ntp_pass_weights: list[float] | None = None
    lambda_transition: float = 1.0
    lambda_kl: float = 1.0
    projection_factor: float = 1.3

    def validate(self) -> None:
        if self.transition not in TRANSITIONS:
            accepted = ", ".join(TRANSITIONS)
            raise ValueError(f"objective.transition must be one of: {accepted}")
        if self.lambda_transition < 0:
            raise ValueError("lambda_transition must be non-negative")
        if self.lambda_kl < 0:
            raise ValueError("lambda_kl must be non-negative")
        if self.projection_factor <= 0:
            raise ValueError("projection_factor must be positive")
        if self.ntp_pass_weights is not None:
            self.ntp_pass_weights = [
                float(weight) for weight in self.ntp_pass_weights
            ]
            if any(
                not math.isfinite(weight) or weight < 0
                for weight in self.ntp_pass_weights
            ):
                raise ValueError(
                    "ntp_pass_weights must contain finite non-negative values"
                )
            if sum(self.ntp_pass_weights) <= 0:
                raise ValueError("ntp_pass_weights must have positive sum")


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
    diagnostic_batches: int = 8
    accuracy_batches: int | None = None
    training_accuracy_interval: int | None = None
    training_accuracy_batches: int = 4
    checkpoint_metric: str = "final_pass_nll"
    checkpoint_mode: str = "min"

    def validate(self) -> None:
        if min(self.generation_prompts, self.diagnostic_batches) < 1:
            raise ValueError("evaluation counts must be positive")
        if self.accuracy_batches is not None and self.accuracy_batches < 1:
            raise ValueError("accuracy_batches must be positive or null")
        if (
            self.training_accuracy_interval is not None
            and self.training_accuracy_interval < 1
        ):
            raise ValueError("training_accuracy_interval must be positive or null")
        if self.training_accuracy_batches < 1:
            raise ValueError("training_accuracy_batches must be positive")
        if not self.checkpoint_metric:
            raise ValueError("checkpoint_metric must be non-empty")
        if self.checkpoint_mode not in {"min", "max"}:
            raise ValueError("checkpoint_mode must be min or max")


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
        if self.model.architecture == "transformer":
            if self.objective.transition != "none":
                raise ValueError("transformer architecture only supports transition=none")
            if self.objective.ntp_pass_weights is not None:
                raise ValueError(
                    "ntp_pass_weights are only valid for memory_tape architecture"
                )
        if (
            self.model.architecture == "memory_tape"
            and self.objective.ntp_pass_weights is not None
            and len(self.objective.ntp_pass_weights) != self.model.memory.n_pass
        ):
            raise ValueError(
                "ntp_pass_weights must have one entry per MemoryTape pass"
            )
        if (
            self.model.architecture != "memory_tape"
            and self.objective.transition in TRANSITION_OBJECTIVES
        ):
            raise ValueError("transition objectives require memory_tape architecture")
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
            data=DataConfig(**payload["data"]),
            objective=ObjectiveConfig(**payload.get("objective", {})),
            training=TrainingConfig(**payload["training"]),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
        )
        config.validate()
        return config


def transition_target(transition: str) -> str | None:
    if transition == "memory":
        return "memory"
    if transition in {"hidden", "hidden_kl"}:
        return "hidden"
    return None


def condition_label(config: ExperimentConfig) -> str:
    if config.model.architecture == "transformer":
        return "transformer_ntp"
    transition = config.objective.transition
    if transition == "none":
        return "memory_tape_ntp"
    if transition == "memory":
        return "memory_tape_nmp"
    if transition == "hidden":
        return "memory_tape_hidden_transition"
    if transition == "hidden_kl":
        return "memory_tape_hidden_transition_kl"
    raise ValueError(f"unknown transition: {transition}")


def active_objective_metadata(
    objective: ObjectiveConfig,
) -> dict[str, float | str | None]:
    if objective.transition not in TRANSITION_OBJECTIVES:
        return {
            "transition_target": None,
            "lambda_transition": None,
            "lambda_kl": None,
        }
    return {
        "transition_target": transition_target(objective.transition),
        "lambda_transition": objective.lambda_transition,
        "lambda_kl": objective.lambda_kl
        if objective.transition == "hidden_kl"
        else None,
    }


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
