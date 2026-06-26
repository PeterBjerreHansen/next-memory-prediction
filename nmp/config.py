from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml

VARIANTS = (
    "transformer_ntp",
    "memory_tape_ntp",
    "memory_tape_nmp",
    "memory_tape_hidden_transition",
    "memory_tape_hidden_transition_kl",
)
LEGACY_VARIANT_ALIASES = {
    "memory_tape_nextlat_no_kl": "memory_tape_hidden_transition",
}
ACCEPTED_VARIANTS = (*VARIANTS, *LEGACY_VARIANT_ALIASES)
TRANSITION_VARIANTS = (
    "memory_tape_nmp",
    "memory_tape_hidden_transition",
    "memory_tape_hidden_transition_kl",
)
PRECISIONS = ("float32", "bfloat16", "float16")


def canonicalize_variant(variant: str) -> str:
    if variant in LEGACY_VARIANT_ALIASES:
        return LEGACY_VARIANT_ALIASES[variant]
    return variant


@dataclass(kw_only=True)
class MemoryConfig:
    n_pass: int = 4

    def validate(self) -> None:
        if self.n_pass < 2:
            raise ValueError("memory.n_pass must be at least 2")


@dataclass(kw_only=True)
class ModelConfig:
    variant: str
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def validate(self) -> None:
        self.variant = canonicalize_variant(self.variant)
        if isinstance(self.memory, dict):
            self.memory = MemoryConfig(**self.memory)
        if self.variant not in VARIANTS:
            accepted = ", ".join(ACCEPTED_VARIANTS)
            raise ValueError(f"variant must be one of: {accepted}")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if min(self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("n_layer, n_head, and n_embd must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.variant != "transformer_ntp":
            self.memory.validate()


@dataclass(kw_only=True)
class DataConfig:
    source: str = "generated"
    train_file: str | None = None
    val_file: str | None = None
    generalization_file: str | None = None
    countdown_max_intermediate: int = 10_000
    countdown_min_target: int = 10
    countdown_max_target: int = 100
    countdown_input_numbers: int = 4
    countdown_num_equations: int = 3
    num_pause_tokens: int = 8
    train_samples: int = 500_000
    val_samples: int = 10_000
    generalization_samples: int = 10_000
    split_seed: int = 444

    def validate(self) -> None:
        if self.source not in {"generated", "local"}:
            raise ValueError("data.source must be generated or local")
        if self.source == "local" and (not self.train_file or not self.val_file):
            raise ValueError("local data requires train_file and val_file")
        positive = {
            "countdown_max_intermediate": self.countdown_max_intermediate,
            "countdown_min_target": self.countdown_min_target,
            "countdown_max_target": self.countdown_max_target,
            "countdown_input_numbers": self.countdown_input_numbers,
            "countdown_num_equations": self.countdown_num_equations,
            "num_pause_tokens": self.num_pause_tokens,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.countdown_min_target >= self.countdown_max_target:
            raise ValueError("countdown_min_target must be less than countdown_max_target")
        if self.countdown_num_equations != self.countdown_input_numbers - 1:
            raise ValueError(
                "countdown_num_equations must equal countdown_input_numbers - 1"
            )
        if self.generalization_samples < 0:
            raise ValueError("generalization_samples must be non-negative")


@dataclass(kw_only=True)
class TransitionObjectiveConfig:
    horizon: int = 1
    lambda_transition: float = 1.0
    lambda_kl: float = 1.0
    lambda_ce: float = 0.0
    target: str | None = None
    projection_factor: float = 1.3

    def validate(self) -> None:
        if self.horizon != 1:
            raise ValueError("transition.horizon must be 1 in this implementation")
        if self.lambda_transition < 0:
            raise ValueError("lambda_transition must be non-negative")
        if self.lambda_kl < 0:
            raise ValueError("transition.lambda_kl must be non-negative")
        if self.lambda_ce < 0:
            raise ValueError("transition.lambda_ce must be non-negative")
        if self.target is not None and self.target not in {"hidden", "memory"}:
            raise ValueError("transition.target must be hidden, memory, or null")
        if self.projection_factor <= 0:
            raise ValueError("transition.projection_factor must be positive")


@dataclass(kw_only=True)
class ObjectiveConfig:
    ntp_pass_weights: list[float] | None = None
    transition: TransitionObjectiveConfig = field(
        default_factory=TransitionObjectiveConfig
    )

    def validate(self) -> None:
        if isinstance(self.transition, dict):
            self.transition = TransitionObjectiveConfig(**self.transition)
        self.transition.validate()
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
    prompt_tokens: int = 32
    generation_tokens: int = 32
    diagnostic_batches: int = 8
    probe_steps: int = 1000
    probe_batch_size: int = 64
    probe_offsets: list[int] = field(default_factory=lambda: list(range(1, 21)))
    checkpoint_metric: str = "val_accuracy"
    checkpoint_mode: str = "max"

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
        default_target = default_transition_target_for_variant(self.model.variant)
        configured_target = self.objective.transition.target
        if configured_target is None:
            self.objective.transition.target = default_target
        elif default_target is None:
            raise ValueError("transition.target is only valid for transition variants")
        elif configured_target != default_target:
            raise ValueError(
                f"transition.target must be {default_target!r} for "
                f"{self.model.variant}"
            )
        if self.objective.ntp_pass_weights is not None:
            if self.model.variant == "transformer_ntp":
                raise ValueError(
                    "ntp_pass_weights are only valid for memory-tape variants"
                )
            if len(self.objective.ntp_pass_weights) != self.model.memory.n_pass:
                raise ValueError(
                    "ntp_pass_weights must have one entry per MemoryTape pass"
                )
        self.training.validate()
        self.evaluation.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        model_payload = normalize_model_payload(payload["model"])
        objective_payload = normalize_objective_payload(
            payload.get("objective", {})
        )
        config = cls(
            name=str(payload["name"]),
            seed=int(payload.get("seed", 0)),
            model=ModelConfig(**model_payload),
            data=DataConfig(**payload.get("data", {})),
            objective=ObjectiveConfig(**objective_payload),
            training=TrainingConfig(**payload["training"]),
            evaluation=EvaluationConfig(**payload.get("evaluation", {})),
        )
        config.validate()
        return config


def normalize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    memory_payload = dict(normalized.get("memory", {}))
    legacy_n_pass = normalized.pop("n_pass", None)
    if legacy_n_pass is not None and "n_pass" not in memory_payload:
        memory_payload["n_pass"] = legacy_n_pass
    normalized["memory"] = memory_payload
    return normalized


def normalize_objective_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    transition_payload = dict(normalized.get("transition", {}))
    nested_legacy_horizon = transition_payload.pop("transition_horizon", None)
    if nested_legacy_horizon is not None and "horizon" not in transition_payload:
        transition_payload["horizon"] = nested_legacy_horizon
    legacy_horizon = normalized.pop("transition_horizon", None)
    if legacy_horizon is not None and "horizon" not in transition_payload:
        transition_payload["horizon"] = legacy_horizon
    legacy_lambda_transition = normalized.pop("lambda_transition", None)
    if (
        legacy_lambda_transition is not None
        and "lambda_transition" not in transition_payload
    ):
        transition_payload["lambda_transition"] = legacy_lambda_transition
    legacy_projection_factor = normalized.pop(
        "dynamics_projection_factor",
        None,
    )
    if (
        legacy_projection_factor is not None
        and "projection_factor" not in transition_payload
    ):
        transition_payload["projection_factor"] = legacy_projection_factor
    legacy_lambda_kl = normalized.pop("lambda_kl", None)
    if legacy_lambda_kl is not None and "lambda_kl" not in transition_payload:
        transition_payload["lambda_kl"] = legacy_lambda_kl
    legacy_lambda_ce = normalized.pop("lambda_ce", None)
    if legacy_lambda_ce is not None and "lambda_ce" not in transition_payload:
        transition_payload["lambda_ce"] = legacy_lambda_ce
    legacy_target = normalized.pop("transition_target", None)
    if legacy_target is not None and "target" not in transition_payload:
        transition_payload["target"] = legacy_target
    normalized["transition"] = transition_payload
    return normalized


def default_transition_target_for_variant(variant: str) -> str | None:
    variant = canonicalize_variant(variant)
    if variant == "memory_tape_nmp":
        return "memory"
    if variant in {
        "memory_tape_hidden_transition",
        "memory_tape_hidden_transition_kl",
    }:
        return "hidden"
    return None


def transition_target_for_variant(
    variant: str,
    transition: TransitionObjectiveConfig | None = None,
) -> str | None:
    configured_target = getattr(transition, "target", None)
    if configured_target is not None:
        return configured_target
    return default_transition_target_for_variant(variant)


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
