from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, condition_label


FROM_SELECTION = "from_selection"


def format_lambda(value: float) -> str:
    return str(float(value))


@dataclass(frozen=True)
class SelectionConfig:
    metric: str = "final_pass_nll"
    mode: str = "min"
    select_lambda_per_variant: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SelectionConfig":
        return cls(**(payload or {}))

    def validate(self) -> None:
        if not self.metric:
            raise ValueError("selection metric must be non-empty")
        if self.mode not in {"min", "max"}:
            raise ValueError("selection mode must be min or max")


@dataclass(frozen=True)
class PostRunConfig:
    evaluate: bool = True
    plot: bool = True
    summarize: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PostRunConfig":
        return cls(**(payload or {}))


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    architecture: str
    transition: str = "none"
    lambda_transition: float | list[float] | str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConditionSpec":
        condition = cls(
            name=str(payload["name"]),
            architecture=str(payload["architecture"]),
            transition=str(payload.get("transition", "none")),
            lambda_transition=payload.get("lambda_transition"),
            overrides=dict(payload.get("overrides", {})),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if self.lambda_transition == FROM_SELECTION:
            return
        values = (
            self.lambda_transition
            if isinstance(self.lambda_transition, list)
            else [self.lambda_transition]
        )
        for value in values:
            if value is not None:
                float(value)


@dataclass(frozen=True)
class ExperimentPlan:
    name: str
    base_config: Path
    runs_root: Path
    seeds: tuple[int, ...]
    conditions: tuple[ConditionSpec, ...]
    shared_overrides: dict[str, Any] = field(default_factory=dict)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    post_run: PostRunConfig = field(default_factory=PostRunConfig)
    selection_file: Path | None = None
    source_path: Path | None = None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        source_path: str | Path | None = None,
    ) -> "ExperimentPlan":
        base_config = payload.get("base_config")
        if base_config is None:
            raise ValueError("experiment manifest requires base_config")
        plan = cls(
            name=str(payload["name"]),
            base_config=Path(base_config),
            runs_root=Path(payload.get("runs_root", "runs")),
            seeds=tuple(int(seed) for seed in payload["seeds"]),
            conditions=tuple(
                ConditionSpec.from_dict(condition)
                for condition in payload["conditions"]
            ),
            shared_overrides=dict(payload.get("shared_overrides", {})),
            selection=SelectionConfig.from_dict(payload.get("selection")),
            post_run=PostRunConfig.from_dict(payload.get("post_run")),
            selection_file=(
                None
                if payload.get("selection_file") is None
                else Path(payload["selection_file"])
            ),
            source_path=None if source_path is None else Path(source_path),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        if not self.name:
            raise ValueError("experiment name must be non-empty")
        if not self.seeds:
            raise ValueError("experiment manifest requires at least one seed")
        if not self.conditions:
            raise ValueError("experiment manifest requires conditions")
        self.selection.validate()


@dataclass(frozen=True)
class ExpandedRunSpec:
    experiment: str
    condition: str
    run_id: str
    variant: str
    architecture: str
    transition: str
    seed: int
    lambda_transition: float | None
    ntp_pass_weights: list[float] | None
    run_dir: Path
    resolved_config_path: Path

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "condition": self.condition,
            "run_id": self.run_id,
            "variant": self.variant,
            "architecture": self.architecture,
            "transition": self.transition,
            "seed": self.seed,
            "lambda_transition": self.lambda_transition,
            "ntp_pass_weights": self.ntp_pass_weights,
            "run_dir": str(self.run_dir),
            "resolved_config_path": str(self.resolved_config_path),
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ExpandedRunSpec":
        return cls(
            experiment=str(payload["experiment"]),
            condition=str(payload["condition"]),
            run_id=str(payload["run_id"]),
            variant=str(payload["variant"]),
            architecture=str(payload["architecture"]),
            transition=str(payload["transition"]),
            seed=int(payload["seed"]),
            lambda_transition=(
                None
                if payload.get("lambda_transition") is None
                else float(payload["lambda_transition"])
            ),
            ntp_pass_weights=(
                None
                if payload.get("ntp_pass_weights") is None
                else [float(value) for value in payload["ntp_pass_weights"]]
            ),
            run_dir=Path(payload["run_dir"]),
            resolved_config_path=Path(payload["resolved_config_path"]),
        )


@dataclass(frozen=True)
class ExpandedRun:
    spec: ExpandedRunSpec
    config: ExperimentConfig


def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("experiment manifest root must be a mapping")
    return ExperimentPlan.from_dict(payload, source_path=path)


def experiment_dir(plan: ExperimentPlan, runs_root: str | Path | None = None) -> Path:
    root = Path(runs_root) if runs_root is not None else plan.runs_root
    return root / plan.name


def resolve_manifest_path(plan: ExperimentPlan, path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_path = Path(path)
    if cwd_path.exists():
        return cwd_path
    if plan.source_path is not None:
        manifest_relative = plan.source_path.parent / path
        if manifest_relative.exists():
            return manifest_relative
    return cwd_path


def resolve_selection_file(
    plan: ExperimentPlan,
    runs_root: str | Path | None = None,
    selection_file: str | Path | None = None,
) -> Path | None:
    if selection_file is not None:
        return Path(selection_file)
    if plan.selection_file is None:
        return None
    path = plan.selection_file
    if path.is_absolute():
        return path
    return (Path(runs_root) if runs_root is not None else plan.runs_root) / path


def load_base_config_payload(plan: ExperimentPlan) -> dict[str, Any]:
    path = resolve_manifest_path(plan, plan.base_config)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("base config root must be a mapping")
    return payload


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _lambda_values(
    condition: ConditionSpec,
    selected_lambdas: dict[str, float] | None,
) -> list[float | None]:
    raw = condition.lambda_transition
    if raw == FROM_SELECTION:
        if selected_lambdas is None:
            raise ValueError(
                f"{condition.name} requires selected transition weights"
            )
        if condition.name not in selected_lambdas:
            raise ValueError(
                f"selection file is missing condition {condition.name}"
            )
        return [float(selected_lambdas[condition.name])]
    if isinstance(raw, list):
        return [float(value) for value in raw]
    if raw is None:
        return [None]
    return [float(raw)]


def _run_id(
    *,
    condition: str,
    seed: int,
    lambda_transition: float | None,
) -> str:
    parts = [condition]
    if lambda_transition is not None:
        parts.append(f"lambda_{format_lambda(lambda_transition)}")
    parts.append(f"seed_{seed}")
    return "/".join(parts)


def _apply_run_identity(
    payload: dict[str, Any],
    *,
    architecture: str,
    transition: str,
    seed: int,
    lambda_transition: float | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(payload)
    payload["seed"] = int(seed)
    payload.setdefault("model", {})["architecture"] = architecture
    payload.setdefault("objective", {})["transition"] = transition
    if lambda_transition is not None:
        payload.setdefault("objective", {})["lambda_transition"] = float(
            lambda_transition
        )
    if architecture == "transformer":
        payload.setdefault("objective", {}).pop("ntp_pass_weights", None)
    return payload


def _ntp_pass_weights(config: ExperimentConfig) -> list[float] | None:
    if config.objective.ntp_pass_weights is None:
        return None
    return [float(value) for value in config.objective.ntp_pass_weights]


def expand_plan(
    plan: ExperimentPlan,
    *,
    runs_root: str | Path | None = None,
    seeds: set[int] | None = None,
    selected_lambdas: dict[str, float] | None = None,
    steps: int | None = None,
    device: str | None = None,
) -> list[ExpandedRun]:
    base_payload = load_base_config_payload(plan)
    root = experiment_dir(plan, runs_root)
    expanded: list[ExpandedRun] = []
    for condition in plan.conditions:
        condition_payload = deep_merge(
            deep_merge(base_payload, plan.shared_overrides),
            condition.overrides,
        )
        for lambda_transition in _lambda_values(condition, selected_lambdas):
            for seed in plan.seeds:
                if seeds is not None and seed not in seeds:
                    continue
                run_id = _run_id(
                    condition=condition.name,
                    seed=seed,
                    lambda_transition=lambda_transition,
                )
                payload = _apply_run_identity(
                    condition_payload,
                    architecture=condition.architecture,
                    transition=condition.transition,
                    seed=seed,
                    lambda_transition=lambda_transition,
                )
                payload["name"] = f"{plan.name}-{run_id.replace('/', '-')}"
                config = ExperimentConfig.from_dict(payload)
                if steps is not None:
                    config.training.train_steps = int(steps)
                if device is not None:
                    config.training.device = device
                config.validate()
                run_dir = root / run_id
                expanded.append(
                    ExpandedRun(
                        spec=ExpandedRunSpec(
                            experiment=plan.name,
                            condition=condition.name,
                            run_id=run_id,
                            variant=condition_label(config),
                            architecture=condition.architecture,
                            transition=condition.transition,
                            seed=seed,
                            lambda_transition=lambda_transition,
                            ntp_pass_weights=_ntp_pass_weights(config),
                            run_dir=run_dir,
                            resolved_config_path=run_dir / "config.yaml",
                        ),
                        config=config,
                    )
                )
    return expanded


def write_expanded_runs(path: str | Path, runs: list[ExpandedRun]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run.spec.to_json_dict(), sort_keys=True))
            handle.write("\n")


def load_expanded_run_specs(path: str | Path) -> list[ExpandedRunSpec]:
    specs = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                specs.append(ExpandedRunSpec.from_json_dict(json.loads(line)))
    return specs
