from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from nmp.experiment_plan import FROM_SELECTION
from nmp.experiments import load_selected_lambdas


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Freeze a reference experiment template with selected lambdas."
    )
    parser.add_argument("--development-summary", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("template root must be a mapping")
    return payload


def freeze_payload(payload: dict[str, Any], selected_lambdas: dict[str, float]):
    frozen = dict(payload)
    frozen.pop("selection_file", None)
    frozen["frozen_from"] = "development selected_lambdas.json"
    for condition in frozen.get("conditions", []):
        if condition.get("lambda_transition") == FROM_SELECTION:
            variant = condition["variant"]
            condition["lambda_transition"] = float(selected_lambdas[variant])
    return frozen


def main(argv=None):
    args = parse_args(argv)
    selection_path = args.development_summary / "selected_lambdas.json"
    selected_lambdas = load_selected_lambdas(selection_path)
    frozen = freeze_payload(_load_yaml(args.template), selected_lambdas)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(frozen, handle, sort_keys=False)
    print(args.output)


if __name__ == "__main__":
    main()
