from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from nmp.checkpoint import config_from_checkpoint, load_checkpoint
from nmp.config import save_config
from nmp.experiment_plan import (
    FROM_SELECTION,
    expand_plan,
    experiment_dir,
    load_experiment_plan,
    resolve_selection_file,
    write_expanded_runs,
)
from nmp.experiments import load_selected_lambdas


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an experiment manifest."
    )
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional subset of manifest seeds for local runs.",
    )
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _needs_selection(plan) -> bool:
    return any(
        condition.lambda_transition == FROM_SELECTION
        for condition in plan.conditions
    )


def _load_selected_lambdas(plan, args) -> dict[str, float] | None:
    selection_path = resolve_selection_file(
        plan,
        runs_root=args.runs_root,
        selection_file=args.selection_file,
    )
    if selection_path is None:
        if _needs_selection(plan):
            raise ValueError(
                "manifest uses from_selection but no selection_file is set"
            )
        return None
    return load_selected_lambdas(selection_path)


def train_command(run, plan, *, device: str | None, steps: int | None) -> list[str]:
    latest = run.spec.run_dir / "latest.pt"
    if latest.exists():
        command = [
            sys.executable,
            "-m",
            "nmp.cli.train",
            "--resume-from",
            str(latest),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "nmp.cli.train",
            "--config",
            str(run.spec.resolved_config_path),
            "--run-dir",
            str(run.spec.run_dir),
        ]
    if device is not None:
        command.extend(["--device", device])
    if steps is not None:
        command.extend(["--steps", str(steps)])
    if not plan.post_run.evaluate:
        command.append("--skip-evaluate")
    if not plan.post_run.plot:
        command.append("--skip-plot")
    return command


def run(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def assert_resume_compatible(run) -> None:
    latest = run.spec.run_dir / "latest.pt"
    if not latest.exists():
        return
    checkpoint_config = config_from_checkpoint(load_checkpoint(latest))
    checkpoint_config.training.train_steps = run.config.training.train_steps
    checkpoint_config.training.device = run.config.training.device
    checkpoint_config.validate()
    if checkpoint_config.to_dict() != run.config.to_dict():
        raise ValueError(
            "existing checkpoint config does not match manifest for "
            f"{run.spec.run_dir}"
        )


def write_manifest_artifacts(
    *,
    manifest_path: Path,
    experiment_root: Path,
    expanded_runs,
) -> Path:
    experiment_root.mkdir(parents=True, exist_ok=True)
    copied_manifest = experiment_root / "experiment.yaml"
    shutil.copyfile(manifest_path, copied_manifest)
    expanded_path = experiment_root / "expanded_runs.jsonl"
    write_expanded_runs(expanded_path, expanded_runs)
    for expanded in expanded_runs:
        assert_resume_compatible(expanded)
        save_config(expanded.spec.resolved_config_path, expanded.config)
    return expanded_path


def main(argv=None):
    args = parse_args(argv)
    plan = load_experiment_plan(args.experiment)
    selected_lambdas = _load_selected_lambdas(plan, args)
    expanded = expand_plan(
        plan,
        runs_root=args.runs_root,
        seeds=None if args.seeds is None else set(args.seeds),
        selected_lambdas=selected_lambdas,
        steps=args.steps,
        device=args.device,
    )
    root = experiment_dir(plan, args.runs_root)
    expanded_path = write_manifest_artifacts(
        manifest_path=args.experiment,
        experiment_root=root,
        expanded_runs=expanded,
    )
    for run_spec in expanded:
        run(
            train_command(
                run_spec,
                plan,
                device=args.device,
                steps=args.steps,
            ),
            dry_run=args.dry_run,
        )
    if (
        plan.post_run.summarize
        and not args.skip_summary
        and not args.dry_run
    ):
        from nmp.experiments import summarize_experiment

        summarize_experiment(
            expanded_runs=expanded_path,
            output_dir=root / "summary",
            selection_file=resolve_selection_file(
                plan,
                runs_root=args.runs_root,
                selection_file=args.selection_file,
            ),
            selection_metric=plan.selection.metric,
            selection_mode=plan.selection.mode,
            select_lambda_per_variant=plan.selection.select_lambda_per_variant,
        )


if __name__ == "__main__":
    main()
