from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from nmp.experiments import (
    expected_run_specs,
    format_lambda,
    load_selected_lambdas,
    run_directory,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the Round 1 development or reference matrix."
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--scale",
        required=True,
        choices=["development", "reference"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--selection-file")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional subset of seeds, useful for quick local runs.",
    )
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--ntp-pass-weights",
        help="Forwarded to MemoryTape runs only, e.g. '0,0,0.5,0.5'.",
    )
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def train_command(
    *,
    config_path: str,
    run_dir: Path,
    spec,
    device: str | None,
    steps: int | None,
    ntp_pass_weights: str | None,
) -> list[str]:
    latest = run_dir / "latest.pt"
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
            config_path,
            "--variant",
            spec.variant,
            "--seed",
            str(spec.seed),
            "--run-dir",
            str(run_dir),
        ]
        if spec.lambda_transition is not None:
            command.extend(
                ["--lambda-transition", format_lambda(spec.lambda_transition)]
            )
        if spec.variant != "transformer_ntp" and ntp_pass_weights is not None:
            command.extend(["--ntp-pass-weights", ntp_pass_weights])
    if device is not None:
        command.extend(["--device", device])
    if steps is not None:
        command.extend(["--steps", str(steps)])
    return command


def probe_command(run_dir: Path, *, device: str | None) -> list[str]:
    command = [sys.executable, "-m", "nmp.cli.probe", "--run-dir", str(run_dir)]
    if device is not None:
        command.extend(["--device", device])
    return command


def run(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main(argv=None):
    args = parse_args(argv)
    selected = None
    if args.scale == "reference":
        if args.selection_file is None:
            raise ValueError("--selection-file is required for reference runs")
        selected = load_selected_lambdas(args.selection_file)
    specs = expected_run_specs(args.scale, selected_lambdas=selected)
    if args.seeds is not None:
        wanted = set(args.seeds)
        specs = [spec for spec in specs if spec.seed in wanted]

    for spec in specs:
        run_dir = run_directory(args.runs_root, args.scale, spec)
        run(
            train_command(
                config_path=args.config,
                run_dir=run_dir,
                spec=spec,
                device=args.device,
                steps=args.steps,
                ntp_pass_weights=args.ntp_pass_weights,
            ),
            dry_run=args.dry_run,
        )
        if not args.skip_probe:
            run(probe_command(run_dir, device=args.device), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
