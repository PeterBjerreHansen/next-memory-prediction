from __future__ import annotations

import argparse
from pathlib import Path

from nmp.checkpoint import load_checkpoint
from nmp.config import ExperimentConfig, VARIANTS, load_config
from nmp.evaluation import evaluate_run
from nmp.plotting import plot_run
from nmp.training import train_experiment


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a TinyStories NMP experiment.")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
    )
    parser.add_argument("--lambda-transition", type=float)
    parser.add_argument("--lambda-memory", type=float)
    parser.add_argument("--train-file")
    parser.add_argument("--val-file")
    return parser.parse_args(argv)


def resolve_config(args) -> tuple[ExperimentConfig, Path]:
    if args.resume_from is not None:
        checkpoint = load_checkpoint(args.resume_from)
        config = ExperimentConfig.from_dict(checkpoint["config"])
        run_dir = args.run_dir or (
            args.resume_from if args.resume_from.is_dir() else args.resume_from.parent
        )
    else:
        if args.config is None or args.run_dir is None:
            raise ValueError("--config and --run-dir are required for a new run")
        config = load_config(args.config)
        run_dir = args.run_dir
    if args.steps is not None:
        config.training.train_steps = args.steps
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.variant is not None:
        config.model.variant = args.variant
        config.name = f"{config.name}-{args.variant}"
    if args.lambda_transition is not None and args.lambda_memory is not None:
        raise ValueError(
            "--lambda-transition and legacy --lambda-memory are mutually exclusive"
        )
    lambda_transition = (
        args.lambda_transition
        if args.lambda_transition is not None
        else args.lambda_memory
    )
    if lambda_transition is not None:
        config.objective.lambda_transition = lambda_transition
    if args.train_file is not None or args.val_file is not None:
        if args.train_file is None or args.val_file is None:
            raise ValueError("--train-file and --val-file must be provided together")
        config.data.source = "local"
        config.data.train_file = args.train_file
        config.data.val_file = args.val_file
    config.validate()
    return config, Path(run_dir)


def main(argv=None):
    args = parse_args(argv)
    config, run_dir = resolve_config(args)
    train_experiment(
        config,
        run_dir=run_dir,
        resume_from=args.resume_from,
    )
    evaluate_run(run_dir, device_override=args.device)
    plot_run(run_dir)
    print(run_dir.resolve())


if __name__ == "__main__":
    main()
