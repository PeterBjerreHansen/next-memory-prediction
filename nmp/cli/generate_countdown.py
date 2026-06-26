from __future__ import annotations

import argparse
from pathlib import Path
import random

from nmp.countdown import generate_countdown_example, target_splits


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate Countdown data files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/countdown"))
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--input-numbers", type=int, default=4)
    parser.add_argument("--min-target", type=int, default=10)
    parser.add_argument("--max-target", type=int, default=100)
    parser.add_argument("--max-intermediate", type=int, default=10_000)
    parser.add_argument("--train-samples", type=int, default=500_000)
    parser.add_argument("--val-samples", type=int, default=10_000)
    parser.add_argument("--generalization-samples", type=int, default=10_000)
    return parser.parse_args(argv)


def _write_rows(
    path: Path,
    *,
    rng: random.Random,
    target_pool: list[int],
    samples: int,
    input_numbers: int,
    max_target: int,
    max_intermediate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for _ in range(samples):
            handle.write(
                generate_countdown_example(
                    rng,
                    target_pool=target_pool,
                    input_numbers=input_numbers,
                    max_target=max_target,
                    max_intermediate=max_intermediate,
                )
            )
            handle.write("\n")


def main(argv=None):
    args = parse_args(argv)
    train_targets, heldout_targets = target_splits(
        min_target=args.min_target,
        max_target=args.max_target,
        seed=args.seed,
    )
    suffix = (
        f"b{args.input_numbers}_t{args.max_target}_n{args.train_samples}.txt"
    )
    _write_rows(
        args.output_dir / f"train1_{suffix}",
        rng=random.Random(args.seed),
        target_pool=train_targets,
        samples=args.train_samples,
        input_numbers=args.input_numbers,
        max_target=args.max_target,
        max_intermediate=args.max_intermediate,
    )
    _write_rows(
        args.output_dir / f"val1_{suffix}",
        rng=random.Random(args.seed + 1_000_000),
        target_pool=train_targets,
        samples=args.val_samples,
        input_numbers=args.input_numbers,
        max_target=args.max_target,
        max_intermediate=args.max_intermediate,
    )
    _write_rows(
        args.output_dir / f"val_target1_{suffix}",
        rng=random.Random(args.seed + 2_000_000),
        target_pool=heldout_targets,
        samples=args.generalization_samples,
        input_numbers=args.input_numbers,
        max_target=args.max_target,
        max_intermediate=args.max_intermediate,
    )
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
