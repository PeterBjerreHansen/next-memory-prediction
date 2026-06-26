from __future__ import annotations

import argparse
import json

from nmp.experiments import summarize_round1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and summarize the Round 1 experiment matrix."
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--scale",
        required=True,
        choices=["development", "reference"],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-file")
    args = parser.parse_args(argv)
    result = summarize_round1(
        runs_root=args.runs_root,
        scale=args.scale,
        output_dir=args.output_dir,
        selection_file=args.selection_file,
    )
    print(
        json.dumps(
            {
                "scale": result["scale"],
                "completed_runs": result["completed_runs"],
                "selected_lambdas": result["selected_lambdas"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
