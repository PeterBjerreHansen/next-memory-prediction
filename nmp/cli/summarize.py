from __future__ import annotations

import argparse
import json
from pathlib import Path

from nmp.experiment_plan import experiment_dir, load_experiment_plan
from nmp.experiments import summarize_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and summarize an experiment manifest."
    )
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--expanded-runs", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-file")
    args = parser.parse_args(argv)
    if args.expanded_runs is None:
        if args.experiment is None:
            raise ValueError("--experiment or --expanded-runs is required")
        plan = load_experiment_plan(args.experiment)
        root = experiment_dir(plan, args.runs_root)
        expanded_runs = root / "expanded_runs.jsonl"
        output_dir = args.output_dir or root / "summary"
        selection_metric = plan.selection.metric
        selection_mode = plan.selection.mode
        select_lambda_per_variant = plan.selection.select_lambda_per_variant
    else:
        expanded_runs = args.expanded_runs
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = expanded_runs.parent / "summary"
        selection_metric = "final_pass_nll"
        selection_mode = "min"
        select_lambda_per_variant = True
    result = summarize_experiment(
        expanded_runs=expanded_runs,
        output_dir=output_dir,
        selection_file=args.selection_file,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        select_lambda_per_variant=select_lambda_per_variant,
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "completed_runs": result["completed_runs"],
                "selected_lambdas": result["selected_lambdas"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
