from __future__ import annotations

import argparse
import json

from nmp.evaluation import evaluate_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved NMP run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    result = evaluate_run(
        args.run_dir,
        checkpoint_name=args.checkpoint,
        device_override=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

