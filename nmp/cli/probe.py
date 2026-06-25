from __future__ import annotations

import argparse
import json

from nmp.plotting import plot_probes
from nmp.probing import train_probes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train future-token linear probes.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    rows = train_probes(
        args.run_dir,
        steps=args.steps,
        device_override=args.device,
    )
    plot_probes(args.run_dir)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

