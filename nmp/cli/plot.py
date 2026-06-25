from __future__ import annotations

import argparse

from nmp.plotting import plot_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot a saved NMP run.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    for path in plot_run(args.run_dir):
        print(path)


if __name__ == "__main__":
    main()

