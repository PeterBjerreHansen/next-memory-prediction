#!/usr/bin/env bash
set -euo pipefail

runs_root="runs"
if [[ $# -gt 0 && "$1" != --* ]]; then
  runs_root="$1"
  shift
fi

python -m nmp.cli.run_experiment \
  --experiment configs/experiments/round1_reference_template.yaml \
  --runs-root "$runs_root" \
  "$@"
