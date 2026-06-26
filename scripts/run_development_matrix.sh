#!/usr/bin/env bash
set -euo pipefail

runs_root="runs"
if [[ $# -gt 0 && "$1" != --* ]]; then
  runs_root="$1"
  shift
fi

python -m nmp.cli.run_matrix \
  --runs-root "$runs_root" \
  --scale development \
  --config configs/development.yaml \
  "$@"

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    exit 0
  fi
done

python -m nmp.cli.summarize \
  --runs-root "$runs_root" \
  --scale development \
  --output-dir "${runs_root}/development/summary"
