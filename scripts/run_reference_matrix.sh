#!/usr/bin/env bash
set -euo pipefail

runs_root="runs"
if [[ $# -gt 0 && "$1" != --* ]]; then
  runs_root="$1"
  shift
fi

selection_file="${runs_root}/development/summary/selected_lambdas.json"
if [[ $# -gt 0 && "$1" != --* ]]; then
  selection_file="$1"
  shift
fi

python -m nmp.cli.run_matrix \
  --runs-root "$runs_root" \
  --scale reference \
  --config configs/reference.yaml \
  --selection-file "$selection_file" \
  "$@"

for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    exit 0
  fi
done

python -m nmp.cli.summarize \
  --runs-root "$runs_root" \
  --scale reference \
  --selection-file "$selection_file" \
  --output-dir "${runs_root}/reference/summary"
