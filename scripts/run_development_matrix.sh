#!/usr/bin/env bash
set -euo pipefail

runs_root="runs"
if [[ $# -gt 0 && "$1" != --* ]]; then
  runs_root="$1"
  shift
fi

dry_run=false
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    dry_run=true
  fi
done

if [[ "$dry_run" == false ]]; then
  data_dir="data/countdown"
  required_files=(
    "$data_dir/train1_b4_t100_n100000.txt"
    "$data_dir/val1_b4_t100_n100000.txt"
    "$data_dir/val_target1_b4_t100_n100000.txt"
  )
  required_counts=(100000 10000 10000)

  data_ready() {
    local index path expected actual
    for index in "${!required_files[@]}"; do
      path="${required_files[$index]}"
      expected="${required_counts[$index]}"
      if [[ ! -f "$path" ]]; then
        return 1
      fi
      actual="$(wc -l < "$path" | tr -d '[:space:]')"
      if [[ "$actual" != "$expected" ]]; then
        return 1
      fi
    done
  }

  if ! data_ready; then
    echo "Generating missing or incomplete development Countdown data in $data_dir" >&2
    python -m nmp.cli.generate_countdown_data \
      --output-dir "$data_dir" \
      --seed 444 \
      --input-numbers 4 \
      --train-samples 100000 \
      --val-samples 10000 \
      --generalization-samples 10000
    if ! data_ready; then
      echo "Development Countdown data generation did not create the expected files" >&2
      exit 1
    fi
  fi
fi

python -m nmp.cli.run_experiment \
  --experiment configs/experiments/round1_development.yaml \
  --runs-root "$runs_root" \
  "$@"
