#!/usr/bin/env bash
set -euo pipefail

runs_root="runs_lightning"
if [[ $# -gt 0 && "$1" != --* ]]; then
  runs_root="$1"
  shift
fi

dry_run=false
has_device=false
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    dry_run=true
  elif [[ "$arg" == "--device" ]]; then
    has_device=true
  fi
done

if [[ "$dry_run" == false ]]; then
  data_dir="data/countdown"
  required_files=(
    "$data_dir/train1_b4_t100_n500000.txt"
    "$data_dir/val1_b4_t100_n500000.txt"
    "$data_dir/val_target1_b4_t100_n500000.txt"
  )
  required_counts=(500000 10000 10000)

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
    echo "Generating missing or incomplete reference Countdown data in $data_dir" >&2
    python -m nmp.cli.generate_countdown_data \
      --output-dir "$data_dir" \
      --seed 444 \
      --input-numbers 4 \
      --train-samples 500000 \
      --val-samples 10000 \
      --generalization-samples 10000
    if ! data_ready; then
      echo "Reference Countdown data generation did not create the expected files" >&2
      exit 1
    fi
  fi
fi

if [[ "$has_device" == false ]]; then
  python -m nmp.cli.run_experiment \
    --experiment configs/experiments/round1_lightning_baselines.yaml \
    --runs-root "$runs_root" \
    --device "${LIGHTNING_DEVICE:-cuda}" \
    "$@"
else
  python -m nmp.cli.run_experiment \
    --experiment configs/experiments/round1_lightning_baselines.yaml \
    --runs-root "$runs_root" \
    "$@"
fi
