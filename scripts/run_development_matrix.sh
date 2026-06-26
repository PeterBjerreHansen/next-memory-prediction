#!/usr/bin/env bash
set -euo pipefail

runs_root="${1:-runs}"

run_one() {
  local variant="$1"
  local seed="$2"
  local run_dir="$3"
  local lambda_transition="${4:-}"

  if [[ -f "${run_dir}/latest.pt" ]]; then
    python -m nmp.cli.train \
      --resume-from "${run_dir}/latest.pt"
  else
    command=(
      python -m nmp.cli.train
      --config configs/development.yaml
      --variant "$variant"
      --seed "$seed"
      --run-dir "$run_dir"
    )
    if [[ -n "$lambda_transition" ]]; then
      command+=(--lambda-transition "$lambda_transition")
    fi
    "${command[@]}"
  fi
  python -m nmp.cli.probe --run-dir "$run_dir"
}

for seed in 0 1 2; do
  run_one \
    transformer_ntp \
    "$seed" \
    "${runs_root}/development/transformer_ntp/seed_${seed}"
  run_one \
    memory_tape_ntp \
    "$seed" \
    "${runs_root}/development/memory_tape_ntp/seed_${seed}"

  for variant in memory_tape_nmp memory_tape_hidden_transition; do
    for lambda_transition in 0.1 0.3 1.0 3.0; do
      run_one \
        "$variant" \
        "$seed" \
        "${runs_root}/development/${variant}/lambda_${lambda_transition}/seed_${seed}" \
        "$lambda_transition"
    done
  done
done

python -m nmp.cli.summarize \
  --runs-root "$runs_root" \
  --scale development \
  --output-dir "${runs_root}/development/summary"
