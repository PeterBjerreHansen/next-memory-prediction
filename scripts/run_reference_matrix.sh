#!/usr/bin/env bash
set -euo pipefail

runs_root="${1:-runs}"
selection_file="${2:-${runs_root}/development/summary/selected_lambdas.json}"

selected_lambda() {
  python - "$selection_file" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
values = payload.get("selected_lambdas", payload)
print(values[sys.argv[2]])
PY
}

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
      --config configs/reference.yaml
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

memory_lambda="$(selected_lambda memory_tape_nmp)"
hidden_lambda="$(selected_lambda memory_tape_hidden_transition)"

for seed in 0 1 2; do
  run_one \
    transformer_ntp \
    "$seed" \
    "${runs_root}/reference/transformer_ntp/seed_${seed}"
  run_one \
    memory_tape_ntp \
    "$seed" \
    "${runs_root}/reference/memory_tape_ntp/seed_${seed}"
  run_one \
    memory_tape_nmp \
    "$seed" \
    "${runs_root}/reference/memory_tape_nmp/lambda_${memory_lambda}/seed_${seed}" \
    "$memory_lambda"
  run_one \
    memory_tape_hidden_transition \
    "$seed" \
    "${runs_root}/reference/memory_tape_hidden_transition/lambda_${hidden_lambda}/seed_${seed}" \
    "$hidden_lambda"
done

python -m nmp.cli.summarize \
  --runs-root "$runs_root" \
  --scale reference \
  --selection-file "$selection_file" \
  --output-dir "${runs_root}/reference/summary"
