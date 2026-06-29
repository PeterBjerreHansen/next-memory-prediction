#!/usr/bin/env bash
set -euo pipefail

runs_root="runs"
if [[ $# -gt 0 && "$1" != --* ]]; then
  runs_root="$1"
  shift
fi

has_steps=false
has_selection_file=false
for arg in "$@"; do
  if [[ "$arg" == "--steps" ]]; then
    has_steps=true
  elif [[ "$arg" == "--selection-file" ]]; then
    has_selection_file=true
  fi
done

extra_args=()
experiment_file="configs/experiments/round1_reference_template.yaml"
selection_file="$runs_root/round1_development/summary/selected_lambdas.json"
if [[ "$has_steps" == true && "$has_selection_file" == false && ! -f "$selection_file" ]]; then
  benchmark_selection="$runs_root/round1_reference_benchmark_selected_lambdas.json"
  mkdir -p "$(dirname "$benchmark_selection")"
  cat > "$benchmark_selection" <<'JSON'
{
  "selected_lambdas": {
    "memory_tape_nmp": 1.0,
    "memory_tape_hidden_transition": 0.1,
    "memory_tape_hidden_transition_kl": 0.3
  }
}
JSON
  echo "Using benchmark transition weights from $benchmark_selection" >&2
  extra_args=(--selection-file "$benchmark_selection")
fi

reference_train_file="data/countdown/train1_b4_t100_n500000.txt"
if [[ "$has_steps" == true && ! -f "$reference_train_file" ]]; then
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
    echo "Generating missing or incomplete benchmark Countdown data in $data_dir" >&2
    python -m nmp.cli.generate_countdown_data \
      --output-dir "$data_dir" \
      --seed 444 \
      --input-numbers 4 \
      --train-samples 100000 \
      --val-samples 10000 \
      --generalization-samples 10000
    if ! data_ready; then
      echo "Benchmark Countdown data generation did not create the expected files" >&2
      exit 1
    fi
  fi

  benchmark_experiment="$runs_root/round1_reference_benchmark.yaml"
  mkdir -p "$(dirname "$benchmark_experiment")"
  cat > "$benchmark_experiment" <<'YAML'
name: round1_reference_benchmark
base_config: configs/scales/reference.yaml
runs_root: runs
selection_file: round1_development/summary/selected_lambdas.json

selection:
  metric: final_pass_nll
  mode: min
  select_lambda_per_variant: false

shared_overrides:
  data:
    train_file: data/countdown/train1_b4_t100_n100000.txt
    val_file: data/countdown/val1_b4_t100_n100000.txt
    generalization_file: data/countdown/val_target1_b4_t100_n100000.txt
  training:
    eval_batches: 1
    compile: false
  objective:
    ntp_pass_weights: [0.0, 0.0, 0.5, 0.5]

seeds: [0]

conditions:
  - name: transformer_ntp
    architecture: transformer
    transition: none
    lambda_transition: null

  - name: memory_tape_ntp
    architecture: memory_tape
    transition: none
    lambda_transition: null

  - name: memory_tape_nmp
    architecture: memory_tape
    transition: memory
    lambda_transition: from_selection

  - name: memory_tape_hidden_transition
    architecture: memory_tape
    transition: hidden
    lambda_transition: from_selection

  - name: memory_tape_hidden_transition_kl
    architecture: memory_tape
    transition: hidden_kl
    lambda_transition: from_selection

post_run:
  evaluate: false
  plot: false
  summarize: false
YAML
  echo "Using benchmark reference manifest from $benchmark_experiment" >&2
  experiment_file="$benchmark_experiment"
fi

python -m nmp.cli.run_experiment \
  --experiment "$experiment_file" \
  --runs-root "$runs_root" \
  "${extra_args[@]}" \
  "$@"
