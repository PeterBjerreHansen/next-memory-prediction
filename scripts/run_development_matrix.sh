#!/usr/bin/env bash
set -euo pipefail

for seed in 0 1 2; do
  python -m nmp.cli.train \
    --config configs/development.yaml \
    --variant transformer_ntp \
    --seed "$seed" \
    --run-dir "runs/development/transformer/seed_${seed}"

  python -m nmp.cli.train \
    --config configs/development.yaml \
    --variant memory_tape_ntp \
    --seed "$seed" \
    --run-dir "runs/development/memory_tape/seed_${seed}"

  for lambda_memory in 0.1 0.3 1.0 3.0; do
    python -m nmp.cli.train \
      --config configs/development.yaml \
      --variant memory_tape_nmp \
      --seed "$seed" \
      --lambda-memory "$lambda_memory" \
      --run-dir "runs/development/memory_tape_nmp/lambda_${lambda_memory}/seed_${seed}"
  done
done
