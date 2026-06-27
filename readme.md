# Next Memory Prediction on Countdown

This repository asks whether a model learns better persistent memories when it
is explicitly trained to predict how those memories evolve. The current study
uses the arithmetic Countdown task from the NextLat paper rather than
TinyStories.

Countdown is not a literal counting-down task. Each example gives four input
numbers and a target. The model must generate three valid arithmetic equations
that combine the available numbers into the target, for example:

```text
2,3,4,5,14|2+3=5,5+5=10,10+4=14
```

The prompt side is followed by eight `|` pause tokens during tokenization so
the model has extra compute steps before producing the solution.

## Conditions

1. `transformer_ntp`: causal Transformer with next-token prediction.
2. `memory_tape_ntp`: MemoryTape Transformer with pass-weighted NTP loss.
3. `memory_tape_nmp`: MemoryTape model with final-pass memory transition loss.
4. `memory_tape_hidden_transition`: MemoryTape model with final-pass hidden-state
   transition loss.
5. `memory_tape_hidden_transition_kl`: hidden-state transition plus a
   NextLat-style self-distillation KL loss from actual hidden-state teacher
   logits to predicted-hidden-state student logits.

## Protocol Matrix

| Condition | Architecture | Transition target | NTP mask | SmoothL1 mask | KL mask |
| --- | --- | --- | --- | --- | --- |
| `transformer_ntp` | Transformer | none | solution + EOS | n/a | n/a |
| `memory_tape_ntp` | MemoryTape | none | solution + EOS | n/a | n/a |
| `memory_tape_nmp` | MemoryTape | final memory | solution + EOS | non-EOS/non-PAD transitions | n/a |
| `memory_tape_hidden_transition` | MemoryTape | final hidden | solution + EOS | non-EOS/non-PAD transitions | n/a |
| `memory_tape_hidden_transition_kl` | MemoryTape | final hidden | solution + EOS | non-EOS/non-PAD transitions | target-masked positions |

The transition objective remains the same teacher-forced one-step residual
predictor:

```math
\hat z_{t+1} = z_t + f_\psi([z_t, e(x_{t+1})]).
```

NTP is masked on Countdown prompt and pause tokens. Solution tokens and EOS are
supervised. Transition loss is masked only for transitions into EOS or padding.
For the KL condition, the LM head is detached for the student logits, teacher
logits are stop-gradient, and `objective.lambda_kl` weights the
self-distillation term.

## Data

The custom Countdown tokenizer assigns one atomic token to every integer from
`0` through `data.countdown_max_intermediate`, plus tokens for `|`, `*`, `/`,
`+`, `-`, `=`, `,`, EOS, and PAD.

Training reads explicit Countdown files. Use the generator CLI to create
paper-scale files:

```bash
python -m nmp.cli.generate_countdown_data \
  --output-dir data/countdown \
  --seed 444 \
  --input-numbers 4 \
  --train-samples 500000 \
  --val-samples 10000 \
  --generalization-samples 10000
```

Then run with local files:

```bash
python -m nmp.cli.train \
  --config configs/scales/smoke.yaml \
  --architecture memory_tape \
  --transition memory \
  --train-file data/countdown/train1_b4_t100_n500000.txt \
  --val-file data/countdown/val1_b4_t100_n500000.txt \
  --generalization-file data/countdown/val_target1_b4_t100_n500000.txt \
  --run-dir runs/local/countdown-nmp
```

## Experiments

Run the development matrix:

```bash
bash scripts/run_development_matrix.sh runs
```

Run a dry smoke expansion:

```bash
python -m nmp.cli.run_experiment \
  --experiment configs/experiments/round1_smoke.yaml \
  --runs-root runs \
  --seeds 0 \
  --dry-run
```

The manifests select checkpoints and transition weights by `final_pass_nll` with
`mode: min`. Countdown accuracy is reported separately because exact generated
solution accuracy is sparse early in training.
Reports still include final-pass NLL, perplexity, transition loss, KL
auxiliary loss diagnostics, throughput, representation diagnostics, generated
samples, and per-equation validity metrics.

The development scale uses 100k generated training examples. During training,
NLL validation/checkpointing is intentionally less frequent than the original
small-data pilot, while `evaluation.training_accuracy_interval` enables a
lightweight periodic greedy-generation accuracy diagnostic over a small fixed
validation slice.

## Evaluation

Countdown accuracy is measured by greedy-generating the configured number of
solution equations from the prompt, parsing equations, checking integer
arithmetic, tracking available operands as a multiset, and requiring the final
equation to reach the target. This strict multiset metric is the primary
`val_accuracy`.
Runs also report `val_nextlat_compat_accuracy`, which reproduces the looser
upstream evaluator by using set membership and not consuming operands.
Final evaluation uses `evaluation.accuracy_batches`; the default `null` walks
the full validation and held-out-target corpora once.

Validation metrics include:

- `val_accuracy`
- `val_strict_multiset_accuracy`
- `val_nextlat_compat_accuracy`
- `val_valid_equation_N` for each configured equation position
- optional `generalization_accuracy` and
  `generalization_nextlat_compat_accuracy` on held-out target numbers

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Tests and smoke workflows use local Countdown fixtures and require no network
access.
