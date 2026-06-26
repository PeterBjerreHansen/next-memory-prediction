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
   logits to predicted-hidden-state student logits, with optional CE on the
   same student path via `objective.transition.lambda_ce`.

The transition objective remains the same teacher-forced one-step residual
predictor:

```math
\hat z_{t+1} = z_t + f_\psi([z_t, e(x_{t+1})]).
```

NTP is masked on Countdown prompt and pause tokens. Solution tokens and EOS are
supervised. Transition loss is masked only for transitions into EOS or padding.
For the KL variant, the LM head is detached for the student logits, teacher
logits are stop-gradient, and optional CE uses the predicted hidden state to
predict the following solution token. The default CE weight is `0.0`.

## Data

The custom Countdown tokenizer assigns one atomic token to every integer from
`0` through `data.countdown_max_intermediate`, plus tokens for `|`, `*`, `/`,
`+`, `-`, `=`, `,`, EOS, and PAD.

Smoke configs use deterministic generated data. For paper-scale files:

```bash
python -m nmp.cli.generate_countdown \
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
  --variant memory_tape_nmp \
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
bash scripts/run_development_matrix.sh runs --seeds 0 --dry-run
```

The manifests select transition weights by `val_accuracy` with `mode: max`.
Reports still include final-pass NLL, perplexity, transition loss, KL/CE
auxiliary loss diagnostics, throughput, representation diagnostics, generated
samples, and linear probes.

## Evaluation

Countdown accuracy is measured by greedy-generating the fixed three-equation
solution from the prompt, parsing equations, checking integer arithmetic,
tracking available operands as a multiset, and requiring the final equation to
reach the target. This strict multiset metric is the primary `val_accuracy`.
Runs also report `val_nextlat_compat_accuracy`, which reproduces the looser
upstream evaluator by using set membership and not consuming operands.

Validation metrics include:

- `val_accuracy`
- `val_strict_multiset_accuracy`
- `val_nextlat_compat_accuracy`
- `val_valid_equation_1`
- `val_valid_equation_2`
- `val_valid_equation_3`
- optional `generalization_accuracy` and
  `generalization_nextlat_compat_accuracy` on held-out target numbers

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Tests and smoke workflows use local Countdown fixtures and require no network
access.
