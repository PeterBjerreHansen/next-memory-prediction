# Next Memory Prediction

This repository asks whether a language model learns better persistent
memories when it is explicitly trained to predict how those memories evolve.
The first study uses TinyStories, trunk-matched models, and no chunked-memory
machinery.

## Motivation

Token prediction rewards every predictable surface detail, whether or not that
detail is useful as an abstraction. This is less stark for language than for
pixels—tokens are already compressed and semantically dense—but the underlying
question remains: can an auxiliary latent-space objective put more direct
pressure on a model to represent structure that is both predictable and useful
for future prediction?

NextLat predicts a recurrent future latent while retaining next-token
prediction. Next Memory Prediction explores a related inductive bias: a
multi-pass transformer emits a persistent tape of memories
\(m_1,\ldots,m_t\), and later predictions can attend directly to the causal
prefix of that tape.

The hypothesis is not that latent prediction automatically creates semantic
representations. It is that, alongside next-token prediction and with a
stop-gradient target, it may encourage stable predictive structure. Round 1
therefore separates the effect of the MemoryTape architecture from generic
latent-transition regularization and transition regularization applied
specifically to projected memories.

Background material and working notes live in [`documents/`](documents/).
Copied-code, tokenizer, and dataset provenance is pinned in
[`PROVENANCE.md`](PROVENANCE.md).

## Round 1 conditions

1. `transformer_ntp`: causal Transformer with next-token prediction.
2. `memory_tape_ntp`: MemoryTape Transformer with (pass-weighted) NTP loss.
3. `memory_tape_nmp`: the same MemoryTape model with a final-pass memory
   transition objective.
4. `memory_tape_hidden_transition`: the same MemoryTape model with a final-pass
   hidden-state transition objective.

Conditions 2–4 have identical model architecture and initialization under a
matched seed. Conditions 3 and 4 also use identically shaped and initialized
training-only transition predictors.

The comparisons answer:

- 1 → 2: effect of the MemoryTape architecture.
- 2 → 3: effect of explicit-memory transition regularization.
- 2 → 4: effect of generic hidden-state transition regularization.
- 4 → 3: whether projected memories help beyond hidden-state regularization.

## Objective

For either final-pass memory or normalized hidden state \(z_t\), the shared
residual predictor is:

```math
\hat z_{t+1} = z_t + f_\psi([z_t, e(x_{t+1})]).
```

The target is stop-gradient, and transitions into EOS or padding are excluded:

```math
L_\mathrm{transition}
= \operatorname{SmoothL1}(\hat z_{t+1}, \operatorname{sg}(z_{t+1})).
```

For \(K\) MemoryTape passes:

```math
L = \sum_{k=1}^K w_k L_\mathrm{NTP}^{(k)}
    + \lambda_\mathrm{transition}L_\mathrm{transition}.
```

By default, \(w_k = 1/K\). MemoryTape runs can override this with
`objective.ntp_pass_weights` or the CLI flag `--ntp-pass-weights`, for example
`[0.0, 0.0, 0.5, 0.5]` for a 4-pass model. Supplied weights are normalized
internally.

The auxiliary predictor is training-only. Best checkpoints and transition
weights are selected solely by final-pass validation NLL. Validation metrics
are averaged over valid target tokens or valid transitions, not batches.

Optional KL/logit matching, chunked memory, a plain-Transformer transition
condition, and non-teacher-forced objectives are deferred.

## Install and individual runs

```bash
python -m pip install -e ".[dev]"
```

All variants use the same scale preset:

```bash
python -m nmp.cli.train \
  --config configs/smoke.yaml \
  --variant memory_tape_hidden_transition \
  --lambda-transition 1.0 \
  --run-dir runs/smoke/memory_tape_hidden_transition
```

The legacy `--lambda-memory` flag and `objective.lambda_memory` configuration
field remain accepted, but resolved configurations always use
`lambda_transition`. The old variant name `memory_tape_nextlat_no_kl` is also
accepted as a compatibility alias for `memory_tape_hidden_transition`.

For offline work, provide line-delimited local story files:

```bash
python -m nmp.cli.train \
  --config configs/smoke.yaml \
  --variant memory_tape_nmp \
  --train-file tests/fixtures/tinystories_train.txt \
  --val-file tests/fixtures/tinystories_val.txt \
  --run-dir runs/local/nmp
```

Stories that fit within the context receive EOS and padding. Longer stories
are truncated without an artificial EOS token.

Resume, evaluate, probe, and plot:

```bash
python -m nmp.cli.train --resume-from runs/local/nmp/latest.pt --steps 20
python -m nmp.cli.evaluate --run-dir runs/local/nmp
python -m nmp.cli.probe --run-dir runs/local/nmp --steps 100
python -m nmp.cli.plot --run-dir runs/local/nmp
```

MemoryTape generation supports exact `recompute` inference and recurrent
`final_pass` inference. The recurrent path always reuses final-pass memory.

## Experiment matrices

The development matrix contains 30 runs:

- Seeds `0`, `1`, and `2`.
- One run per seed for both NTP baselines.
- Both transition variants at
  `lambda_transition ∈ {0.1, 0.3, 1.0, 3.0}`.

Run and summarize it with:

```bash
bash scripts/run_development_matrix.sh runs
```

To do a quick single-seed dry run of the launcher without training:

```bash
bash scripts/run_development_matrix.sh runs --seeds 0 --dry-run
```

To train MemoryTape runs with later-pass NTP weighting:

```bash
bash scripts/run_development_matrix.sh runs --ntp-pass-weights "[0, 0, 0.5, 0.5]"
```

This selects a transition weight separately for conditions 3 and 4 using the
lowest mean best-checkpoint final-pass validation NLL across seeds.

The reference matrix promotes all four conditions to 100k steps and three
seeds, for 12 runs:

```bash
bash scripts/run_reference_matrix.sh runs
```

The reference launcher consumes
`runs/development/summary/selected_lambdas.json`. Both launchers resume
existing checkpoints and train probes before generating their summaries.

The summary command can also be run directly:

```bash
python -m nmp.cli.summarize \
  --runs-root runs \
  --scale development \
  --output-dir runs/development/summary
```

Each summary verifies the complete expected matrix and emits JSON, CSV,
Markdown, selected weights, and a comparison plot. Reports include individual
seeds, mean and standard deviation, paired-seed deltas, token and transition
losses, parameter counts, throughput, generation agreement, representation
diagnostics, and hidden/memory probes. These are validation results; the same
10,000-story validation set is used for checkpoint selection, weight
selection, and reporting.

## Tests

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Tests and smoke workflows use local fixtures and require no network access.
