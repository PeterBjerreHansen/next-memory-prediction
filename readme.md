# Next Memory Prediction

This repository asks whether a language model learns better persistent
memories when it is explicitly trained to predict how those memories evolve.
The first study is deliberately small and controlled: TinyStories,
trunk-matched models, and no chunked-memory machinery.

## Motivation

Token prediction rewards every predictable surface detail, whether or not that
detail is useful as an abstraction. This is less stark for language than for
pixels—tokens are already compressed and semantically dense—but the underlying
question remains: can an auxiliary latent-space objective put more direct
pressure on a model to represent structure that is both predictable and useful
for future prediction?

NextLat predicts a recurrent future latent while retaining next-token
prediction. Its latent state is an information bottleneck through which
successive predictions pass. Next Memory Prediction explores a different
inductive bias: a multi-pass transformer emits a persistent tape of memories
\(m_1,\ldots,m_t\), and later predictions can attend directly to the whole
causal prefix of that tape. An auxiliary temporal objective then asks each
memory to help predict its successor.

The hypothesis is not that latent prediction automatically creates semantic
representations. It is that, alongside next-token prediction and with a
stop-gradient target, it may encourage memories to encode stable predictive
structure that survives across positions and passes. The repository is built
to test that claim rather than assume it.

Background material and working notes live in [`documents/`](documents/),
including the NextLat and Belief State Transformer papers and a note outlining
the case against predicting latents.

## Experiment

The study compares three models:

1. `transformer_ntp`: a causal transformer trained with next-token prediction.
2. `memory_tape_ntp`: a multi-pass Memory-Tape Transformer trained with
   next-token prediction on every pass.
3. `memory_tape_nmp`: the same Memory-Tape Transformer with auxiliary temporal
   next-memory prediction.

Chunked MPTT and a direct NextLat baseline are deferred. The active package
contains no chunk IDs, chunk tokens, or chunked-attention paths.

For final-pass memory tape \(M=[m_1,\ldots,m_T]\), a residual dynamics MLP
predicts the next memory from the current memory and next-token embedding:

```math
\hat m_{t+1} = m_t + f_\psi([m_t, e(x_{t+1})]).
```

The target memory is stop-gradient. Transitions into EOS or padding are
excluded:

```math
L_\mathrm{NMP}
= \operatorname{SmoothL1}(\hat m_{t+1}, \operatorname{sg}(m_{t+1})).
```

The current objective is:

```math
L = \frac{1}{K}\sum_{k=1}^K L_\mathrm{NTP}^{(k)}
    + \lambda_\mathrm{memory}L_\mathrm{NMP}.
```

Optional stop-gradient logit matching via KL, analogous to NextLat's
consistency term, is planned but intentionally not part of this first
implementation.

## Repository layout

- `nmp/`: models, objectives, data, training, evaluation, probes, and plots.
- `configs/`: one reusable preset for each experiment scale.
- `scripts/`: experiment-matrix helpers.
- `tests/`: offline unit and end-to-end smoke tests.
- `documents/`: papers and research notes.

Copied-code and data provenance is pinned in
[`PROVENANCE.md`](PROVENANCE.md).

## Install and run

```bash
python -m pip install -e ".[dev]"
```

Run any model variant from the same smoke preset:

```bash
python -m nmp.cli.train \
  --config configs/smoke.yaml \
  --variant transformer_ntp \
  --run-dir runs/smoke/transformer

python -m nmp.cli.train \
  --config configs/smoke.yaml \
  --variant memory_tape_ntp \
  --run-dir runs/smoke/memory_tape

python -m nmp.cli.train \
  --config configs/smoke.yaml \
  --variant memory_tape_nmp \
  --run-dir runs/smoke/memory_tape_nmp
```

The default data source is the pinned Hugging Face TinyStories revision. For
offline work, supply line-delimited local files:

```bash
python -m nmp.cli.train \
  --config configs/smoke.yaml \
  --variant memory_tape_nmp \
  --train-file tests/fixtures/tinystories_train.txt \
  --val-file tests/fixtures/tinystories_val.txt \
  --run-dir runs/local/nmp
```

Resume, evaluate, probe, and plot:

```bash
python -m nmp.cli.train --resume-from runs/local/nmp/latest.pt --steps 20
python -m nmp.cli.evaluate --run-dir runs/local/nmp
python -m nmp.cli.probe --run-dir runs/local/nmp --steps 100
python -m nmp.cli.plot --run-dir runs/local/nmp
```

Each run records its resolved configuration, provenance, JSONL metrics,
checkpoints, generated samples, evaluation results, probe results, and plots.
Best checkpoints are selected by final-pass validation NLL, not combined
training loss.

## Protocol

Development runs use seeds `0`, `1`, and `2`. For `memory_tape_nmp`, sweep
`lambda_memory` over `0.1`, `0.3`, `1.0`, and `3.0`, then choose the reference
value by lowest mean final-pass validation NLL across seeds. The helper script
is [`scripts/run_development_matrix.sh`](scripts/run_development_matrix.sh).

Run the CPU test suite with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Tests and smoke workflows use local fixtures and require no network access.

# Provenance

The active transformer and memory-tape implementation is adapted from:

- Repository: `https://github.com/PeterBjerreHansen/multi-pass-transformer-training`
- Commit: `e76089399d28c5a5a7fac6471455e5fa7f857225`
- Copied components: transformer primitives, causal transformer, multi-pass
  recurrence, memory-tape cross-attention, memory gates, and generation modes.

The vendored 1,000-token TinyStories tokenizer comes from:

- Repository: `https://github.com/JaydenTeoh/NextLat`
- Commit: `3770be6009cea2b3c455a9ce7f2ca88b504bb955`
- Source path: `data/tinystories/tokenizer.json`
- Local representation: the exact tokenizer JSON is compressed and embedded
  in `nmp/tokenizer_asset.py`; it is reconstructed in memory when loaded.

The default dataset is:

- Hugging Face dataset: `karpathy/tinystories-gpt4-clean`
- Revision: `0397e27157956705a0260709da3095bb9c43d6a7`
- License declared by the dataset card: CDLA-Sharing-1.0

Local extensions include structured model outputs, padding-aware objectives,
temporal next-memory prediction, experiment configuration, checkpointing,
TinyStories loading, representation diagnostics, probing, and plotting.
