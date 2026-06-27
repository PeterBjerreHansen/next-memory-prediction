# Provenance

The active transformer and memory-tape implementation is adapted from:

- Repository: `https://github.com/PeterBjerreHansen/multi-pass-transformer-training`
- Commit: `e76089399d28c5a5a7fac6471455e5fa7f857225`
- Copied components: transformer primitives, causal transformer, multi-pass
  recurrence, memory-tape cross-attention, scalar memory gating, and generation
  modes.
- Deliberate local deletions: non-scalar memory gate modes, unrelated upstream
  architectures, and legacy objective compatibility aliases.

The Countdown task setup is adapted from:

- Repository: `https://github.com/JaydenTeoh/NextLat`
- Commit: `3770be6009cea2b3c455a9ce7f2ca88b504bb955`
- Source paths: `data/countdown.py`, `data/countdown/generate.py`,
  `data/countdown/countdown.py`, and `data/countdown/countdown_utils.py`.
- Local representation: custom atomic integer tokenizer, deterministic
  generator, strict multiset evaluator, and NextLat-compatible loose evaluator
  in `nmp/countdown.py`.

Local extensions include structured outputs, configurable MemoryTape NTP pass
weights, padding-aware objectives, horizon-one memory- and hidden-state
transition objectives, NextLat-style hidden-state self-distillation KL,
Countdown target masking, experiment configuration, checkpointing, diagnostics,
and reporting.

Research-paper references are listed in `documents/references.md`; paper PDFs
are intentionally not committed.
