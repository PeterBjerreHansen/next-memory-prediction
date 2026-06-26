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

Local extensions include structured outputs, padding-aware objectives,
memory- and hidden-state transition objectives, experiment configuration,
checkpointing, TinyStories loading, diagnostics, probing, and reporting.
