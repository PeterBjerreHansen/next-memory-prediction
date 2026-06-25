from __future__ import annotations

from typing import Any


MODEL_SOURCE_REPOSITORY = "https://github.com/PeterBjerreHansen/multi-pass-transformer-training"
MODEL_SOURCE_COMMIT = "e76089399d28c5a5a7fac6471455e5fa7f857225"
TOKENIZER_SOURCE_REPOSITORY = "https://github.com/JaydenTeoh/NextLat"
TOKENIZER_SOURCE_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
DATASET_NAME = "karpathy/tinystories-gpt4-clean"
DATASET_REVISION = "0397e27157956705a0260709da3095bb9c43d6a7"


def provenance_manifest() -> dict[str, Any]:
    return {
        "model_source": {
            "repository": MODEL_SOURCE_REPOSITORY,
            "commit": MODEL_SOURCE_COMMIT,
        },
        "tokenizer_source": {
            "repository": TOKENIZER_SOURCE_REPOSITORY,
            "commit": TOKENIZER_SOURCE_COMMIT,
            "source_path": "data/tinystories/tokenizer.json",
            "local_representation": "compressed in nmp/tokenizer_asset.py",
        },
        "dataset": {
            "name": DATASET_NAME,
            "revision": DATASET_REVISION,
            "license": "cdla-sharing-1.0",
        },
    }
