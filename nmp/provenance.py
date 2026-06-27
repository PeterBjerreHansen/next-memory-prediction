from __future__ import annotations

from typing import Any


MODEL_SOURCE_REPOSITORY = "https://github.com/PeterBjerreHansen/multi-pass-transformer-training"
MODEL_SOURCE_COMMIT = "e76089399d28c5a5a7fac6471455e5fa7f857225"
COUNTDOWN_SOURCE_REPOSITORY = "https://github.com/JaydenTeoh/NextLat"
COUNTDOWN_SOURCE_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
COUNTDOWN_PAPER = "https://arxiv.org/abs/2511.05963"


def provenance_manifest() -> dict[str, Any]:
    return {
        "model_source": {
            "repository": MODEL_SOURCE_REPOSITORY,
            "commit": MODEL_SOURCE_COMMIT,
        },
        "countdown_source": {
            "repository": COUNTDOWN_SOURCE_REPOSITORY,
            "commit": COUNTDOWN_SOURCE_COMMIT,
            "source_paths": [
                "data/countdown.py",
                "data/countdown/generate.py",
                "data/countdown/countdown.py",
                "data/countdown/countdown_utils.py",
            ],
            "paper": COUNTDOWN_PAPER,
            "local_representation": (
                "custom atomic integer tokenizer, generator, strict multiset "
                "evaluator, and NextLat-compatible loose evaluator implemented "
                "in nmp/countdown.py"
            ),
        },
    }
