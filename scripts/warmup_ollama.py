"""Manual Ollama warm-up entrypoint."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ollama_warmup import OllamaWarmupError, warmup_all_models  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warm up configured Ollama models.")
    parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    print("Starting Ollama warm-up...")
    try:
        results = warmup_all_models()
    except OllamaWarmupError as exc:
        print(f"Ollama warm-up failed: {exc}")
        return 1

    if not results:
        print("Ollama warm-up completed: no models enabled for warm-up.")
        return 0

    for result in results:
        print(
            f"{result.model_type} model warm-up succeeded: "
            f"{result.model_name} ({result.elapsed_sec:.2f}s)"
        )
    print("Ollama warm-up completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
