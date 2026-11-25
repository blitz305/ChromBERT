"""Configuration, checksums, and JSON used by the CLI and runtime."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def profiling_enabled(config):
    enabled = config.get("profile", config.get("profile_gep", False))
    if type(enabled) is not bool:
        raise ValueError("profile must be true or false.")
    return enabled


# Keep the bundle's code inventory independent of the CLI and the model source.
CODE = Path(__file__).resolve().parent
BENCHMARK_SCRIPTS = (
    "benchmark.py", "benchmark_io.py", "runner.py", "training.py", "launcher.py",
    "profiling.py", "interpretability.py", "archives.py", "predictions.py",
    "runtime.py", "model_sources.py", "variants.py",
    *(f"tasks/{path.name}" for path in sorted((CODE / "tasks").glob("*.py"))),
)


def code_hashes(bundle):
    return {name: sha256(bundle / name) for name in BENCHMARK_SCRIPTS}
