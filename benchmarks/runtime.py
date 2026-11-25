"""Environment checks and resource sampling for benchmark runs."""

import importlib.util
import os
import shutil
import sys
import time
from contextlib import contextmanager
from threading import Event, Thread

from benchmark_io import code_hashes, sha256
from model_sources import get_source
from tasks import get_task


RUNTIME_MODULES = (
    "torch", "flash_attn", "lightning", "torchmetrics",
    "pandas", "numpy", "scipy", "sklearn", "h5py", "psutil", "tqdm",
)


def input_files(config, bundle, assets):
    paths = {}
    for relative in config["inputs"]:
        staged = bundle / "inputs" / relative
        paths[relative] = staged if staged.is_file() else assets / relative
    return paths


def preflight(config, bundle, assets):
    from importlib.metadata import version

    paths = input_files(config, bundle, assets)
    missing = [str(path) for path in paths.values() if not path.is_file() or not path.stat().st_size]
    if missing:
        raise ValueError("Missing or empty official assets:\n  " + "\n  ".join(missing))
    files = {}
    for name, path in paths.items():
        stat = path.stat()
        record = {"path": str(path), "bytes": stat.st_size}
        if name.endswith(".hdf5"):
            record["mtime_ns"] = stat.st_mtime_ns
        else:
            record["sha256"] = sha256(path)
            expected = config["input_sha256"].get(name)
            if expected and record["sha256"] != expected:
                raise ValueError(f"Official input checksum mismatch: {path}")
        files[name] = record

    task = get_task(config["task"])
    modules = (*RUNTIME_MODULES, *task.REQUIRED_MODULES)
    if "training" in config:
        modules += ("tensorboard",)
    missing_modules = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise RuntimeError("Missing Python dependencies in the container: " + ", ".join(missing_modules))
    if task.USES_BEDTOOLS and shutil.which("bedtools") is None:
        raise RuntimeError("Install bedtools in the container before running this official tutorial.")

    import flash_attn
    import torch

    source_metadata = get_source(config).metadata(config, bundle)
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("These benchmarks require one visible BF16-capable CUDA GPU.")
    return {
        "task": config["task"],
        **source_metadata,
        "task_config_sha256": sha256(bundle / "task.json"),
        "benchmark_code_sha256": code_hashes(bundle),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attention": flash_attn.__version__,
        "python": sys.version.split()[0],
        "software": {name: version(name) for name in ("lightning", "torchmetrics", "pandas", "numpy", "scipy", "scikit-learn")},
        "implementation": "python-tutorial-protocol",
        "input_files": files,
    }


def sample_rss(stop, peak):
    import psutil

    process = psutil.Process()
    while True:
        total = 0
        for child in [process, *process.children(recursive=True)]:
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        peak[0] = max(peak[0], total)
        if stop.wait(0.2):
            return


def sample_stage(torch, action):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = action()
    torch.cuda.synchronize()
    return result, {
        "wall_seconds": time.perf_counter() - start,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


@contextmanager
def sample_cpu_memory():
    peak = [0]
    stop = Event()
    sampler = Thread(target=sample_rss, args=(stop, peak), daemon=True)
    sampler.start()
    try:
        yield peak
    finally:
        stop.set()
        sampler.join()
