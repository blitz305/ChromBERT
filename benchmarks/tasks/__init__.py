"""Find a task implementation and write its prepared splits."""

from importlib import import_module

from .gep import OFFICIAL_CHECKPOINT as GEP_OFFICIAL_CHECKPOINT

TASKS = {
    "gep": "gep",
    "gep-all": "gep",
    "gep-sampled": "gep",
    "ezh2": "ezh2",
    "cistrome": "cistrome",
    "cistrome-sampled": "cistrome",
    "eqtl": "eqtl",
    "eqtl-official": "eqtl",
    "starr": "starr",
}


def get_task(name):
    return import_module(f".{TASKS[name]}", __name__).PROTOCOL


def prepare_splits(config, inputs, destination):
    task = get_task(config["task"])
    frames, audit = task.prepare_split_data(config, inputs)

    actual = {name: len(frame) for name, frame in frames.items()}
    expected = config["expected_counts"]
    if actual.keys() != expected.keys() or any(
        count is not None and count != actual[name] for name, count in expected.items()
    ):
        raise ValueError(f"Benchmark split sizes differ: expected {expected}, got {actual}.")
    if audit is not None:
        audit.update(task.split_audit(config, frames))
        if config.get("smoke"):
            audit["full"] = {key: audit[key] for key in ("splits", "overlap")}
            seed = config["split"]["seed"]
            frames = {name: task.smoke_sample(frame, seed) for name, frame in frames.items()}
            audit.update(task.split_audit(config, frames))

    destination.mkdir()
    paths = {}
    for name, frame in frames.items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=task.WRITE_SPLIT_INDEX)
        paths[name] = path
    return paths, audit


def predict(task, model, data, profile=None):
    """Keep the saved GELU comparison scripts able to import batch prediction."""
    from runner import predict_batches

    return predict_batches(get_task(task), model, data, profile)


def setup_training(config, assets, paths):
    """Compatibility entry point for archived diagnostics."""
    return get_task(config["task"]).setup_training(config, assets, paths)


def setup_inference(config, assets, paths, data, options, checkpoint):
    """Compatibility entry point for archived diagnostics."""
    return get_task(config["task"]).setup_inference(config, assets, paths, data, options, checkpoint)
