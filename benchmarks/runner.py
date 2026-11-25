"""Orchestrate benchmark training, inference, and artifact collection."""

import gc
import shutil
import tempfile
import time
from pathlib import Path
from types import ModuleType

from archives import evaluation_stages, reference_run, verify_results
from benchmark_io import profiling_enabled, read_json, sha256, write_json
from interpretability import FACTOR_LIST, extract_regulator_rankings
from model_sources import get_source
from predictions import predict_batches as _predict_batches
from predictions import prediction_scores as _prediction_scores
from predictions import rows, save_predictions as _save_predictions
from runtime import input_files, preflight, sample_cpu_memory, sample_stage
from tasks import get_task, prepare_splits
from variants import variant_from_config


# Archived experiment scripts import these functions from the runner.
def prediction_scores(task, predictions):
    return _prediction_scores(get_task(task), predictions)


def save_predictions(task, output, name, test_rows, labels, predictions, indices=None, probabilities=None):
    return _save_predictions(get_task(task), output, name, test_rows, labels, predictions, indices, probabilities)


def predict_batches(protocol, model, data, profile=None, *, stop_after_capture=False):
    if isinstance(protocol, ModuleType):
        protocol = protocol.PROTOCOL
    return _predict_batches(protocol, model, data, profile, stop_after_capture=stop_after_capture)


def profile_reference(config, archive, assets, work, profile, metrics):
    import torch

    paths, checkpoint, metrics["reference_files"] = reference_run(config, archive)
    protocol = get_task(config["task"])
    data, options = None, None
    if protocol.HAS_TRAINING and "training" in config:
        data, options = protocol.prepare_training(config, assets, paths)
        protocol.train(config, assets, paths, data, options, work, profile, profile_only=True)
        gc.collect()
        torch.cuda.empty_cache()

    model, loader = protocol.setup_inference(config, assets, paths, data, options, checkpoint)
    batch = protocol.predict(model, loader, profile, stop_after_capture=True)
    metrics["sampled_inference"] = batch.sampled


def run(config, bundle, assets, output, metrics, reference=None):
    from contextlib import ExitStack

    import torch

    task = config["task"]
    protocol = get_task(task)
    profile = None
    if profiling_enabled(config):
        free, total = torch.cuda.mem_get_info()
        print(f"Profiler device memory before model setup: {free} free / {total} bytes", flush=True)
        if reference is None and not protocol.SUPPORTS_PROFILING:
            raise ValueError(f"{task} does not support profiler mode.")
        from profiling import BenchmarkProfiler

        profile = BenchmarkProfiler(torch, output, task)
    metrics["predictions"] = {}
    metrics["stages"] = {}
    metrics["tutorial_metrics"] = {}
    started = time.perf_counter()
    with ExitStack() as stack:
        work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="official-work-", dir=output.parent)))
        stack.enter_context(variant_from_config(config).activate())
        overlay = work / "assets"
        inputs = {}
        for relative, path in input_files(config, bundle, assets).items():
            link = overlay / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path)
            inputs[relative] = link

        print(f"Preparing official {task} splits", flush=True)
        split_start = time.perf_counter()
        if protocol.HAS_TRAINING and "training" in config:
            import lightning.pytorch as pl

            pl.seed_everything(config["training"]["seed"], workers=True)
        if config.get("profile_only"):
            profile_reference(config, reference, overlay, work, profile, metrics)
            metrics["profiling"] = profile.results
            metrics["run_wall_seconds"] = time.perf_counter() - started
            return
        reference_checkpoint = None
        if reference is not None:
            archived_splits, reference_checkpoint, metrics["reference_files"] = reference_run(config, reference)
            (output / "splits").mkdir()
            split_paths = {name: output / "splits" / path.name for name, path in archived_splits.items()}
            for name, path in archived_splits.items():
                shutil.copyfile(path, split_paths[name])
            if protocol.AUDIT_SPLITS:
                shutil.copyfile(reference / "split_manifest.json", output / "split_manifest.json")
        else:
            split_paths, audit = prepare_splits(config, inputs, output / "splits")
            if audit is not None:
                write_json(output / "split_manifest.json", audit)
        metrics["split_seconds"] = time.perf_counter() - split_start
        data, model_options = None, None
        checkpoint = None
        if protocol.HAS_TRAINING and "training" in config:
            data, model_options = protocol.prepare_training(config, overlay, split_paths)
            print(f"Training {task} with the recorded benchmark configuration", flush=True)
            checkpoints, metrics["stages"]["training"] = sample_stage(
                torch, lambda: protocol.train(config, overlay, split_paths, data, model_options, work, profile),
            )
            selected, final, metrics["checkpoint_selection"] = checkpoints
            destination = output / "checkpoints"
            destination.mkdir()
            if final is None:
                checkpoint = output / config["output_checkpoints"][0]
                shutil.copyfile(selected, checkpoint)
            else:
                checkpoint = destination / "tutorial-final.ckpt"
                shutil.copyfile(final, checkpoint)
                shutil.copyfile(selected, destination / "tutorial-best-validation.ckpt")
            gc.collect()
            torch.cuda.empty_cache()

        test_rows = rows(split_paths["test"])
        for stage_name, name in evaluation_stages(config).items():
            weights = checkpoint
            if stage_name == "fixed_checkpoint_inference":
                weights = reference_checkpoint
            elif stage_name == "official_checkpoint_inference":
                weights = inputs[protocol.OFFICIAL_CHECKPOINT]
            print(f"Evaluating {name}", flush=True)
            setup_start = time.perf_counter()
            model, loader = protocol.setup_inference(config, overlay, split_paths, data, model_options, weights)
            torch.cuda.synchronize()
            setup_seconds = time.perf_counter() - setup_start
            stage_profile = profile if stage_name == "inference" else None
            batch, stage = sample_stage(torch, lambda: protocol.predict(model, loader, stage_profile))
            metrics["predictions"][name] = protocol.write_predictions(output, name, test_rows, batch)
            secondary = protocol.secondary_scores(batch)
            if secondary is not None:
                metrics["tutorial_metrics"][name] = secondary
            stage.update(protocol.prediction_stage_metrics(batch, len(test_rows), stage["wall_seconds"]))
            stage["setup_seconds"] = setup_seconds
            stage["samples_per_second"] = len(test_rows) / stage["wall_seconds"]
            metrics["stages"][stage_name] = stage
            if stage_name == "inference" and config.get("interpretability"):
                print(f"Ranking regulators for {task}", flush=True)
                rankings, cost = sample_stage(
                    torch, lambda: extract_regulator_rankings(task, model, loader, test_rows, inputs[FACTOR_LIST], output),
                )
                metrics["interpretability"] = rankings
                metrics["stages"]["interpretability"] = cost
            del model, loader
            torch.cuda.empty_cache()
    metrics["run_wall_seconds"] = time.perf_counter() - started
    if profile:
        metrics["profiling"] = profile.results


def execute(config, bundle, assets, output, verify, reference=None):
    output.mkdir(parents=True, exist_ok=False)
    status = output / "status.json"
    write_json(status, {"state": "running"})
    try:
        if "training" in config and "seed" not in config["training"]:
            config = {**config, "training": {**config["training"], "seed": config["split"]["seed"]}}
        environment = preflight(config, bundle, assets)
        if reference is not None:
            reference_config = read_json(reference / "config.json")
            reference_inputs = reference_config["input_files"]
            for name, record in environment["input_files"].items():
                if any(value != reference_inputs[name].get(key) for key, value in record.items() if key != "path"):
                    raise ValueError(f"Reference run used different input contents or metadata: {name}")
            if variant_from_config(config).match_environment:
                for key in ("visible_gpu", "gpu", "torch", "cuda", "flash_attention", "python", "software"):
                    if environment[key] != reference_config[key]:
                        raise ValueError(f"Variant comparison used a different {key} from the reference.")
        write_json(output / "config.json", {**config, **environment})
        get_source(config).archive(config, bundle, output)
        metrics = {}
        with sample_cpu_memory() as peak:
            run(config, bundle, assets, output, metrics, reference)
        metrics["training_applicable"] = "training" in config
        metrics["peak_cpu_rss_bytes"] = peak[0]
        metrics["cpu_rss_sampling_interval_seconds"] = 0.2
        metrics["input_file_bytes"] = {name: record["bytes"] for name, record in environment["input_files"].items()}
        metrics["output_files"] = {
            path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(output.rglob("*")) if path.is_file() and path != status
        }
        write_json(output / "metrics.json", metrics)
        write_json(output / "check.json", verify(output))
    except BaseException as error:
        import traceback
        write_json(status, {"state": "failed", "error": str(error), "traceback": traceback.format_exc()})
        raise
    write_json(status, {"state": "complete", "verified": True})
    print(f"Results verified: {output / 'check.json'}", flush=True)
