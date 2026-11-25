"""Validate saved benchmark results and resolve reference runs."""

import re

from benchmark_io import profiling_enabled, read_json, sha256
from interpretability import MEAN_EMBEDDINGS, RANKINGS, verify_regulator_rankings
from model_sources import get_source
from predictions import rows, scores_match
from tasks import get_task
from variants import variant_from_config


def evaluation_stages(config):
    stages = {}
    if variant_from_config(config).uses_fixed_checkpoint_evaluation(config) and "training" in config:
        stages["fixed_checkpoint_inference"] = "fixed_checkpoint_predictions"
    stages["inference"] = "test_predictions"
    official_checkpoint = get_task(config["task"]).OFFICIAL_CHECKPOINT
    if official_checkpoint in config["inputs"]:
        stages["official_checkpoint_inference"] = "official_checkpoint_predictions"
    return stages


def validate_reference_status(archive, splits_only):
    status = read_json(archive / "status.json")
    verified = status.get("state") == "complete" and status.get("verified") is True
    recorded_splits = splits_only and status.get("state") == "failed" and (archive / "metrics.json").is_file()
    if not (verified or recorded_splits):
        raise ValueError("Use a full benchmark archive with verified results or checked splits for training.")


def resolve_reference_for_launch(reference, mode, variant):
    if reference is None:
        return None
    reference = reference.resolve()
    if (reference / "result/config.json").is_file():
        reference = reference / "result"
    validate_reference_status(reference, variant.uses_splits_only(variant.record(mode)))
    config = read_json(reference / "config.json")
    if variant.requires_baseline and "training" in config and "seed" not in config["training"]:
        log = reference.parent / "run.log" if reference.name == "result" else reference / "run.log"
        seed = config["split"]["seed"]
        with log.open(encoding="utf-8", errors="replace") as stream:
            if not any(line.strip() == f"Global seed set to {seed}" for line in stream):
                raise ValueError(f"Reference run did not record training seed {seed}: {reference}")
    return reference


def reference_run(config, archive):
    original = read_json(archive / "config.json")
    variant = variant_from_config(config)
    splits_only = variant.uses_splits_only(config)
    validate_reference_status(archive, splits_only)
    if original.get("smoke"):
        raise ValueError("Use a full benchmark archive as the reference.")
    if variant.requires_baseline and (variant_from_config(original).name != "original"
                                      or profiling_enabled(original) or original.get("profile_only")):
        raise ValueError(f"The {variant.name} reference must use the original variant with profiling disabled.")
    source = get_source(config)
    if source is not get_source(original):
        raise ValueError("Reference run has a different model source.")
    matches = (*source.reference_fields, "task", "expected_counts", "split", "dataset_preset", "dataset",
               "model_preset", "model", "inference_batch_size", "output_checkpoints")
    for key in matches:
        if original.get(key) != config.get(key):
            raise ValueError(f"Reference run has different {key}.")
    if "training" in config:
        archived_training = {"seed": original["split"]["seed"], **original["training"]}
        if config["training"] != archived_training:
            raise ValueError("Reference run has different training settings.")

    recorded = read_json(archive / "metrics.json")["output_files"]

    def checked(relative):
        path = archive / relative
        if sha256(path) != recorded[relative]["sha256"]:
            raise ValueError(f"Reference file checksum differs: {relative}")
        return path

    paths = {name: checked(f"splits/{name}.csv") for name in config["expected_counts"]}
    checkpoints = config["output_checkpoints"]
    checkpoint = checked(checkpoints[0]) if checkpoints and not splits_only else None
    files = {f"splits/{name}.csv": recorded[f"splits/{name}.csv"] for name in paths}
    if checkpoint is not None:
        files[checkpoints[0]] = recorded[checkpoints[0]]
    return paths, checkpoint, files


def verify_results(output, config, metrics):
    task = config["task"]
    protocol = get_task(task)
    counts = {split: len(rows(output / "splits" / f"{split}.csv")) for split in config["expected_counts"]}
    expected_counts = config["expected_counts"]
    if config.get("smoke"):
        expected_counts = dict.fromkeys(expected_counts, 16)
    if any(n <= 0 or (expected_counts[split] is not None and n != expected_counts[split])
           for split, n in counts.items()):
        raise ValueError(f"Sample counts differ from the benchmark configuration: {counts}")
    if protocol.AUDIT_SPLITS:
        import pandas as pd

        manifest = read_json(output / "split_manifest.json")
        frames = {name: pd.read_csv(output / "splits" / f"{name}.csv") for name in counts}
        verified = protocol.split_audit(config, frames)
        if any(manifest[key] != value for key, value in verified.items()):
            raise ValueError("Saved splits differ from the split manifest.")
    test = rows(output / "splits/test.csv")
    evaluations = evaluation_stages(config)
    expected = set(evaluations.values())
    if set(metrics["predictions"]) != expected:
        raise ValueError(f"Missing or unexpected prediction branches: {sorted(metrics['predictions'])}")
    summary = {}
    for name in evaluations.values():
        predictions = rows(output / f"{name}.csv")
        if len(predictions) != len(test):
            raise ValueError(f"{name}: incorrect prediction count.")
        recomputed = protocol.score_predictions(predictions, test)
        protocol.verify_scores(config, recomputed)
        recorded = metrics["predictions"][name]
        if recomputed.keys() != recorded.keys() or any(
            not scores_match(value, recorded[key], rel_tol=1e-7, abs_tol=1e-9) for key, value in recomputed.items()
        ):
            raise ValueError(f"{name}: recomputed scores differ.")
        summary[name] = recomputed
    if protocol.SECONDARY_SCORES:
        tutorial_metrics = metrics["tutorial_metrics"]
        if set(tutorial_metrics) != expected:
            raise ValueError("TorchMetrics results are missing for a GEP checkpoint.")
        for name, recorded in tutorial_metrics.items():
            score_names = summary[name].keys() - {"samples"}
            if set(recorded) != score_names or any(
                not scores_match(recorded[key], summary[name][key], rel_tol=1e-4, abs_tol=1e-6)
                for key in score_names
            ):
                raise ValueError(f"{name}: TorchMetrics and independent recomputation differ.")
    if metrics["run_wall_seconds"] <= 0 or metrics["peak_cpu_rss_bytes"] <= 0:
        raise ValueError("Resource measurements are missing.")
    required_stages = set(evaluations)
    if "training" in config:
        required_stages.add("training")
    result = {"task": task, "counts": counts, "predictions": summary, "predictions_matched_to_split": True}
    if protocol.AUDIT_SPLITS:
        result["split_audit"] = manifest
    if config.get("interpretability"):
        required_stages.add("interpretability")
        result["interpretability"] = verify_regulator_rankings(task, test, output, metrics["interpretability"])
    if set(metrics["stages"]) != required_stages:
        raise ValueError("The required benchmark stages were not recorded.")
    for name, stage in metrics["stages"].items():
        if stage["wall_seconds"] <= 0 or not 0 < stage["torch_peak_allocated_bytes"] <= stage["torch_peak_reserved_bytes"]:
            raise ValueError(f"Invalid GPU resource measurements for {name}.")
    return result


def verify_result(output):
    config = read_json(output / "config.json")
    metrics = read_json(output / "metrics.json")
    protocol = get_task(config["task"])
    if (config["task"] == "gep" and "variant" not in config and "training_split_sha256" in config
            and "environment" in config and "output_files" not in metrics):
        return verify_legacy_gep_profile(output, config, metrics)
    source = get_source(config)
    source.verify(config, output)
    required = {"config.json", *source.artifacts(config)}
    profile_only = config.get("profile_only", False)
    if not profile_only:
        required.update(f"splits/{split}.csv" for split in config["expected_counts"])
        required.update(f"{name}.csv" for name in metrics["predictions"])
        required.update(config["output_checkpoints"])
        if protocol.AUDIT_SPLITS:
            required.add("split_manifest.json")
        if config.get("interpretability"):
            required.update((RANKINGS, MEAN_EMBEDDINGS))
    if profile_only or profiling_enabled(config):
        stages = {"inference"}
        if "training" in config and (profile_only or protocol.PROFILE_TRAINING):
            stages.add("training")
        profiles = metrics.get("profiling", {})
        if set(profiles) != stages:
            raise ValueError(f"Expected profiling stages: {', '.join(sorted(stages))}.")
        for stage in stages:
            if profiles[stage]["cuda_events"] <= 0:
                raise ValueError(f"The {stage} profile contains no CUDA events.")
            required.update((f"{stage}_trace.json", f"{stage}_operators.txt"))
    records = metrics["output_files"]
    if not required <= records.keys():
        raise ValueError(f"Missing recorded outputs: {sorted(required - records.keys())}")
    for filename, record in records.items():
        path = output / filename
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Output does not match the recorded size or checksum: {filename}")
    if profile_only:
        return {
            "task": config["task"], "profile_only": True,
            "reference_files": metrics["reference_files"], "sampled_inference": metrics["sampled_inference"],
            "cuda_events": {stage: profile["cuda_events"] for stage, profile in profiles.items()},
        }
    return verify_results(output, config, metrics)


def verify_legacy_gep_profile(output, config, metrics):
    """Recheck the traces and recorded prediction check for the fixed-weight GEP archive."""
    validate_reference_status(output, splits_only=False)
    check = read_json(output / "check.json")
    digests = [config["checkpoint_sha256"], config["sample_test_sha256"],
               *config["training_split_sha256"].values()]
    if set(config["training_split_sha256"]) != {"train", "valid"} or not all(
        re.fullmatch(r"[0-9a-f]{64}", digest) for digest in digests
    ):
        raise ValueError("Invalid legacy GEP checkpoint or split digests.")
    stages = {"training", "inference"}
    expected_traces = {f"{stage}_trace.json" for stage in stages}
    profiles = metrics["profiling"]
    if not check.get("profile_only") or not check["matched_predictions"] > 0 or not (
        0 <= check["max_abs_difference_from_archive"] < 1e-5
    ):
        raise ValueError("The legacy GEP archive did not record a successful prediction check.")
    if set(profiles) != stages or any(profiles[name]["cuda_events"] <= 0 for name in stages):
        raise ValueError("The legacy GEP archive is missing CUDA profile events.")
    if set(check["trace_sha256"]) != expected_traces or any(
        sha256(output / name) != check["trace_sha256"][name] for name in expected_traces
    ):
        raise ValueError("Legacy GEP trace checksum mismatch.")
    return check
