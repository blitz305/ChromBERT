"""Paired STARR-seq activity predictions under EP300/CREBBP perturbation."""

import csv
import gzip
from bisect import bisect_left
from collections import defaultdict
from itertools import zip_longest

from predictions import rows
from training import excluded_track_options, lightning_data
from .protocol import PredictionBatch, TaskProtocol
from ._splits import REGION_BED, audit_frames, table_stats, validate_regions

STARR_TABLES = {name: f"demo/starr/{name}.csv" for name in ("train", "valid", "test", "supervised")}
REGULATORS = "config/hg38_6k_regulators_list.txt"
FACTORS = "config/hg38_6k_factors_list.txt"
STARR_CHAIN = "config/hg38ToHg19.over.chain.gz"
SUPPORTS_PROFILING = False
SET_PYTHON_PROTOBUF = True


def lift_midpoints(chain_path, midpoints):
    """Map requested hg38 coordinates using the official UCSC hg38 to hg19 chain."""
    by_chrom = defaultdict(list)
    result = {key: set() for key in midpoints}
    for chrom, point in sorted(midpoints):
        by_chrom[chrom].append(point)
    with gzip.open(str(chain_path), "rt", encoding="ascii") as chain:
        for line in chain:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "chain":
                _, _, chrom, _, strand, start, _, dst_chrom, dst_size, dst_strand, dst_start, *_ = fields
                if strand != "+":
                    raise ValueError("Unexpected reference strand in the UCSC chain.")
                points = by_chrom[chrom]
                start, dst_start, dst_size = int(start), int(dst_start), int(dst_size)
                continue
            size = int(fields[0])
            if points:
                for index in range(bisect_left(points, start), bisect_left(points, start + size)):
                    point = points[index]
                    mapped = dst_start + point - start
                    if dst_strand == "-":
                        mapped = dst_size - 1 - mapped
                    result[chrom, point].add((dst_chrom, mapped))
            if len(fields) == 3:
                start += size + int(fields[1])
                dst_start += size + int(fields[2])
    return result


def prepare(config, inputs):
    import numpy as np
    import pandas as pd

    region_column = "build_region_index"
    raw = {name: pd.read_csv(inputs[path]) for name, path in STARR_TABLES.items()}
    test = raw["supervised"]
    if len(raw["train"]) != 34818 or len(test) != 721 or test[region_column].duplicated().any():
        raise ValueError("Unexpected STARR tutorial training or paired-evaluation size.")
    if not np.isfinite(test[["wt", "ep300", "diff"]].to_numpy()).all() or not np.allclose(
        test["ep300"] - test["wt"], test["diff"], rtol=1e-8, atol=1e-10,
    ):
        raise ValueError("STARR measured effects do not match the paired signals.")
    validate_regions(inputs[REGION_BED], raw.values(), chain_path=inputs[STARR_CHAIN], lift=lift_midpoints)
    for name in ("train", "valid", "test"):
        if not np.isfinite(raw[name]["label"]).all():
            raise ValueError(f"STARR {name} has non-finite activity labels.")
    valid = raw["valid"].loc[~raw["valid"][region_column].isin(test[region_column])]
    held_out = set(test[region_column]) | set(valid[region_column])
    train = raw["train"].loc[~raw["train"][region_column].isin(held_out)]
    frames = {"train": train, "valid": valid, "test": test}

    return frames, {
        "raw": {name: table_stats(frame) for name, frame in raw.items()},
        "removed_for_disjoint_splits": {name: len(raw[name]) - len(frames[name]) for name in ("train", "valid")},
        "raw_pair_overlap": {
            name: len(set(raw[name][region_column]) & set(test[region_column]))
            for name in ("train", "valid", "test")
        },
    }


def split_audit(config, frames):
    return audit_frames(frames, ("build_region_index",))


def smoke_sample(frame, seed):
    return frame.sample(16, random_state=seed)


def starr_dataset(assets, supervised_file, **options):
    import chrombert

    known = set((assets / REGULATORS).read_text(encoding="utf-8").splitlines())
    factors = set((assets / FACTORS).read_text(encoding="utf-8").splitlines())
    ignored = sorted({name.strip().lower() for name in known if name.strip()}
                     - {name.strip().lower() for name in factors if name.strip()})
    if not ignored:
        raise ValueError("STARR non-TF track list is empty.")
    return chrombert.get_preset_dataset_config(
        "general", basedir=str(assets), supervised_file=supervised_file,
        ignore=True, ignore_object=";".join(ignored), **options,
    )


def setup_training(config, assets, paths):
    dataset = starr_dataset(assets, None, **config["dataset"])
    options = excluded_track_options(config, dataset, paths["train"])
    return lightning_data(dataset, paths, include_test=False), options


def setup_inference(config, assets, paths, data, model_options, checkpoint):
    import chrombert

    model = chrombert.get_preset_model_config(
        "general", basedir=str(assets), dropout=0, finetune_ckpt=str(checkpoint), **model_options,
    ).init_model().cuda().bfloat16().eval()
    dataset = starr_dataset(
        assets, str(paths["test"]), shuffle=False, batch_size=config["inference_batch_size"],
        num_workers=config["dataset"]["num_workers"], perturbation_value=0,
    )
    return model, (
        dataset.init_dataloader(perturbation=False, perturbation_object="none"),
        dataset.init_dataloader(perturbation=True, perturbation_object="crebbp;ep300"),
    )


def predict_pairs(model, loaders, profile=None, *, stop_after_capture=False):
    from contextlib import closing, nullcontext

    import torch
    from tqdm import tqdm

    predictions = []
    changed_pairs = 0
    batch_count = len(loaders[0])
    batches = tqdm(zip_longest(*loaders), total=batch_count, desc="Predict paired STARR")
    capture = closing(profile.inference_batches(model, batches, batch_count,
                                                stop_after_capture=stop_after_capture)) if profile else nullcontext(batches)
    with torch.no_grad(), capture as batches:
        for wt, perturbed in batches:
            if wt is None or perturbed is None or not (
                torch.equal(wt["build_region_index"], perturbed["build_region_index"])
                and torch.equal(wt["position_ids"], perturbed["position_ids"])
            ):
                raise ValueError("STARR inference batch region IDs do not match.")
            changed = wt["input_ids"] != perturbed["input_ids"]
            changed_pairs += int(changed.any(dim=1).sum())
            values = []
            for stage, batch in (("wild_type", wt), ("perturbed", perturbed)):
                with profile.annotate(stage) if profile else nullcontext():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    values.append(model(batch).float().reshape(-1).cpu().tolist())
            predictions.extend(zip(wt["build_region_index"].tolist(), *values))
    if not changed_pairs:
        raise ValueError("EP300/CREBBP perturbation did not change any STARR test inputs.")
    return predictions, changed_pairs


def save_predictions(path, targets, predictions):
    columns = ["build_region_index", "wt", "ep300", "diff", "prediction_wt", "prediction_ep300", "prediction_diff"]
    if len(predictions) != len(targets):
        raise ValueError("STARR did not predict every paired sample.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for target, (region, wt, ep300) in zip(targets, predictions):
            if int(target["build_region_index"]) != region:
                raise ValueError("STARR predictions do not match the saved paired test split.")
            writer.writerow([target[key] for key in columns[:4]] + [wt, ep300, ep300 - wt])


def score_predictions(predictions, targets):
    import numpy as np
    from scipy.stats import pearsonr

    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Incorrect number of STARR predictions.")
    for prediction, target in zip(predictions, targets):
        if any(prediction[key] != target[key] for key in ("build_region_index", "wt", "ep300", "diff")):
            raise ValueError("A STARR prediction row differs from its paired target.")

    def array(key):
        return np.asarray([float(row[key]) for row in predictions])

    wt, ep300, observed = (array(name) for name in ("wt", "ep300", "diff"))
    pred_wt, pred_ep300, effect = (array(name) for name in ("prediction_wt", "prediction_ep300", "prediction_diff"))
    if not all(np.isfinite(a).all() for a in (wt, ep300, observed, pred_wt, pred_ep300, effect)):
        raise ValueError("STARR predictions or targets contain non-finite values.")
    if not np.allclose(observed, ep300 - wt, rtol=1e-8, atol=1e-10) or not np.array_equal(effect, pred_ep300 - pred_wt):
        raise ValueError("STARR saved differences do not match their paired values.")

    def pearson(actual, predicted):
        if len(actual) >= 2 and np.any(actual != actual[0]) and np.any(predicted != predicted[0]):
            return float(pearsonr(actual, predicted)[0])
        return None

    return {
        "samples": len(predictions), "wt_pearson": pearson(wt, pred_wt),
        "perturbed_pearson": pearson(ep300, pred_ep300),
        "effect_pearson": pearson(observed, effect), "effect_rmse": float(np.sqrt(np.mean((observed - effect) ** 2))),
        "wt_prediction_variance": float(np.var(pred_wt)), "perturbed_prediction_variance": float(np.var(pred_ep300)),
        "effect_prediction_variance": float(np.var(effect)),
    }


class StarrProtocol(TaskProtocol):
    SUPPORTS_PROFILING = SUPPORTS_PROFILING
    SET_PYTHON_PROTOBUF = SET_PYTHON_PROTOBUF
    AUDIT_SPLITS = True
    SUPPORTS_SMOKE = True

    prepare = staticmethod(prepare)
    split_audit = staticmethod(split_audit)
    smoke_sample = staticmethod(smoke_sample)
    setup_training = staticmethod(setup_training)
    setup_inference = staticmethod(setup_inference)
    score_predictions = staticmethod(score_predictions)

    def predict(self, model, loader, profile=None, *, stop_after_capture=False):
        predictions, changed_pairs = predict_pairs(model, loader, profile, stop_after_capture=stop_after_capture)
        return PredictionBatch(predictions, {"pairs": len(predictions), "changed_pairs": changed_pairs})

    def write_predictions(self, output, name, targets, batch):
        path = output / f"{name}.csv"
        save_predictions(path, targets, batch.values)
        return self.score_predictions(rows(path), targets)

    def prediction_stage_metrics(self, batch, target_count, wall_seconds):
        return {
            "forward_samples_per_second": 2 * target_count / wall_seconds,
            "input_changed_pairs": batch.sampled["changed_pairs"],
        }


PROTOCOL = StarrProtocol()
