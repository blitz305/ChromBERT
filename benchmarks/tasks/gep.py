"""Gene expression change prediction around the TSS."""

from random import Random

from training import dataset_config, lightning_data, load_inference_model
from ._splits import REGION_BED, proportional_counts
from .protocol import TaskProtocol

GEP_DIRECTORY = "demo/transdifferentiation/transcriptome"
GEP_FILES = ("fibroblast_expression.csv", "myoblast_expression.csv", "gep_fibroblast_to_myoblast.ckpt")
OFFICIAL_CHECKPOINT = f"{GEP_DIRECTORY}/{GEP_FILES[-1]}"
PREDICTION_IDS = ("gene_id", "build_region_index")
BATCH_INDEX = "center_build_region_index"
IS_REGRESSION = True


def sample_by_expression_change(frame, count, seed):
    labels = frame["label"]
    groups = (
        frame.index[labels < -1],
        frame.index[(labels >= -1) & (labels < 0)],
        frame.index[labels == 0],
        frame.index[(labels > 0) & (labels <= 1)],
        frame.index[labels > 1],
    )
    counts = {group: len(indices) for group, indices in enumerate(groups)}
    quotas = proportional_counts(counts, count)
    rng = Random(seed)
    selected = [index for group, indices in enumerate(groups)
                for index in rng.sample(indices.tolist(), quotas[group])]
    return frame.loc[frame.index.isin(selected)]


def prepare(config, inputs):
    settings, counts = config["split"], config["expected_counts"]
    import numpy as np
    import pandas as pd

    fibroblast = pd.read_csv(inputs[f"{GEP_DIRECTORY}/fibroblast_expression.csv"])
    myoblast = pd.read_csv(inputs[f"{GEP_DIRECTORY}/myoblast_expression.csv"])
    expression = fibroblast.merge(
        myoblast, on=["chrom", "tss", "gene_id"], suffixes=("_fibroblast", "_myoblast"),
    )
    expression["label"] = np.log1p(expression["tpm_myoblast"]) - np.log1p(expression["tpm_fibroblast"])
    expression["start"] = expression["tss"] // 1000 * 1000
    expression["end"] = expression["start"] + 1000
    coordinates = ["chrom", "start", "end"]
    region_index = pd.read_csv(inputs[REGION_BED], sep="\t", names=[*coordinates, "build_region_index"])
    columns = [*coordinates, "build_region_index", "label", "tss", "gene_id"]
    table = expression.merge(region_index, on=coordinates, how="inner")[columns]
    seed = settings["seed"]
    train = table.sample(frac=0.8, random_state=seed)
    leftover = table.drop(train.index)
    test = leftover.sample(frac=0.5, random_state=seed)
    valid = leftover.drop(test.index)
    splits = {"train": train, "valid": valid, "test": test}
    if settings.get("stratify_expression_change"):
        return {split: sample_by_expression_change(frame, counts[split], seed) for split, frame in splits.items()}
    if settings.get("downsample", True):
        return {split: frame.sample(n=counts[split], random_state=seed) for split, frame in splits.items()}
    return splits


def setup_training(config, assets, paths):
    data = lightning_data(dataset_config(config, assets), paths)
    data.setup()
    return data, config["model"].copy()


def setup_inference(config, assets, paths, data, model_options, checkpoint):
    model = load_inference_model(config, assets, model_options, checkpoint)
    return model, data.test_dataloader()


def tutorial_metrics(labels, predictions):
    import torch
    import torchmetrics as tm

    actual = torch.tensor(labels, dtype=torch.float64)
    predicted = torch.tensor(predictions, dtype=torch.float64)
    metrics = {"mse": tm.MeanSquaredError, "mae": tm.MeanAbsoluteError}
    scores = {"pearsonr": None, "spearmanr": None, "r2": None}
    if torch.any(actual != actual[0]):
        metrics["r2"] = tm.R2Score
        if torch.any(predicted != predicted[0]):
            metrics.update(pearsonr=tm.PearsonCorrCoef, spearmanr=tm.SpearmanCorrCoef)
    scores.update({name: float(metric().set_dtype(torch.float64)(predicted, actual))
                   for name, metric in metrics.items()})
    return scores


class GepProtocol(TaskProtocol):
    PREDICTION_IDS = PREDICTION_IDS
    BATCH_INDEX = BATCH_INDEX
    IS_REGRESSION = IS_REGRESSION
    OFFICIAL_CHECKPOINT = OFFICIAL_CHECKPOINT
    SECONDARY_SCORES = True

    prepare = staticmethod(prepare)
    setup_training = staticmethod(setup_training)
    setup_inference = staticmethod(setup_inference)

    def secondary_scores(self, batch):
        return tutorial_metrics(*batch.values[:2])


PROTOCOL = GepProtocol()
