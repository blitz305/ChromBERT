"""Single-output inference, prediction files, and metric calculations."""

import csv
import math


def label(value):
    return float({"true": 1.0, "false": 0.0}.get(value.casefold(), value))


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def scores_match(actual, recorded, *, rel_tol, abs_tol):
    if actual is None or recorded is None:
        return actual is None and recorded is None
    return math.isclose(actual, recorded, rel_tol=rel_tol, abs_tol=abs_tol)


def prediction_scores(protocol, predictions):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    actual = np.asarray([label(row["label"]) for row in predictions])
    predicted = np.asarray([float(row["prediction"]) for row in predictions])
    if not actual.size or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Predictions and labels must be finite and nonempty.")
    if protocol.IS_REGRESSION:
        error = predicted - actual
        mse = float(np.mean(error ** 2))
        varying_labels = np.any(actual != actual[0])
        defined_correlation = varying_labels and np.any(predicted != predicted[0])
        return {
            "samples": len(actual),
            "pearsonr": float(pearsonr(actual, predicted)[0]) if defined_correlation else None,
            "spearmanr": float(spearmanr(actual, predicted)[0]) if defined_correlation else None,
            "mse": mse,
            "mae": float(np.mean(np.abs(error))),
            "r2": float(1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2)) if varying_labels else None,
        }
    if set(actual) != {0.0, 1.0}:
        raise ValueError("The official classification benchmark must contain both labels.")
    ranking = predicted
    if "probability" in predictions[0]:
        ranking = np.asarray([float(row["probability"]) for row in predictions])
        if not np.isfinite(ranking).all() or np.any((ranking < 0) | (ranking > 1)):
            raise ValueError("Notebook probabilities must be between zero and one.")
    return {
        "samples": len(actual),
        "positives": int(actual.sum()),
        "auroc": float(roc_auc_score(actual, ranking)),
        "auprc": float(average_precision_score(actual, ranking)),
        "bce": float(np.mean(np.logaddexp(0, predicted) - actual * predicted)),
    }


def save_predictions(protocol, output, name, test_rows, labels, predictions, indices=None, probabilities=None):
    if len(test_rows) != len(labels) or len(labels) != len(predictions):
        raise ValueError(f"{name}: prediction and test split counts differ.")
    if (indices is not None and len(indices) != len(labels)) or (
        probabilities is not None and len(probabilities) != len(labels)
    ):
        raise ValueError(f"{name}: prediction identifiers or probabilities are incomplete.")
    path = output / f"{name}.csv"
    identity = protocol.PREDICTION_IDS
    columns = [*identity, "label", "prediction"]
    if probabilities is not None:
        columns.append("probability")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for position, (row, observed, prediction) in enumerate(zip(test_rows, labels, predictions)):
            if not math.isclose(label(row["label"]), float(observed), rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f"{name}: test label differs at row {position + 1}.")
            if indices is not None and int(row["build_region_index"]) != int(indices[position]):
                raise ValueError(f"{name}: test region differs at row {position + 1}.")
            value = {key: row[key] for key in identity}
            value.update(label=observed, prediction=prediction)
            if probabilities is not None:
                value["probability"] = probabilities[position]
            writer.writerow(value)
    return prediction_scores(protocol, rows(path))


def predict_batches(protocol, model, data, profile=None, *, stop_after_capture=False):
    from contextlib import closing, nullcontext

    import torch
    from tqdm import tqdm

    if not isinstance(data.sampler, torch.utils.data.SequentialSampler):
        raise ValueError("The test loader must preserve the tutorial split order.")
    labels, predictions, indices = [], [], []
    index_key = protocol.BATCH_INDEX
    batches = tqdm(data, desc="Predict")
    capture = closing(profile.inference_batches(model, batches, len(data),
                                                stop_after_capture=stop_after_capture)) if profile else nullcontext(batches)
    with torch.no_grad(), capture as batches:
        for incoming in batches:
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in incoming.items()}
            predictions.extend(model(batch).float().reshape(-1).cpu().tolist())
            # Copy label values before the DataLoader reclaims the batch's shared storage.
            labels.extend(incoming["label"].reshape(-1).tolist())
            indices.extend(incoming[index_key].reshape(-1).tolist())
    probabilities = None
    if not protocol.IS_REGRESSION:
        probabilities = torch.tensor(predictions, dtype=torch.float32).sigmoid().tolist()
    return labels, predictions, indices, probabilities
