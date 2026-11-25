"""Rank regulators from ChromBERT's grouped test-set embeddings."""

FACTOR_LIST = "config/hg38_6k_factors_list.txt"
RANKINGS = "regulator_rankings.csv"
MEAN_EMBEDDINGS = "regulator_means.npz"


def regulator_groups(task, targets):
    import numpy as np

    labels = np.asarray([
        float({"true": "1", "false": "0"}.get(row["label"].casefold(), row["label"]))
        for row in targets
    ])
    if task == "ezh2":
        groups = {"classical": labels == 1, "nonclassical": labels == 0}
    elif task == "gep":
        groups = {"upregulated": labels > 1, "unchanged": np.abs(labels) < 0.5}
    else:
        raise ValueError(f"{task} has no regulator-ranking protocol.")
    if any(not mask.any() for mask in groups.values()):
        raise ValueError(f"{task} interpretability requires both test-set groups.")
    return groups


def rank_regulators(task, names, means):
    import numpy as np
    import pandas as pd

    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Regulator names must be unique and nonempty.")
    normalized = {}
    for group, values in means.items():
        if values.ndim != 2 or len(values) != len(names) or not np.isfinite(values).all():
            raise ValueError(f"Invalid mean regulator embeddings for {group}.")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        normalized[group] = values / np.where(norms == 0, 1, norms)
    result = pd.DataFrame({"regulator": names})
    if task == "gep":
        similarity = np.sum(normalized["upregulated"] * normalized["unchanged"], axis=1)
        result["similarity"] = np.clip(similarity, -1, 1)
        result = result.sort_values(["similarity", "regulator"]).reset_index(drop=True)
        result["rank"] = result.index + 1
        return result

    if "ezh2" not in names:
        raise ValueError("EZH2 is missing from the retained regulator embeddings.")
    ezh2_index = list(names).index("ezh2")
    thresholds = {}
    for group, vectors in normalized.items():
        # The Notebook sets the threshold over all pairwise similarities, including the diagonal.
        matrix = np.clip(vectors @ vectors.T, -1, 1)
        thresholds[group] = np.percentile(matrix, 95)
        result[group + "_similarity"] = matrix[ezh2_index]
    result["similarity_delta"] = result["classical_similarity"] - result["nonclassical_similarity"]
    for group, ascending in (("classical", False), ("nonclassical", True)):
        candidates = result.loc[
            (result["regulator"] != "ezh2") & (result[group + "_similarity"] > thresholds[group])
        ].sort_values(["similarity_delta", "regulator"], ascending=[ascending, True])
        result[group + "_rank"] = pd.Series(range(1, len(candidates) + 1), index=candidates.index, dtype="Int64")
    return result.sort_values("regulator").reset_index(drop=True)


def extract_regulator_rankings(task, model, loader, targets, factors_path, output):
    import numpy as np
    import torch
    from tqdm import tqdm

    groups = regulator_groups(task, targets)
    counts = {group: int(mask.sum()) for group, mask in groups.items()}
    factors = {line.strip().lower() for line in factors_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    encoder = model.eval().get_embedding_manager().cuda().eval()
    names = np.asarray([name.lower() for name in encoder.list_regulator])
    kept = np.isin(names, list(factors))
    names = names[kept]
    index_key = "center_build_region_index" if task == "gep" else "build_region_index"
    expected_indices = [int(row["build_region_index"]) for row in targets]
    sums = {group: 0 for group in groups}
    position = 0
    with torch.no_grad():
        for incoming in tqdm(loader, desc="Regulator embeddings"):
            indices = incoming[index_key].reshape(-1).tolist()
            end = position + len(indices)
            if indices != expected_indices[position:end]:
                raise ValueError("Interpretability loader differs from the benchmark test split.")
            masks = {group: mask[position:end] for group, mask in groups.items()}
            position = end
            if not any(mask.any() for mask in masks.values()):
                continue
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in incoming.items()}
            embeddings = encoder(batch).float().cpu().numpy()[:, kept]
            for group, mask in masks.items():
                sums[group] += embeddings[mask].sum(axis=0, dtype=np.float64)
    if position != len(targets):
        raise ValueError("Interpretability loader returned an incomplete test split.")
    means = {group: totals / counts[group] for group, totals in sums.items()}
    rankings = rank_regulators(task, names, means)
    rankings.to_csv(output / RANKINGS, index=False)
    np.savez_compressed(output / MEAN_EMBEDDINGS, regulators=names, **means)
    return {"group_counts": counts, "regulators": len(names)}


def verify_regulator_rankings(task, targets, output, recorded):
    import numpy as np
    import pandas as pd

    groups = regulator_groups(task, targets)
    with np.load(output / MEAN_EMBEDDINGS, allow_pickle=False) as archive:
        names = archive["regulators"]
        expected = rank_regulators(task, names, {group: archive[group] for group in groups})
    observed = pd.read_csv(output / RANKINGS, keep_default_na=False, na_values=[""])
    try:
        pd.testing.assert_frame_equal(observed, expected, check_dtype=False, rtol=1e-9, atol=1e-12)
    except AssertionError as error:
        raise ValueError("Regulator ranks differ from the archived mean embeddings.") from error
    counts = {group: int(mask.sum()) for group, mask in groups.items()}
    if recorded != {"group_counts": counts, "regulators": len(names)}:
        raise ValueError("Interpretability counts differ from the archived test split or embeddings.")
    return recorded
