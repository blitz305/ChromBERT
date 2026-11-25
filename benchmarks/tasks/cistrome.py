"""Cistrome imputation using a regulator and cell type prompt."""

import csv
import random
from array import array
from collections import defaultdict

from training import dataset_config, load_inference_model
from ._splits import REGION_BED, proportional_counts
from .protocol import TaskProtocol

CISTROME_PEAKS = "demo/prompt_imputation/MAZ_K562_narrowPeak.bed"
USES_BEDTOOLS = True
SAMPLE_SIZE = 250_000
SAMPLE_SEED = 42
SAMPLE_FILENAME = "MAZ_K562_sample250k_ids.csv"


def create_sample(full_split, output):
    """Write the fixed MAZ × K562 sample from the full labeled split."""
    groups = defaultdict(lambda: array("I"))
    with full_split.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows, [])
        chrom_col, index_col, label_col = (header.index(key) for key in ("chrom", "build_region_index", "label"))
        previous = -1
        for row in rows:
            region = int(row[index_col])
            if region <= previous or row[label_col] not in {"True", "False"}:
                raise ValueError("The full split must have ordered, unique region IDs and Boolean labels.")
            groups[(row[chrom_col], row[label_col] == "True")].append(region)
            previous = region
    chromosomes = sorted({chrom for chrom, _ in groups})
    totals = {label: sum(len(ids) for (_, observed), ids in groups.items() if observed == label)
              for label in (False, True)}
    if totals != {False: 2_074_276, True: 63_618} or len(chromosomes) != 24:
        raise ValueError("The full MAZ × K562 test split does not match the expected population.")

    target = proportional_counts(totals, SAMPLE_SIZE)
    counts = {}
    for label in (False, True):
        counts[label] = proportional_counts(
            {chrom: len(groups[(chrom, label)]) for chrom in chromosomes}, target[label],
        )
    if len(groups[("chrY", True)]) != 1 or counts[True]["chrY"] != 0 or counts[True]["chr1"] < 2:
        raise ValueError("Unexpected chrY positive count or proportional allocation.")
    counts[True]["chrY"] = 1
    counts[True]["chr1"] -= 1

    rng = random.Random(SAMPLE_SEED)
    picked = []
    for chrom in chromosomes:
        for label in (False, True):
            population = groups[(chrom, label)]
            offsets = rng.sample(range(len(population)), counts[label][chrom])
            picked.extend((population[offset], label) for offset in offsets)
    picked.sort()
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("build_region_index", "label"))
        writer.writerows(picked)
    return {
        "seed": SAMPLE_SEED,
        "method": "Proportional largest-remainder quotas within each label, then Python random.Random.sample in sorted chromosome order, negatives before positives.",
        "adjustment": "Keep the sole chrY positive and transfer one positive quota from chr1.",
        "samples": len(picked),
        "positives": target[True],
        "negatives": target[False],
        "by_chromosome": {chrom: {"positives": counts[True][chrom], "negatives": counts[False][chrom]}
                          for chrom in chromosomes},
    }


def select_fixed_sample(table, path):
    import pandas as pd

    fixed = pd.read_csv(path, dtype={"build_region_index": "int64", "label": "bool"})
    ids = fixed["build_region_index"]
    if not ids.is_unique or not ids.is_monotonic_increasing:
        raise ValueError("The fixed Cistrome test IDs must be unique and sorted.")
    selected = table.set_index("build_region_index").loc[ids].reset_index()
    if len(selected) != len(fixed) or not (selected["label"].to_numpy() == fixed["label"].to_numpy()).all():
        raise ValueError("The fixed Cistrome labels differ from the full split.")
    return selected[table.columns]


def prepare(config, inputs):
    settings, counts = config["split"], config["expected_counts"]
    from chrombert.scripts.chrombert_make_dataset import get_overlap

    table = get_overlap(inputs[CISTROME_PEAKS], inputs[REGION_BED], no_filter=True)
    table["label"] = table["label"] > 0
    if "sample_indices" in settings:
        test = select_fixed_sample(table, inputs[settings["sample_indices"]])
        verify_scores(config, {"positives": int(test["label"].sum())})
        return {"test": test}
    if not settings.get("downsample", True):
        return {"test": table}
    selected = table[table["chrom"] == settings["chromosome"]]
    test = selected.groupby("label").sample(counts["test"] // 2, random_state=settings["seed"])
    return {"test": test}


def setup_inference(config, assets, paths, data, model_options, checkpoint):
    model = load_inference_model(config, assets, model_options, checkpoint)
    dataset = dataset_config(config, assets, paths["test"])
    return model.bfloat16().eval(), dataset.init_dataloader()


def verify_scores(config, scores):
    settings = config["split"]
    if "sample_indices" in settings:
        if scores["positives"] != settings["expected_positives"]:
            raise ValueError("The fixed Cistrome sample has an unexpected number of positives.")
    elif settings.get("downsample", True) and scores["positives"] * 2 != scores["samples"]:
        raise ValueError("The sampled Cistrome tutorial must contain equal numbers of positive and negative bins.")


class CistromeProtocol(TaskProtocol):
    USES_BEDTOOLS = USES_BEDTOOLS
    HAS_TRAINING = False
    PROFILE_TRAINING = False

    prepare = staticmethod(prepare)
    setup_inference = staticmethod(setup_inference)
    verify_scores = staticmethod(verify_scores)


PROTOCOL = CistromeProtocol()
