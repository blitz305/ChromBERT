"""Lung eQTL classification with the official DNA prompt model."""

import math
import os

from ._splits import REGION_BED, audit_frames, check_regions, table_stats
from .protocol import TaskProtocol

EQTL_TABLE = "demo/eqtl/lung_eqtl.tsv"
FASTA = "other/hg38.fa"
DNA_MODEL = "checkpoint/dnabert2-117m"
REQUIRED_MODULES = ("pyfaidx", "transformers")
PREDICTION_IDS = ("variant_id", "build_region_index")
PROFILE_TRAINING = False


def audit_fasta(frame, path):
    from pyfaidx import Fasta

    with Fasta(str(path), rebuild=False, build_index=False) as fasta:
        for row in frame.itertuples(index=False):
            start = row.pos - 1 - 500
            seq = fasta[row.chrom][start:start + 1000].seq.upper() if start >= 0 else ""
            ref_matches = seq[500] == row.base_ref if len(seq) == 1000 else False
            if len(seq) != 1000 or not (ref_matches or (row.base_ref == "N" and row.label == 0)):
                raise ValueError(f"FASTA reference/window differs from the tutorial input at {row.variant_id}.")


def prepare(config, inputs):
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    task = config["task"]
    seed = config["split"]["seed"]
    region_column = "build_region_index"
    table = pd.read_csv(inputs[EQTL_TABLE], sep="\t", keep_default_na=False)
    if len(table) != 1535 or set(table["label"]) != {0, 1} or (table["variant_id"] == "").any():
        raise ValueError("Unexpected lung eQTL data size, labels or variant IDs.")
    if table.groupby("variant_id")[region_column].nunique().max() != 1:
        raise ValueError("A variant ID is assigned to multiple genomic bins.")
    check_regions(inputs[REGION_BED], table)
    audit_fasta(table, inputs[FASTA])
    raw = {"full": table}
    if task == "eqtl-official":
        train, test = train_test_split(table, test_size=0.4, random_state=seed)
        frames = {"train": train, "test": test}
    else:
        def split(frame, folds):
            indices = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            left, right = next(indices.split(frame, frame["label"], frame[region_column]))
            return frame.iloc[left], frame.iloc[right]

        remaining, test = split(table, 5)
        train, valid = split(remaining, 4)
        frames = {"train": train, "valid": valid, "test": test}
    return frames, {"raw": {"full": table_stats(table)}}


def split_audit(config, frames):
    return audit_frames(frames, PREDICTION_IDS, allow_overlap=config["task"] == "eqtl-official", binary=True)


def smoke_sample(frame, seed):
    return frame.groupby("label", group_keys=False).sample(8, random_state=seed)


def eqtl_dataset(config, assets, path, shuffle=False):
    import chrombert

    return chrombert.get_preset_dataset_config(
        "prompt_dna", basedir=str(assets), supervised_file=str(path),
        shuffle=shuffle, **config["dataset"],
    ).init_dataloader()


def fit(config, assets, paths, directory, profile=None):
    from contextlib import closing, nullcontext

    import chrombert
    import torch
    from tqdm import tqdm
    from transformers import get_linear_schedule_with_warmup

    training = config["training"]
    official = config["task"] == "eqtl-official"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = chrombert.get_preset_model_config(
        "prompt_dna", basedir=str(assets), dnabert2_ckpt=str(assets / DNA_MODEL),
    ).init_model().cuda()
    model.freeze_pretrain(trainable=training["trainable_encoder_layers"])
    model.dnabert2.freeze()
    train = eqtl_dataset(config, assets, paths["train"], shuffle=True)
    valid = None if official else eqtl_dataset(config, assets, paths["valid"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    accumulate = training["accumulate_grad_batches"]
    epochs = 1 if config.get("smoke") else training["max_epochs"]
    steps = (len(train) * epochs // accumulate if official else math.ceil(len(train) / accumulate) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, max(steps, 1))
    updates = 0
    picked = directory / "tutorial-selected.ckpt"
    best = math.inf

    def forward(raw):
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
        pred = model(batch).reshape(-1)
        return loss_fn(pred, batch["label"].float().reshape(-1))

    for epoch in range(epochs):
        model.train()
        stop = False
        batches = tqdm(train, desc=f"Train eQTL epoch {epoch + 1}")
        capture = nullcontext(batches)
        if profile:
            last = min(len(train), accumulate)
            capture = closing(profile.capture_batches("training", model, iter(batches), max(1, last - 1), last))
        with capture as training_batches:
            for index, batch in enumerate(training_batches, 1):
                group_size = accumulate
                if not official:
                    group_size = min(accumulate, len(train) - ((index - 1) // accumulate) * accumulate)
                (forward(batch) / group_size).backward()
                if index % accumulate == 0 or (not official and index == len(train)):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    updates += 1
                    if official and updates > training["max_steps"] and epoch >= training["min_epoch_index"]:
                        stop = True
                        break
        if profile:
            if updates != 1:
                raise RuntimeError("Expected one eQTL optimizer update during profiling.")
            return
        if valid is not None:
            model.eval()
            with torch.no_grad():
                total = sum(float(forward(batch)) * len(batch["label"]) for batch in valid)
            bce = total / len(valid.dataset)
            if math.isfinite(bce) and bce < best:
                best = bce
                model.save_ckpt(str(picked))
        if stop:
            break
    if official:
        picked = directory / "tutorial-final.ckpt"
        model.save_ckpt(str(picked))
    if not picked.is_file() or updates <= 0:
        raise RuntimeError("eQTL training did not produce usable weights.")
    return picked, None, {
        "criterion": "tutorial_last_update" if official else "validation/bce",
        "best_score": None if official else best, "epochs_started": epoch + 1,
        "optimizer_updates": updates, "scheduler_budget_updates": steps,
    }


def setup_inference(config, assets, paths, data, model_options, checkpoint):
    import chrombert

    model = chrombert.get_preset_model_config(
        "prompt_dna", basedir=str(assets), dnabert2_ckpt=str(assets / DNA_MODEL),
        finetune_ckpt=str(checkpoint), dropout=0,
    ).init_model().cuda().eval()
    return model, eqtl_dataset(config, assets, paths["test"])


class EqtlProtocol(TaskProtocol):
    REQUIRED_MODULES = REQUIRED_MODULES
    PREDICTION_IDS = PREDICTION_IDS
    PROFILE_TRAINING = PROFILE_TRAINING
    AUDIT_SPLITS = True
    SUPPORTS_SMOKE = True

    prepare = staticmethod(prepare)
    split_audit = staticmethod(split_audit)
    smoke_sample = staticmethod(smoke_sample)
    setup_inference = staticmethod(setup_inference)

    def train(self, config, assets, paths, data, options, work, profile=None, *, profile_only=False):
        return fit(config, assets, paths, work, profile if profile_only else None)


PROTOCOL = EqtlProtocol()
