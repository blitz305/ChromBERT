"""Classification of classical and nonclassical EZH2 sites."""

from training import dataset_config, excluded_track_options, lightning_data, load_inference_model
from ._splits import REGION_BED
from .protocol import TaskProtocol

USES_BEDTOOLS = True
WRITE_SPLIT_INDEX = True
SET_PYTHON_PROTOBUF = True


def prepare(config, inputs):
    settings, counts = config["split"], config["expected_counts"]
    from chrombert.scripts.chrombert_make_dataset import get_overlap
    from sklearn.model_selection import train_test_split

    region_bed = inputs[REGION_BED]
    ezh2 = get_overlap(inputs["demo/ezh2/hESC_GSM1003524_EZH2.bed"], region_bed)
    h3k27me3 = get_overlap(inputs["demo/ezh2/hESC_GSM1498900_H3K27me3.bed"], region_bed, no_filter=True)
    ezh2["label"] = ezh2["label"] > 0
    h3k27me3["label"] = h3k27me3["label"] > 0
    table = ezh2.rename(columns={"label": "EZH2"}).merge(h3k27me3)
    seed = settings["seed"]
    train, remaining = train_test_split(table, test_size=0.2, random_state=seed, stratify=table["label"])
    valid, test = train_test_split(remaining, test_size=0.5, random_state=seed, stratify=remaining["label"])
    splits = {"train": train, "valid": valid, "test": test}
    if settings.get("downsample", False):
        for split, frame in splits.items():
            if counts[split] < len(frame):
                splits[split], _ = train_test_split(
                    frame, train_size=counts[split], random_state=seed, stratify=frame["label"],
                )
    return splits


def setup_training(config, assets, paths):
    dataset = dataset_config(config, assets)
    options = excluded_track_options(config, dataset, paths["train"])
    return lightning_data(dataset, paths), options


def setup_inference(config, assets, paths, data, model_options, checkpoint):
    model = load_inference_model(config, assets, model_options, checkpoint)
    return model.eval(), data.test_config.init_dataloader(batch_size=1)


class Ezh2Protocol(TaskProtocol):
    USES_BEDTOOLS = USES_BEDTOOLS
    WRITE_SPLIT_INDEX = WRITE_SPLIT_INDEX
    SET_PYTHON_PROTOBUF = SET_PYTHON_PROTOBUF

    prepare = staticmethod(prepare)
    setup_training = staticmethod(setup_training)
    setup_inference = staticmethod(setup_inference)


PROTOCOL = Ezh2Protocol()
