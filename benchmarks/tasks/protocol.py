"""Shared task contract for split preparation, training, inference, and scoring."""

import math
import os
from abc import ABC, abstractmethod
from collections import namedtuple

from predictions import label, predict_batches, prediction_scores, save_predictions
from training import fit


PredictionBatch = namedtuple("PredictionBatch", "values sampled")


class TaskProtocol(ABC):
    PREDICTION_IDS = ("build_region_index",)
    BATCH_INDEX = "build_region_index"
    IS_REGRESSION = False
    OFFICIAL_CHECKPOINT = None
    AUDIT_SPLITS = False
    WRITE_SPLIT_INDEX = False
    USES_BEDTOOLS = False
    REQUIRED_MODULES = ()
    SET_PYTHON_PROTOBUF = False
    HAS_TRAINING = True
    SUPPORTS_SMOKE = False
    SUPPORTS_PROFILING = True
    PROFILE_TRAINING = True
    SECONDARY_SCORES = False

    @abstractmethod
    def prepare(self, config, inputs):
        """Prepare the task's splits, with an audit for tasks that declare it."""

    def prepare_split_data(self, config, inputs):
        prepared = self.prepare(config, inputs)
        return prepared if self.AUDIT_SPLITS else (prepared, None)

    def split_audit(self, config, frames):
        return None

    def smoke_sample(self, frame, seed):
        raise ValueError("This task has no smoke mode.")

    def setup_training(self, config, assets, paths):
        return None, None

    def prepare_training(self, config, assets, paths):
        if self.SET_PYTHON_PROTOBUF:
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        return self.setup_training(config, assets, paths)

    def train(self, config, assets, paths, data, options, work, profile=None, *, profile_only=False):
        """Return the selected checkpoint, optional final checkpoint, and selection record."""
        return fit(config, assets, data, options, work, profile, profile_only=profile_only)

    @abstractmethod
    def setup_inference(self, config, assets, paths, data, options, checkpoint):
        """Return the inference model and ordered inputs."""

    def predict(self, model, loader, profile=None, *, stop_after_capture=False):
        values = predict_batches(self, model, loader, profile, stop_after_capture=stop_after_capture)
        return PredictionBatch(values, {"samples": len(values[0])})

    def write_predictions(self, output, name, targets, batch):
        return save_predictions(self, output, name, targets, *batch.values)

    def prediction_stage_metrics(self, batch, target_count, wall_seconds):
        return {}

    def score_predictions(self, predictions, targets):
        for prediction, target in zip(predictions, targets):
            if any(prediction[key] != target[key] for key in self.PREDICTION_IDS) or not math.isclose(
                label(prediction["label"]), label(target["label"]), rel_tol=1e-6, abs_tol=1e-7,
            ):
                raise ValueError("A prediction row differs from the test split.")
        return prediction_scores(self, predictions)

    def verify_scores(self, config, scores):
        pass

    def secondary_scores(self, batch):
        return None
