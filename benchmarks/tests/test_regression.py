"""CPU regression tests for the benchmark's recorded data contracts."""

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark import sha256, write_json
from runner import prediction_scores, reference_run, save_predictions
from tasks import starr
from tasks._splits import proportional_counts


class PredictionTests(unittest.TestCase):
    def test_single_prediction_scores(self):
        targets = [
            {"variant_id": "a", "build_region_index": "1", "label": "0"},
            {"variant_id": "b", "build_region_index": "2", "label": "1"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            scores = save_predictions("eqtl", output, "test", targets, [0, 1], [-2, 3], [1, 2])
            self.assertEqual(scores["auroc"], 1)
            self.assertEqual(scores["auprc"], 1)
            for labels, indices, error in [
                ([1, 0], [1, 2], "label differs"),
                ([0, 1], [2, 1], "region differs"),
            ]:
                with self.assertRaisesRegex(ValueError, error):
                    save_predictions("eqtl", output, "test", targets, labels, [-2, 3], indices)

    def test_gep_constant_scores(self):
        rows = [{"label": str(i), "prediction": "0.5"} for i in range(3)]
        result = prediction_scores("gep", rows)
        self.assertIsNone(result["pearsonr"])
        self.assertIsNone(result["spearmanr"])
        self.assertAlmostEqual(result["mae"], 5 / 6)

    def test_cistrome_requires_both_labels(self):
        with self.assertRaisesRegex(ValueError, "both labels"):
            prediction_scores("cistrome", [{"label": "True", "prediction": "1"}])

    def test_starr_paired_scoring(self):
        targets = [
            {"build_region_index": "1", "wt": "2.0", "ep300": "1.0", "diff": "-1.0"},
            {"build_region_index": "2", "wt": "3.0", "ep300": "1.0", "diff": "-2.0"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.csv"
            starr.save_predictions(path, targets, [(1, 2.0, 1.0), (2, 3.0, 1.0)])
            with path.open(newline="") as stream:
                observed = list(csv.DictReader(stream))
            self.assertEqual(starr.score_predictions(observed, targets)["effect_rmse"], 0)
            observed[0]["prediction_diff"] = "0"
            with self.assertRaisesRegex(ValueError, "saved differences"):
                starr.score_predictions(observed, targets)

    def test_sampling_quotas(self):
        self.assertEqual(proportional_counts({0: 24, 1: 5, 2: 1}, 6), {0: 5, 1: 1, 2: 0})


class ReferenceTests(unittest.TestCase):
    def test_failed_metrics_can_supply_unchanged_splits_only(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)
            (archive / "splits").mkdir()
            split = archive / "splits/test.csv"
            split.write_text("label\n0\n", encoding="utf-8")
            config = {"task": "cistrome", "expected_counts": {"test": 1}, "output_checkpoints": []}
            write_json(archive / "config.json", config)
            write_json(archive / "status.json", {"state": "failed"})
            write_json(archive / "metrics.json", {"output_files": {"splits/test.csv": {"sha256": sha256(split)}}})
            comparison = dict(config, gelu="builtin", gelu_comparison="end_to_end")
            self.assertEqual(reference_run(comparison, archive)[0], {"test": split})
            with self.assertRaisesRegex(ValueError, "different split"):
                reference_run(dict(comparison, split={"seed": 17}), archive)
            with self.assertRaises(ValueError):
                reference_run(dict(comparison, gelu_comparison="profile_only"), archive)
            split.write_text("label\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                reference_run(comparison, archive)


if __name__ == "__main__":
    unittest.main()
