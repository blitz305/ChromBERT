"""CPU coverage for task, variant, bundle, and launch contracts."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from archives import verify_result, verify_results
from benchmark import bundle, main
from benchmark_io import BENCHMARK_SCRIPTS, CODE, code_hashes, read_json, sha256, write_json
from launcher import available_gpu
from model_sources import MODEL_SOURCES
from tasks import TASKS, get_task, prepare_splits, starr
from tasks.protocol import PredictionBatch, TaskProtocol
from variants import add_variant_options, chosen_variant, get_variant, variant_from_config


def prediction_artifacts(directory, task, targets, outputs):
    """Save a miniature completed single-output run for the real verifier."""
    protocol = get_task(task)
    (directory / "splits").mkdir()
    columns = ["gene_id", "build_region_index", "label"]
    rows = [",".join(row[key] for key in columns) for row in targets]
    (directory / "splits/test.csv").write_text(",".join(columns) + "\n" + "\n".join(rows) + "\n")
    batch = PredictionBatch(([float(r["label"]) for r in targets], outputs, [1, 2, 3], None), {})
    scores = protocol.write_predictions(directory, "test_predictions", targets, batch)
    metrics = {
        "predictions": {"test_predictions": scores}, "peak_cpu_rss_bytes": 10, "run_wall_seconds": 0.1,
        "stages": {"inference": {
            "wall_seconds": 0.1, "torch_peak_allocated_bytes": 1, "torch_peak_reserved_bytes": 2,
        }},
    }
    config = {"task": task, "expected_counts": {"test": 3}, "inputs": [], "output_checkpoints": []}
    return config, metrics, scores


class TaskContractTests(unittest.TestCase):
    def test_every_task_resolves_to_the_contract(self):
        protocols = {alias: get_task(alias) for alias in TASKS}
        self.assertTrue(all(isinstance(task, TaskProtocol) for task in protocols.values()))
        self.assertIs(protocols["gep"], protocols["gep-sampled"])
        self.assertIs(protocols["eqtl"], protocols["eqtl-official"])
        self.assertIs(protocols["cistrome"], protocols["cistrome-sampled"])
        self.assertFalse(protocols["cistrome"].HAS_TRAINING)
        self.assertFalse(protocols["ezh2"].SUPPORTS_SMOKE)
        self.assertTrue(protocols["eqtl"].AUDIT_SPLITS)
        self.assertTrue(protocols["starr"].AUDIT_SPLITS)

    def test_ezh2_writer_keeps_split_index_and_rejects_wrong_counts(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "splits"
            frame = pd.DataFrame({"build_region_index": [8], "label": [True]}, index=[5])
            protocol = get_task("ezh2")
            config = {"task": "ezh2", "expected_counts": {"test": 1}, "split": {"seed": 42}}
            with patch.object(protocol, "prepare", return_value={"test": frame}):
                with self.assertRaisesRegex(ValueError, "sizes differ"):
                    prepare_splits(dict(config, expected_counts={"test": 2}), {}, destination)
                self.assertFalse(destination.exists())
                splits, audit = prepare_splits(config, {}, destination)
            self.assertIsNone(audit)
            self.assertIn("\n5,8,True", splits["test"].read_text())

    def test_starr_records_pair_counts_and_forward_throughput(self):
        protocol = get_task("starr")
        values = [(1, 2.0, 1.0), (2, 3.0, 1.0)]
        with patch.object(starr, "predict_pairs", return_value=(values, 2)):
            batch = protocol.predict(object(), object(), stop_after_capture=True)
        self.assertEqual(batch.sampled, {"pairs": 2, "changed_pairs": 2})
        self.assertEqual(protocol.prediction_stage_metrics(batch, 2, 0.5), {
            "forward_samples_per_second": 8.0, "input_changed_pairs": 2,
        })

    def test_eqtl_keeps_task_specific_training(self):
        from tasks import eqtl

        protocol = get_task("eqtl")
        tag = object()
        args = ({}, Path("/assets"), {}, None, None, Path("/work"), tag)
        with patch.object(eqtl, "fit", return_value=tag) as recorded:
            self.assertIs(protocol.train(*args), tag)
            recorded.assert_called_once_with(args[0], args[1], args[2], args[5], None)
            recorded.reset_mock()
            self.assertIs(protocol.train(*args, profile_only=True), tag)
            recorded.assert_called_once_with(args[0], args[1], args[2], args[5], tag)

    def test_gep_near_constant_predictions_keep_pearson_acceptance(self):
        targets = [
            {"gene_id": "a", "build_region_index": "1", "label": "-0.44589446492360874"},
            {"gene_id": "b", "build_region_index": "2", "label": "0"},
            {"gene_id": "c", "build_region_index": "3", "label": "-1.1707348442868757"},
        ]
        # The representative tied and last-row predictions from the sampled GEP archive.
        outputs = [-0.01833612099289894, -0.01833612099289894, -0.01833610236644745]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            config, metrics, scores = prediction_artifacts(directory, "gep", targets, outputs)
            values = {key: value for key, value in scores.items() if key != "samples"}
            metrics["tutorial_metrics"] = {"test_predictions": values}
            self.assertTrue(verify_results(directory, config, metrics)["predictions_matched_to_split"])
            values["pearsonr"] += 0.0015
            with self.assertRaisesRegex(ValueError, "TorchMetrics and independent recomputation differ"):
                verify_results(directory, config, metrics)


class BundleTests(unittest.TestCase):
    def test_official_bundle_carries_new_modules_and_rejects_a_missing_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            bundle("gep-sampled", directory)
            self.assertTrue((directory / "official.ipynb").is_file())
            self.assertEqual(set(code_hashes(directory)), set(BENCHMARK_SCRIPTS))
            (directory / "predictions.py").unlink()
            with self.assertRaises(FileNotFoundError):
                code_hashes(directory)

    def test_bundle_uses_cached_input_from_external_workspace(self):
        config = read_json(CODE / "configs/eqtl.json")
        relative, cached = next(iter(config["bundled_inputs"].items()))
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "shared"
            source = workspace / cached
            source.parent.mkdir(parents=True)
            source.write_text("variant_id,label\nv1,1\n", encoding="utf-8")
            config["input_sha256"][relative] = sha256(source)
            destination = Path(tmp) / "bundle"
            with patch("benchmark.read_json", return_value=config):
                bundle("eqtl", destination, workspace)
            self.assertEqual((destination / "inputs" / relative).read_bytes(), source.read_bytes())

    def test_archives_can_verify_a_source_without_an_official_notebook(self):
        class FixtureSource:
            def verify(self, config, directory):
                if sha256(directory / "source.txt") != config["source_sha256"]:
                    raise ValueError("Fixture source checksum mismatch")

            def artifacts(self, config):
                return {"source.txt"}

        targets = [{"gene_id": "unused", "build_region_index": str(i), "label": str(y)}
                   for i, y in enumerate([0, 1, 0], 1)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(MODEL_SOURCES, {"fixture": FixtureSource()}):
            directory = Path(tmp)
            config, metrics, _ = prediction_artifacts(directory, "ezh2", targets, [-1, 3, -2])
            source = directory / "source.txt"
            source.write_text("model revision a\n")
            write_json(directory / "config.json", dict(config, model_source="fixture", source_sha256=sha256(source)))
            metrics["output_files"] = {
                path.relative_to(directory).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in directory.rglob("*") if path.is_file()
            }
            write_json(directory / "metrics.json", metrics)
            self.assertTrue(verify_result(directory)["predictions_matched_to_split"])
            (directory / "test_predictions.csv").write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "size or checksum"):
                verify_result(directory)

    def test_legacy_gep_profile_checks_trace_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stages = ("training", "inference")
            for stage in stages:
                (directory / (stage + "_trace.json")).write_text("{}")
            digest = "a" * 64
            write_json(directory / "config.json", {
                "task": "gep", "environment": {}, "checkpoint_sha256": digest,
                "sample_test_sha256": digest, "training_split_sha256": {"train": digest, "valid": digest},
            })
            write_json(directory / "status.json", {"state": "complete", "verified": True})
            write_json(directory / "metrics.json", {"profiling": {stage: {"cuda_events": 1} for stage in stages}})
            check = {
                "profile_only": True, "matched_predictions": 16, "max_abs_difference_from_archive": 0.0,
                "trace_sha256": {path.name: sha256(path) for path in directory.glob("*_trace.json")},
            }
            write_json(directory / "check.json", check)
            self.assertEqual(verify_result(directory), check)
            (directory / "inference_trace.json").write_text("damaged")
            with self.assertRaisesRegex(ValueError, "trace checksum mismatch"):
                verify_result(directory)


class VariantTests(unittest.TestCase):
    def test_cli_alias_and_archived_config_resolve_same_variant(self):
        parser = argparse.ArgumentParser()
        add_variant_options(parser)
        old = chosen_variant(parser.parse_args(["--gelu", "builtin"]))
        self.assertIs(old, chosen_variant(parser.parse_args(["--variant", "gelu-builtin"])))
        self.assertIs(old, variant_from_config({"gelu": "builtin"}))

    def test_gelu_patch_restores_original_on_error(self):
        modules = {}
        for name in ("chrombert", "chrombert.base", "chrombert.base.utils", "chrombert.base.utils.feed_forward"):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(modules[parent], child, modules[name])
        ff = modules["chrombert.base.utils.feed_forward"]
        ff.GELU = original = object()
        modules["torch"] = SimpleNamespace(nn=SimpleNamespace(GELU=lambda **kwargs: kwargs))
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "run failed"):
                with get_variant("gelu-builtin").activate():
                    self.assertEqual(ff.GELU(), {"approximate": "tanh"})
                    raise RuntimeError("run failed")
        self.assertIs(ff.GELU, original)


class LaunchTests(unittest.TestCase):
    def test_launch_routes_to_the_explicit_shared_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp).resolve()
            argv = ["benchmark.py", "launch", "gep", "--gpu", "3", "--workspace", tmp]
            with patch.object(sys, "argv", argv), patch("launcher.launch") as mocked:
                main()
            task, mode, gpu, actual_workspace, build_bundle, *_ = mocked.call_args[0]
            self.assertEqual((task, mode, gpu, actual_workspace), ("gep", "full", 3, workspace))
            self.assertEqual(build_bundle.keywords["workspace"], workspace)

    def test_rejects_resident_memory_without_a_compute_process(self):
        with patch("launcher.capture", side_effect=["3, GPU-3, 20", ""]):
            with self.assertRaisesRegex(ValueError, "not idle"):
                available_gpu(3)

    def test_blocks_project_process_on_another_device(self):
        with patch("launcher.capture", side_effect=["2, GPU-2, 22\n3, GPU-3, 4", "GPU-2, 17"]):
            with patch("launcher.workspace_process", return_value=True):
                with self.assertRaisesRegex(ValueError, "Project GPU process"):
                    available_gpu(3, Path("/project/chrombert"))
