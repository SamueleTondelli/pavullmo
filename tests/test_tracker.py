"""Tests for the static experiment tracker dataset and maintenance CLI."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import update_tracker  # noqa: E402


class TrackerValidationTests(unittest.TestCase):
    def test_repository_tracker_bundle_validates(self) -> None:
        data_path = ROOT / "docs" / "data" / "tracker.json"
        js_path = ROOT / "docs" / "data" / "tracker.js"
        data = update_tracker.load_json(data_path)
        errors = update_tracker.validate_tracker_data(data, js_path=js_path)
        self.assertEqual(errors, [])
        self.assertEqual(len(data["model_versions"]), 10)
        self.assertEqual(len(data["training_runs"]), 23)
        self.assertEqual(len(data["ablations"]), 18)

    def test_validation_rejects_broken_invariants(self) -> None:
        base_data = update_tracker.load_json(ROOT / "docs" / "data" / "tracker.json")

        bad_fk = copy.deepcopy(base_data)
        bad_fk["training_runs"][0]["version_id"] = "nonexistent-version"
        self.assertTrue(
            any("unknown `version_id`" in e for e in update_tracker.validate_tracker_data(bad_fk))
        )

        dup_id = copy.deepcopy(base_data)
        dup_id["model_versions"][1]["id"] = dup_id["model_versions"][0]["id"]
        self.assertTrue(
            any("Duplicate model_versions id" in e for e in update_tracker.validate_tracker_data(dup_id))
        )

        bad_date = copy.deepcopy(base_data)
        bad_date["training_runs"][0]["date"] = "22-11-2026"
        self.assertTrue(
            any("invalid `date`" in e for e in update_tracker.validate_tracker_data(bad_date))
        )

        bad_bench = copy.deepcopy(base_data)
        for run in bad_bench["training_runs"]:
            if run.get("benchmarks"):
                run["benchmarks"]["icl"] = 1.25
                break
        self.assertTrue(
            any("out of [0, 1]" in e for e in update_tracker.validate_tracker_data(bad_bench))
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            js_path = tmp / "tracker.js"
            js_path.write_text("window.PAVULLMO_TRACKER_DATA = {};\n", encoding="utf-8")
            errs = update_tracker.validate_tracker_data(base_data, js_path=js_path)
            self.assertTrue(any("out of sync" in e for e in errs))


class TrackerCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.data_path = self.tmp / "tracker.json"
        self.js_path = self.tmp / "tracker.js"
        shutil.copy2(ROOT / "docs" / "data" / "tracker.json", self.data_path)
        shutil.copy2(ROOT / "docs" / "data" / "tracker.js", self.js_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_parse_legacy_and_toml_config_strings(self) -> None:
        legacy = "VOCAB_SIZE=16384;N_BLOCKS=11;LR=0.00242;QK_NORM=True"
        parsed_legacy = update_tracker.parse_config_string(legacy)
        self.assertEqual(parsed_legacy["VOCAB_SIZE"], "16384")
        self.assertEqual(parsed_legacy["N_BLOCKS"], "11")
        self.assertEqual(parsed_legacy["LR"], "0.00242")

        toml_line = (
            'config = {name = "35m_icl_maxxing_1b", model = {vocab_size = 16384,'
            'n_blocks = 12,qk_norm = true}, optimizer = {lr = 0.003}}'
        )
        parsed_toml = update_tracker.parse_config_string(toml_line)
        self.assertEqual(parsed_toml["name"], "35m_icl_maxxing_1b")
        self.assertEqual(parsed_toml["vocab_size"], "16384")
        self.assertEqual(parsed_toml["qk_norm"], "true")
        self.assertEqual(parsed_toml["lr"], "0.003")

    def test_add_run_with_csv_eval_json_and_benchmarks_dir(self) -> None:
        csv_path = self.tmp / "runs.csv"
        csv_path.write_text(
            "experiment_name,dataset_variant,train_loss,validation_loss,train_script,model_path,config\n"
            'trial_csv,fineweb2_clean,3.2105,3.1415,pretrain.py,/tmp/m.pt,"config = {name = ""trial_csv"", model = {vocab_size = 16384}}"\n',
            encoding="utf-8",
        )
        eval_path = self.tmp / "eval.json"
        eval_path.write_text(
            json.dumps(
                {
                    "evaluations": {
                        "final_ema": {
                            "final_train_loss": 3.2105,
                            "full_val_loss": 3.0901,
                            "full_test_loss": 3.1022,
                            "benchmark_overall_bidir": 0.415,
                            "benchmark_category_bidir": {
                                "icl": 0.36,
                                "knowledge": 0.5,
                                "language-manipulation": 0.71,
                                "math": 0.42,
                                "math-languages": 0.28,
                                "reasoning": 0.22,
                            },
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        rc = update_tracker.main(
            [
                "--data",
                str(self.data_path),
                "--js",
                str(self.js_path),
                "add-run",
                "--id",
                "trial_csv_run",
                "--name",
                "trial_csv",
                "--date",
                "2026-11-23",
                "--version-id",
                "v6-iclmax-35m",
                "--tokens",
                "500M tok",
                "--from-csv",
                str(csv_path),
                "--experiment",
                "trial_csv",
                "--eval-json",
                str(eval_path),
                "--eval-key",
                "final_ema",
                "--notes",
                "CSV and eval ingestion test",
            ]
        )
        self.assertEqual(rc, 0)
        updated = update_tracker.load_json(self.data_path)
        self.assertEqual(updated["training_runs"][0]["id"], "trial_csv_run")
        self.assertEqual(updated["training_runs"][0]["val_loss"], 3.0901)
        self.assertEqual(updated["training_runs"][0]["val_loss_prefix"], 3.1415)
        self.assertEqual(updated["training_runs"][0]["test_loss"], 3.1022)
        self.assertEqual(updated["training_runs"][0]["benchmarks"]["overall_bidir"], 0.415)
        self.assertEqual(update_tracker.validate_tracker_data(updated, js_path=self.js_path), [])

        # Also test --benchmarks-dir ingestion on a second run.
        scores_root = self.tmp / "scores"
        for idx, (cat, _) in enumerate(update_tracker.BENCHMARK_CATEGORIES):
            cat_dir = scores_root / cat / "trial_dir_model"
            cat_dir.mkdir(parents=True, exist_ok=True)
            (cat_dir / "task1-bench-scores.json").write_text(
                json.dumps({"metrics": {"macro_bidirectional_accuracy": 0.3 + 0.02 * idx}}),
                encoding="utf-8",
            )

        rc2 = update_tracker.main(
            [
                "--data",
                str(self.data_path),
                "--js",
                str(self.js_path),
                "add-run",
                "--id",
                "trial_dir_run",
                "--name",
                "trial_dir_run",
                "--date",
                "2026-11-24",
                "--version-id",
                "v6-iclmax-35m",
                "--dataset",
                "custom_mix",
                "--tokens",
                "300M tok",
                "--benchmarks-dir",
                str(scores_root),
                "--benchmark-model",
                "trial_dir_model",
            ]
        )
        self.assertEqual(rc2, 0)
        updated2 = update_tracker.load_json(self.data_path)
        self.assertEqual(updated2["training_runs"][0]["id"], "trial_dir_run")
        self.assertAlmostEqual(updated2["training_runs"][0]["benchmarks"]["overall_bidir"], 0.35)

    def test_add_version_and_add_ablation_round_trip(self) -> None:
        rc_ver = update_tracker.main(
            [
                "--data",
                str(self.data_path),
                "--js",
                str(self.js_path),
                "add-version",
                "--id",
                "v7-test-100m",
                "--name",
                "100M Test Recipe",
                "--commit",
                "abcdef1",
                "--commit-date",
                "2026-11-25",
                "--branch",
                "main",
                "--total-params",
                "105000000",
                "--params-label",
                "105.00M",
                "--layers",
                "16",
                "--embed-dim",
                "768",
                "--ffn-dim",
                "2048",
                "--vocab-size",
                "16384",
                "--pos-encoding",
                "RoPE (base=50,000)",
                "--attention-variant",
                "GQA-2 (12Q/4KV)",
                "--optimizer-summary",
                "Muon + AdamW",
                "--optimizers",
                "Muon + AdamW",
                "--lr",
                "2.0e-3",
                "--schedule",
                "WSD 60/40",
                "--weight-decay",
                "0.10",
                "--batch-size",
                "BS 32",
            ]
        )
        self.assertEqual(rc_ver, 0)

        rc_abl = update_tracker.main(
            [
                "--data",
                str(self.data_path),
                "--js",
                str(self.js_path),
                "add-ablation",
                "--id",
                "ABL-99",
                "--date",
                "2026-11-25",
                "--title",
                "Test Ablation",
                "--category",
                "Architecture",
                "--status",
                "Promising",
                "--testing",
                "Tested GQA-4 vs GQA-2 at 100M.",
                "--results",
                "Matched validation loss within 0.002 nats.",
                "--conclusion",
                "Keep GQA-2 pending 1B verification.",
                "--source",
                "notes/test.md",
            ]
        )
        self.assertEqual(rc_abl, 0)

        final_data = update_tracker.load_json(self.data_path)
        self.assertEqual(final_data["model_versions"][0]["id"], "v7-test-100m")
        self.assertEqual(final_data["ablations"][0]["id"], "ABL-99")
        self.assertEqual(update_tracker.validate_tracker_data(final_data, js_path=self.js_path), [])


if __name__ == "__main__":
    unittest.main()
