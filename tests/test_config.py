"""Configuration compatibility, composition, validation, and logging checks."""

from __future__ import annotations

import copy
import sys
import tempfile
import tomllib
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from loguru import logger
from pydantic import Field, ValidationError

from src.config import (
    Config, LoggingConfig, ModelConfig, PretrainingConfig, TokenDatasetConfig,
    configure_logging,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.messages = []
        logger.remove()
        logger.add(self.messages.append, level="DEBUG", format="{level} {message}")
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        logger.remove()
        logger.add(sys.stderr)
        self.directory.cleanup()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def warnings(self):
        return [message.record["message"] for message in self.messages
                if message.record["level"].name == "WARNING"]

    def test_defaults_and_strict_nested_validation(self):
        config = PretrainingConfig.from_dict({"name": "test", "training": {"batch_size": 16}})
        self.assertEqual(config.training.batch_size, 16)
        self.assertEqual(config.training.grad_accum_steps, 1)
        self.assertEqual(config.model.embed_dim, 512)
        self.assertEqual(config.logging.throughput_interval_steps, 20)
        self.assertEqual(config.to_dict()["optimizer"]["muon"]["ns_iters"], 5)
        for value in ("16", True, 0):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PretrainingConfig.from_dict({"name": "test", "training": {"batch_size": value}})

    def test_independent_token_dataset_configs_and_typed_overrides(self):
        config = PretrainingConfig(name="test")
        self.assertIsInstance(config.train_dataset, TokenDatasetConfig)
        self.assertIsInstance(config.validation_dataset, TokenDatasetConfig)
        self.assertIsNot(config.train_dataset, config.validation_dataset)
        self.assertEqual(config.train_dataset.path, "train_10m")
        self.assertEqual(config.validation_dataset.path, "validation")
        self.assertNotIn("data", config.to_dict())
        changed = config.with_overrides({"train_dataset.path": "/datasets/custom"})
        self.assertEqual(changed.train_dataset.path, "/datasets/custom")
        self.assertEqual(changed.validation_dataset.path, "validation")
        self.assertEqual(config.train_dataset.path, "train_10m")
        for invalid in ({}, {"path": ""}, {"path": 42}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                TokenDatasetConfig.from_dict(invalid)
        with self.assertRaisesRegex(ValueError, "unknown override path"):
            config.with_overrides({"train_dataset.paht": "typo"})

    def test_activation_memory_budget_survives_overrides_and_round_trips(self):
        config = PretrainingConfig(name="test")
        self.assertEqual(config.runtime.act_checkpoint_mem_budget, 1.0)
        for budget in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(budget=budget):
                changed = config.with_overrides({"runtime.act_checkpoint_mem_budget": budget})
                self.assertEqual(changed.runtime.act_checkpoint_mem_budget, budget)
                self.assertEqual(PretrainingConfig.from_dict(changed.to_dict()), changed)
                self.assertEqual(PretrainingConfig.from_toml(changed.to_toml()), changed)
        for budget in (-0.1, 1.1):
            with self.subTest(budget=budget), self.assertRaises(ValidationError):
                config.with_overrides({"runtime.act_checkpoint_mem_budget": budget})
        eager = config.with_overrides({
            "runtime.compile_model": False, "runtime.act_checkpoint_mem_budget": 0.5,
        })
        self.assertEqual(eager.runtime.act_checkpoint_mem_budget, 0.5)
        self.assertTrue(any("ignored" in text for text in self.warnings()))

    def test_dataset_composition_round_trip_and_unknown_settings(self):
        self.write("datasets.toml", '[train_dataset]\npath="train-custom"\nfuture_option=true\n')
        source = self.write("experiment.toml", 'extends=["datasets.toml"]\nname="test"\n'
                            '[validation_dataset]\npath="validation-custom"\n')
        config = PretrainingConfig.load(source, overrides={"train_dataset.path": "override"})
        self.assertEqual(config.train_dataset.path, "override")
        self.assertEqual(config.validation_dataset.path, "validation-custom")
        self.assertTrue(config.train_dataset.model_extra["future_option"])
        self.assertTrue(any("train_dataset.future_option" in text for text in self.warnings()))
        self.assertEqual(PretrainingConfig.from_toml(config.to_toml()), config)
        dataset = TokenDatasetConfig(path="standalone")
        path = self.root / "token-dataset.toml"
        dataset.save(path)
        self.assertEqual(TokenDatasetConfig.load(path), dataset)

    def test_old_data_sections_migrate_without_mutating_or_losing_extras(self):
        data = {"name": "test", "data": {"train": "old-train", "validation": "old-val", "future": 7},
                "train_dataset": {"path": "new-train", "future_setting": True}}
        original = copy.deepcopy(data)
        config = PretrainingConfig.from_dict(data)
        self.assertEqual(data, original)
        self.assertEqual(config.train_dataset.path, "new-train")
        self.assertEqual(config.validation_dataset.path, "old-val")
        self.assertTrue(config.train_dataset.model_extra["future_setting"])
        self.assertEqual(config.model_extra["data"], {"future": 7})
        self.assertTrue(any("deprecated config section 'data'" in text for text in self.warnings()))
        self.assertTrue(any("Unknown config section 'data'" in text for text in self.warnings()))
        migrated = PretrainingConfig.from_toml('name="test"\n[data]\ntrain="old"\nvalidation="val"\n')
        self.assertNotIn("data", migrated.to_dict())
        self.assertEqual(migrated.train_dataset.path, "old")
        self.assertEqual(migrated.validation_dataset.path, "val")
        self.assertEqual(PretrainingConfig.from_toml(migrated.to_toml()), migrated)

    def test_experiment_name_is_required_at_every_load_boundary(self):
        path = self.write("preset.toml", "[training]\nbatch_size=16\n")
        loaders = [
            lambda: PretrainingConfig(),
            lambda: PretrainingConfig.from_dict({}),
            lambda: PretrainingConfig.from_toml("[training]\nbatch_size=16\n"),
            lambda: PretrainingConfig.load(path),
        ]
        for load in loaders:
            with self.subTest(load=load), self.assertRaises(ValidationError) as caught:
                load()
            self.assertTrue(any(
                error["loc"] == ("name",) and error["type"] == "missing"
                for error in caught.exception.errors()
            ))
        for name in ("", " ", None):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                PretrainingConfig.from_dict({"name": name})
        config = PretrainingConfig.load(path, overrides={"name": "explicit_trial"})
        self.assertEqual(config.name, "explicit_trial")
        self.assertEqual(config.training.batch_size, 16)

    def test_schedule_is_nested_and_old_snapshots_migrate(self):
        config = PretrainingConfig.from_dict({
            "name": "test",
            "schedule": {"kind": "wsd", "warmup_steps": 3, "decay_fraction": 0.5},
            "optimizer": {"schedule": {"warmup_steps": 4}},
        })
        self.assertEqual(config.optimizer.schedule.kind, "wsd")
        self.assertEqual(config.optimizer.schedule.warmup_steps, 4)
        self.assertEqual(config.optimizer.schedule.decay_fraction, 0.5)
        self.assertNotIn("schedule", config.to_dict())
        self.assertTrue(any("deprecated config section 'schedule'" in text for text in self.warnings()))
        self.assertEqual(PretrainingConfig.from_toml(config.to_toml()).to_dict(), config.to_dict())

    def test_model_and_cross_section_validation(self):
        invalid = [
            {"model": {"embed_dim": 17, "attn_heads": 4}},
            {"model": {"embed_dim": 12, "attn_heads": 4}},
            {"optimizer": {"lr": float("inf")}},
            {"optimizer": {"kind": "unimplemented"}},
            {"optimizer": {"muon": {"ns_coefficients": [1.0, 2.0]}}},
            {"optimizer": {"schedule": {"min_lr": 0.1}}},
            {"training": {"max_steps": 5}},
            {"name": "../unsafe"},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                PretrainingConfig.from_dict({"name": "test", **data})

    def test_unknown_fields_and_sections_warn_with_paths_and_survive(self):
        config = PretrainingConfig.model_validate({
            "name": "test",
            "training": {"batch_szie": 7},
            "optimizer": {"muon": {"future_option": 2}},
            "profiling": {"enabled": True, "start_step": 100},
        })
        warnings = self.warnings()
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("field 'training.batch_szie'" in text for text in warnings))
        self.assertTrue(any("field 'optimizer.muon.future_option'" in text for text in warnings))
        self.assertTrue(any("section 'profiling'" in text for text in warnings))
        self.assertEqual(config.training.batch_size, 32)
        exported = config.to_dict()
        self.assertEqual(exported["training"]["batch_szie"], 7)
        self.assertEqual(exported["profiling"]["start_step"], 100)
        self.assertEqual(PretrainingConfig.from_toml(config.to_toml()).to_dict(), exported)

    def test_config_evolution_in_both_directions(self):
        class Profiling(Config):
            enabled: bool = False
            steps: int = 20

        class Old(Config):
            batch_size: int = 32

        class New(Old):
            profiling: Profiling = Field(default_factory=Profiling)

        self.assertFalse(New.from_toml(Old().to_toml()).profiling.enabled)
        new = New(profiling=Profiling(enabled=True, steps=5))
        old = Old.from_toml(new.to_toml())
        self.assertEqual(New.from_toml(old.to_toml()).profiling.steps, 5)
        self.assertTrue(any("section 'profiling'" in text for text in self.warnings()))

    def test_generic_helpers_work_on_individual_configs(self):
        path = self.write("model.toml", "embed_dim=256\nattn_heads=4\n")
        config = ModelConfig.load(path)
        destination = self.root / "saved/model.toml"
        config.save(destination)
        self.assertEqual(ModelConfig.load(destination), config)

    def test_composition_merges_raw_presets_before_defaults(self):
        self.write("models/base.toml", "[model]\nembed_dim=256\nattn_heads=4\n")
        self.write("models/tuned.toml", 'extends=["base.toml"]\n[model]\nqk_norm=true\n')
        self.write("optimizers/muon.toml", '[optimizer]\nkind="muon_adamw"\nlr=0.0024\n')
        self.write("later.toml", "[model]\nn_blocks=8\n")
        path = self.write("experiment.toml", '''
extends=["models/tuned.toml", "optimizers/muon.toml", "later.toml"]
name="composed"
[optimizer]
lr=0.001
''')
        config = PretrainingConfig.load(path)
        self.assertEqual(config.model.embed_dim, 256)
        self.assertEqual(config.model.attn_heads, 4)
        self.assertEqual(config.model.n_blocks, 8)
        self.assertTrue(config.model.qk_norm)
        self.assertEqual(config.optimizer.kind, "muon_adamw")
        self.assertEqual(config.optimizer.lr, 0.001)
        self.assertNotIn("extends", config.to_dict())
        self.assertEqual(self.warnings(), [])

    def test_appended_extends_follow_existing_includes_before_own_fields_and_overrides(self):
        self.write("base.toml", '[model]\nqk_norm=true\n[optimizer]\nlr=0.004\n')
        self.write("nested/runtime.toml", '[runtime]\ndataset_dir="/datasets"\n')
        first = self.write("first.toml", 'extends=["nested/runtime.toml"]\n[model]\nqk_norm=false\n[optimizer]\nlr=0.003\n')
        second = self.write("second.toml", '[optimizer]\nlr=0.002\n[runtime]\noutput_dir="/outputs/runs"\n')
        source = 'extends=["base.toml"]\nname="test"\n[optimizer]\nlr=0.001\n'
        path = self.write("run.toml", source)
        config = PretrainingConfig.load(path, extends=[first.name, second])
        self.assertFalse(config.model.qk_norm)
        self.assertEqual(config.optimizer.lr, 0.001)
        self.assertEqual(config.runtime.dataset_dir, "/datasets")
        self.assertEqual(config.runtime.output_dir, "/outputs/runs")
        overridden = PretrainingConfig.load(path, extends=[first, second], overrides={"optimizer.lr": 0.0005})
        self.assertEqual(overridden.optimizer.lr, 0.0005)
        self.assertNotIn("extends", config.to_dict())
        self.assertEqual(path.read_text(), source)

    def test_appended_extends_validate_paths_and_detect_cycles(self):
        path = self.write("run.toml", 'name="test"\n')
        for includes in ("file.toml", path, [""], [12]):
            with self.subTest(includes=includes), self.assertRaises(ValueError):
                PretrainingConfig.load(path, extends=includes)
        with self.assertRaises(FileNotFoundError):
            PretrainingConfig.load(path, extends=["missing.toml"])
        self.write("child.toml", 'extends=["run.toml"]\n')
        with self.assertRaisesRegex(ValueError, "circular config includes"):
            PretrainingConfig.load(path, extends=["child.toml"])

    def test_list_replacement_and_ordered_files(self):
        first = self.write("first.toml", '''
[optimizer.muon]
momentum=0.8
ns_coefficients=[1.0, 2.0, 3.0]
''')
        second = self.write("second.toml", '''
name="test"
[optimizer.muon]
ns_coefficients=[4.0, 5.0, 6.0]
''')
        config = PretrainingConfig.load([first, second])
        self.assertEqual(config.optimizer.muon.momentum, 0.8)
        self.assertEqual(config.optimizer.muon.ns_coefficients, [4.0, 5.0, 6.0])

    def test_already_parsed_overrides_take_precedence_over_files(self):
        path = self.write("run.toml", 'name="test"\n[optimizer]\nlr=0.002\n')
        overrides = {
            "optimizer.lr": 0.004,
            "model.qk_norm": True,
            "optimizer.kind": "muon_adamw",
            "runtime.dataset_dir": "/datasets",
        }
        config = PretrainingConfig.load(path, overrides=overrides)
        self.assertEqual(config.optimizer.lr, 0.004)
        self.assertTrue(config.model.qk_norm)
        self.assertEqual(config.optimizer.kind, "muon_adamw")
        self.assertEqual(config.runtime.dataset_dir, "/datasets")
        with self.assertRaisesRegex(ValueError, "unknown override path"):
            PretrainingConfig.load(path, overrides={"training.batch_szie": 8})
        with self.assertRaises(ValidationError):
            PretrainingConfig.load(path, overrides={"training.batch_size": "16"})

    def test_environment_does_not_configure_the_new_api(self):
        path = self.write("run.toml", 'name="test"\n[optimizer]\nlr=0.002\n')
        with patch.dict("os.environ", {
            "PAVULLMO_OPTIMIZER__LR": "0.003", "LR": "0.004",
            "PAVULLMO_TRAINING__BATCH_SZIE": "8",
        }):
            self.assertEqual(PretrainingConfig.load(path).optimizer.lr, 0.002)
        with self.assertRaises(TypeError):
            PretrainingConfig.load(path, env={"PAVULLMO_OPTIMIZER__LR": "0.003"})

    def test_validated_copy_preserves_extras_without_mutation(self):
        config = PretrainingConfig.from_dict({"name": "test", "profiling": {"enabled": True}})
        before = copy.deepcopy(config.to_dict())
        edited = config.with_overrides({"training.batch_size": 16})
        self.assertEqual(config.to_dict(), before)
        self.assertEqual(edited.training.batch_size, 16)
        self.assertEqual(edited.to_dict()["profiling"], {"enabled": True})
        with self.assertRaisesRegex(ValueError, "unknown override path"):
            config.with_overrides({"training.batch_szie": 16})
        with self.assertRaises(ValidationError):
            config.with_overrides({"training.batch_size": "16"})
        with self.assertRaisesRegex(ValueError, "unknown override path"):
            config.with_overrides({"training": {"batch_szie": 16}})

    def test_saving_preserved_extras_does_not_repeat_load_warnings(self):
        config = PretrainingConfig.from_dict({"name": "test", "future": {"enabled": True}})
        before = self.warnings()
        config.save(self.root / "config.toml")
        self.assertEqual(self.warnings(), before)

    def test_compact_strings_preserve_nested_configs_and_escape_line_breaks(self):
        config = PretrainingConfig.from_dict({
            "name": "compact", "training": {"batch_size": 16},
            "future": {"message": 'Italiano: città, "quotes"; a=b\\c\nsecond\rthird',
                       "date": date(2026, 10, 15),
                       "items": [{"value": 1}, {"nested": {"value": [True, False]}}]},
        })
        before = copy.deepcopy(config.to_dict())
        text = config.to_string()
        self.assertNotIn("\n", text)
        self.assertNotIn("\r", text)
        self.assertEqual(tomllib.loads(text), {"config": before})
        self.assertEqual(PretrainingConfig.from_string(text).to_dict(), before)
        self.assertEqual(config.to_dict(), before)
        self.assertEqual(PretrainingConfig.from_string(config.to_toml()).to_dict(), before)
        for section in (config.model, config.optimizer, config.train_dataset):
            self.assertEqual(type(section).from_string(section.to_string()), section)
        with self.assertRaisesRegex(ValueError, "inline table"):
            PretrainingConfig.from_string("config = 42")
        with self.assertRaisesRegex(ValueError, "None"):
            Config.from_dict({"future": None}).to_string()

    def test_resolved_snapshot_is_independent_of_source_files(self):
        base = self.write("base.toml", "[training]\nbatch_size=8\n")
        source = self.write("source.toml", 'extends=["base.toml"]\nname="test"\n')
        config = PretrainingConfig.load(source)
        output = self.root / "runs/test/config.toml"
        config.save(output)
        source.unlink()
        base.unlink()
        saved = tomllib.loads(output.read_text())
        self.assertNotIn("extends", saved)
        self.assertEqual(saved["training"]["grad_accum_steps"], 1)
        self.assertEqual(PretrainingConfig.load(output).to_dict(), config.to_dict())

    def test_invalid_include_graphs(self):
        first = self.write("a.toml", 'extends=["b.toml"]\n')
        self.write("b.toml", 'extends=["a.toml"]\n')
        with self.assertRaisesRegex(ValueError, "circular config includes"):
            PretrainingConfig.load(first)
        for value in ('"base.toml"', "[1]", '[""]'):
            path = self.write("invalid.toml", f"extends={value}\n")
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "list of nonempty"):
                PretrainingConfig.load(path)
        missing = self.write("missing.toml", 'extends=["absent.toml"]\n')
        with self.assertRaises(FileNotFoundError):
            PretrainingConfig.load(missing)
        with self.assertRaises(ValueError):
            PretrainingConfig.load([])
        with self.assertRaisesRegex(ValueError, "file location"):
            PretrainingConfig.from_toml('extends=["base.toml"]')

    def test_failed_save_does_not_replace_previous_snapshot(self):
        destination = self.write("saved.toml", 'name="original"\n')
        config = PretrainingConfig(name="test")
        with patch("src.config.serialization.os.fsync", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                config.save(destination)
        self.assertEqual(destination.read_text(), 'name="original"\n')
        self.assertEqual(list(self.root.glob("*.tmp")), [])
        config.training.batch_size = 0
        with self.assertRaises(ValidationError):
            config.save(destination)
        self.assertEqual(destination.read_text(), 'name="original"\n')

    def test_toml_native_dates_are_preserved_in_unknown_settings(self):
        config = PretrainingConfig.from_toml('name="test"\n[future]\ndate=2026-10-14\n')
        self.assertEqual(config.to_dict()["future"]["date"], date(2026, 10, 14))
        self.assertEqual(PretrainingConfig.from_toml(config.to_toml()).to_dict(), config.to_dict())

    def test_null_fails_instead_of_silently_losing_data(self):
        config = PretrainingConfig.from_dict({"name": "test", "future": {"value": None}})
        with self.assertRaisesRegex(ValueError, "future.value"):
            config.to_toml()

    def test_unknown_warnings_inside_lists_of_config_models(self):
        class Worker(Config):
            threads: int = 1

        class Distributed(Config):
            workers: list[Worker] = Field(default_factory=list)

        Distributed.from_dict({"workers": [{"threads": 2, "future_flag": True}]})
        self.assertTrue(any("workers[0].future_flag" in text for text in self.warnings()))

    def test_loguru_file_sink_and_level_filtering(self):
        path = self.root / "logs/config.log"
        settings = LoggingConfig(console=False, file=str(path), level="WARNING")
        configure_logging(settings)
        logger.info("filtered information")
        logger.warning("visible warning")
        logger.remove()
        text = path.read_text()
        self.assertIn("visible warning", text)
        self.assertNotIn("filtered information", text)


if __name__ == "__main__":
    unittest.main()
