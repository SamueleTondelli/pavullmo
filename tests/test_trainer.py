"""PretrainingTrainer construction and complete, deterministic checkpoint round trips."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
import signal
import shutil
import struct
import tempfile
import unittest
from itertools import islice
from pathlib import Path
from unittest.mock import patch

import torch

from config import PretrainingConfig
from config.serialization import dump_toml
from components.checkpoint import CheckpointManager
from components.pretraining_trainer import PretrainingTrainer


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.device_patch = patch("components.generic_trainer._select_device", return_value=torch.device("cpu"))
        self.device_patch.start()
        self.trainers = []
        self.artifact("train", 129)
        self.artifact("validation", 65)

    def tearDown(self):
        for trainer in self.trainers:
            trainer.close()
        self.device_patch.stop()
        self.directory.cleanup()

    def artifact(self, name, token_count, tokenizer_hash="tokenizer-sha256"):
        directory = self.root / "datasets" / name
        directory.mkdir(parents=True)
        shards = []
        for index, start in enumerate(range(0, token_count, 19)):
            tokens = [i % 32 for i in range(start, min(start + 19, token_count))]
            data = struct.pack(f"<{len(tokens)}H", *tokens)
            file = f"tokens-{index}.bin"
            (directory / file).write_bytes(data)
            shards.append({"file": file, "tokens": len(tokens), "sha256": hashlib.sha256(data).hexdigest()})
        metadata = {
            "storage": {"dtype": "uint16", "endianness": "little", "layout": "flat_token_stream"},
            "token_count": token_count, "shards": shards,
            "tokenizer": {"vocab_size": 32, "sha256": tokenizer_hash},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata))
        return directory

    def config(self, optimizer="adamw", schedule="cosine"):
        return PretrainingConfig.from_dict({
            "name": "test", "train_dataset": {"path": "train"}, "validation_dataset": {"path": "validation"},
            "model": {"vocab_size": 32, "n_blocks": 1, "embed_dim": 16, "attn_heads": 4,
                      "ffn_dim": 32, "seq_len": 8, "dropout": 0.2},
            "optimizer": {"kind": optimizer, "adamw": {"fused": False},
                          "schedule": {"kind": schedule, "warmup_steps": 1}},
            "training": {"batch_size": 2, "grad_accum_steps": 2, "epochs": 3},
            "runtime": {"dataset_dir": str(self.root / "datasets"), "output_dir": str(self.root / "runs"),
                        "runs_csv": str(self.root / "results.csv"),
                        "num_workers": 0, "pin_memory": False, "compile_model": False},
            "logging": {"console": False},
        })

    def trainer(self, config=None):
        trainer = PretrainingTrainer(config or self.config())
        self.trainers.append(trainer)
        return trainer

    def resume(self, path, **kwargs):
        trainer = PretrainingTrainer.from_checkpoint(path, **kwargs)
        self.trainers.append(trainer)
        return trainer

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_state_equal(first, second)
        else:
            self.assertEqual(left, right)

    def update(self, trainer):
        """Exercise a completed update without implementing the training loop."""
        trainer.train_loader.set_epoch(trainer.epoch, trainer.next_batch)
        iterator = iter(trainer.train_loader)
        trainer.model.zero_grad(set_to_none=True)
        for _ in range(trainer.config.training.grad_accum_steps):
            inputs, targets = next(iterator)
            inputs, targets = inputs.to(trainer.device), targets.to(trainer.device)
            with torch.autocast(trainer.device.type, dtype=torch.bfloat16, enabled=trainer.device.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(trainer.train_model(inputs).flatten(0, 1), targets.flatten())
            (loss / trainer.config.training.grad_accum_steps).backward()
            trainer.next_batch += 1
            trainer.tokens_seen += inputs.numel()
        trainer.optimizer.step_optimizer()
        trainer.optimizer.step_schedule()
        trainer.global_step += 1
        trainer.train_loss = loss.item()
        if trainer.next_batch == trainer.optimizer_steps_per_epoch * trainer.config.training.grad_accum_steps:
            trainer.epoch += 1
            trainer.next_batch = 0

    def test_constructor_resolves_components_and_horizon_without_mutating_input(self):
        config = self.config()
        trainer = self.trainer(config)
        self.assertEqual(trainer.optimizer_steps_per_epoch, 4)
        self.assertEqual(trainer.total_steps, 12)
        self.assertEqual(trainer.optimizer.total_steps, 12)
        self.assertEqual(trainer.config.to_dict(), config.to_dict())
        self.assertNotIn("total_steps", trainer.config.optimizer.schedule.to_dict())
        self.assertEqual(trainer.global_step, 0)
        self.assertIs(trainer.train_model, trainer.model)
        self.assertTrue(all(p.dtype == torch.float32 for p in trainer.model.parameters()))
        snapshot = PretrainingConfig.load(trainer.run_dir / "config.toml")
        self.assertEqual(snapshot.to_dict(), trainer.config.to_dict())
        capped = self.trainer(config.with_overrides({"name": "capped", "training.max_steps": 5}))
        self.assertEqual(capped.total_steps, 5)

    def test_absolute_artifact_paths_and_compilation_settings(self):
        config = self.config().with_overrides({
            "train_dataset.path": str(self.root / "datasets/train"),
            "validation_dataset.path": str(self.root / "datasets/validation"),
            "runtime.dataset_dir": "/unused", "runtime.compile_model": True,
            "runtime.compile_mode": "reduce-overhead",
        })
        with patch("components.generic_trainer.torch.compile", side_effect=lambda model, **kwargs: model) as compile:
            trainer = self.trainer(config)
        compile.assert_called_once_with(trainer.model, mode="reduce-overhead", dynamic=False)

    def test_constructor_rejects_incompatible_artifacts_and_horizons(self):
        with self.assertRaisesRegex(ValueError, "unknown override path"):
            self.trainer(self.config().with_overrides({"optimizer.schedule.total_steps": 20}))
        with self.assertRaises(ValueError):
            self.trainer(self.config().with_overrides({"optimizer.schedule.warmup_steps": 12}))
        path = self.root / "datasets/validation/metadata.json"
        metadata = json.loads(path.read_text())
        metadata["tokenizer"]["sha256"] = "different-tokenizer"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "same tokenizer"):
            self.trainer()
        metadata["tokenizer"]["sha256"] = "tokenizer-sha256"
        metadata["tokenizer"]["vocab_size"] = 64
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.trainer()

    def test_resume_migrates_old_data_config_and_preserves_next_update(self):
        trainer = self.trainer()
        self.update(trainer)
        path = trainer.save_checkpoint()
        payload = CheckpointManager.read(path)
        legacy = trainer.config.to_dict()
        legacy["data"] = {"train": legacy.pop("train_dataset")["path"],
                          "validation": legacy.pop("validation_dataset")["path"]}
        payload["config_toml"] = dump_toml(legacy)
        torch.save(payload, path)
        self.update(trainer)
        expected = copy.deepcopy(trainer.state_dict())
        restored = self.resume(path)
        self.assertEqual(restored.config.train_dataset.path, "train")
        self.assertEqual(restored.config.validation_dataset.path, "validation")
        self.assertNotIn("data", restored.config.to_dict())
        self.update(restored)
        self.assert_state_equal(expected, restored.state_dict())

    def test_older_checkpoint_computed_config_field_is_ignored_on_resume(self):
        trainer = self.trainer()
        path = trainer.save_checkpoint()
        payload = CheckpointManager.read(path)
        old_config = trainer.config.to_dict()
        old_config["optimizer"]["schedule"]["total_steps"] = trainer.total_steps
        payload["config_toml"] = PretrainingConfig.from_dict(old_config).to_toml()
        payload["trainer_state"]["training_signature"]["optimizer"]["schedule"]["total_steps"] = trainer.total_steps
        torch.save(payload, path)
        restored = self.resume(path)
        self.assertEqual(restored.optimizer.total_steps, 12)
        self.assertNotIn("total_steps", type(restored.config.optimizer.schedule).model_fields)
        broken = copy.deepcopy(restored.state_dict())
        broken["progress"]["total_steps"] = 99
        with self.assertRaisesRegex(ValueError, "duration differs"):
            restored.load_state_dict(broken)

    def test_full_round_trip_restores_counters_metrics_and_all_rng(self):
        trainer = self.trainer(self.config("muon_adamw"))
        self.update(trainer)
        trainer.validation_loss = 1.2
        trainer.best_validation_loss = 1.1
        trainer.clipped_steps = 1
        path = trainer.save_checkpoint()
        payload = CheckpointManager.read(path)
        expected = payload["trainer_state"]
        random.random()
        torch.rand(10)
        restored = self.resume(path)
        self.assert_state_equal(expected, restored.state_dict())
        self.assertEqual(restored.source_checkpoint, str(path))
        self.assertEqual(restored.next_batch, 2)
        self.assertEqual(restored.train_loader.batch_sampler.start_batch, 2)
        self.assertEqual(PretrainingConfig.from_toml(payload["config_toml"]).to_dict(), restored.config.to_dict())

    def test_resume_next_updates_match_uninterrupted_training_across_epoch(self):
        for optimizer in ("adamw", "muon_adamw"):
            for schedule in ("cosine", "wsd"):
                with self.subTest(optimizer=optimizer, schedule=schedule):
                    trainer = self.trainer(self.config(optimizer, schedule))
                    for _ in range(3):
                        self.update(trainer)
                    path = trainer.save_checkpoint()
                    for _ in range(2):
                        self.update(trainer)
                    expected = copy.deepcopy(trainer.state_dict())
                    restored = self.resume(path)
                    for _ in range(2):
                        self.update(restored)
                    self.assert_state_equal(expected, restored.state_dict())
                    self.assertEqual(restored.epoch, 1)

    def test_resume_allows_relocated_artifacts_and_output(self):
        trainer = self.trainer()
        self.update(trainer)
        path = trainer.save_checkpoint()
        relocated = self.root / "relocated"
        shutil.copytree(self.root / "datasets", relocated)
        restored = self.resume(path, overrides={
            "runtime.dataset_dir": str(relocated), "runtime.output_dir": str(self.root / "other-runs"),
            "name": "relocated",
        })
        self.assertEqual(restored.global_step, 1)
        self.assertEqual(restored.dataset_identities, trainer.dataset_identities)
        self.assertEqual(restored.run_dir, self.root / "other-runs/relocated")

    def test_resume_rejects_changed_settings_and_datasets(self):
        trainer = self.trainer()
        path = trainer.save_checkpoint()
        for overrides in ({"optimizer.lr": 0.02}, {"training.batch_size": 4}, {"model.dropout": 0.0}):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "training settings"):
                self.resume(path, overrides=overrides)
        metadata_path = self.root / "datasets/train/metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["shards"][0]["sha256"] = "different-shard"
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "dataset identities"):
            self.resume(path)

    def test_loading_existing_trainer_rejects_corruption_before_weight_changes(self):
        trainer = self.trainer(self.config("muon_adamw"))
        self.update(trainer)
        state = copy.deepcopy(trainer.state_dict())
        before = copy.deepcopy(trainer.model.state_dict())
        corruptions = []
        missing = copy.deepcopy(state)
        missing["optimizers"].pop("muon")
        corruptions.append(missing)
        cursor = copy.deepcopy(state)
        cursor["progress"]["next_batch"] = 999
        corruptions.append(cursor)
        shapes = copy.deepcopy(state)
        shapes["model"]["embeddings.weight"] = torch.zeros(1)
        corruptions.append(shapes)
        momentum = copy.deepcopy(state)
        first = next(iter(momentum["optimizers"]["muon"]["state"].values()))
        first["momentum_buffer"] = torch.zeros(1)
        corruptions.append(momentum)
        for section, key, value in (("progress", "next_batch", 1), ("progress", "tokens_seen", 0)):
            invalid = copy.deepcopy(state)
            invalid[section][key] = value
            corruptions.append(invalid)
        invalid = copy.deepcopy(state)
        invalid["schedulers"]["adamw"]["last_epoch"] = 0
        corruptions.append(invalid)
        for corrupt in corruptions:
            with self.assertRaises(ValueError):
                trainer.load_state_dict(corrupt)
            self.assert_state_equal(before, trainer.model.state_dict())

    def test_atomic_save_retention_latest_and_unrelated_files(self):
        trainer = self.trainer()
        first = trainer.save_checkpoint()
        unrelated = trainer.checkpoints.directory / "unrelated.pt"
        unrelated.write_bytes(b"keep")
        original = first.read_bytes()
        with patch("components.checkpoint.torch.save", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                trainer.save_checkpoint()
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(list(trainer.checkpoints.directory.glob("*.tmp")), [])
        initial = copy.deepcopy(trainer.state_dict())
        for _ in range(2):
            self.update(trainer)
            latest = trainer.save_checkpoint()
        self.assertFalse(first.exists())
        self.assertEqual(trainer.checkpoints.latest_checkpoint, latest)
        self.assertEqual(unrelated.read_bytes(), b"keep")
        trainer.load_state_dict(initial)
        newest = trainer.save_checkpoint()
        self.assertEqual(trainer.checkpoints.latest_checkpoint, newest)
        self.update(trainer)
        trainer.load_checkpoint()
        self.assertEqual(trainer.global_step, 0)

    def test_tensorboard_resume_retains_completed_step_and_purges_future_steps(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        trainer = self.trainer()
        self.update(trainer)
        trainer.writer.add_scalar("train/loss", trainer.train_loss, trainer.global_step)
        path = trainer.save_checkpoint()
        self.update(trainer)
        trainer.writer.add_scalar("train/loss", trainer.train_loss, trainer.global_step)
        trainer.writer.flush()
        trainer.load_checkpoint(path)
        trainer.writer.flush()
        events = EventAccumulator(str(trainer.run_dir / "tensorboard")).Reload()
        self.assertEqual([event.step for event in events.Scalars("train/loss")], [1])

    def test_model_only_and_invalid_formats_are_not_resumable(self):
        trainer = self.trainer(self.config().with_overrides({"checkpoint.save_training_state": False}))
        path = trainer.save_checkpoint()
        payload = CheckpointManager.read(path)
        self.assertFalse(payload["full_state"])
        self.assertEqual(set(payload["trainer_state"]), {"model"})
        with self.assertRaisesRegex(ValueError, "full checkpoint"):
            self.resume(path)
        with self.assertRaisesRegex(ValueError, "full checkpoint"):
            trainer.load_checkpoint(path)
        bad = self.root / "invalid.pt"
        torch.save({"format_version": 999}, bad)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            CheckpointManager.read(bad)

    def stop_at(self, trainer, step):
        write = trainer.writer.add_scalar

        def log(name, value, global_step, *args, **kwargs):
            write(name, value, global_step, *args, **kwargs)
            if name == "train/loss" and global_step == step:
                trainer.request_stop()

        return patch.object(trainer.writer, "add_scalar", side_effect=log)

    def test_train_completes_logs_validates_and_records_resolved_toml_once(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        trainer = self.trainer(self.config().with_overrides({
            "training.max_steps": 6, "evaluation.interval_steps": 3,
            "checkpoint.interval_steps": 2, "checkpoint.keep_last": 4,
        }))
        before = copy.deepcopy(trainer.model.state_dict())
        with patch.object(trainer, "validate", wraps=trainer.validate) as validate:
            trainer.train()
        self.assertEqual(validate.call_count, 2)
        self.assertEqual((trainer.global_step, trainer.epoch, trainer.next_batch, trainer.tokens_seen), (6, 1, 4, 192))
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in trainer.model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in trainer.model.parameters()))
        self.assertEqual([path.name for path in trainer.checkpoints._checkpoints()],
                         ["step_00000002.pt", "step_00000004.pt", "step_00000006.pt"])
        payload = CheckpointManager.read(trainer.checkpoints.latest_checkpoint)
        self.assert_state_equal(payload["trainer_state"], trainer.state_dict())
        events = EventAccumulator(str(trainer.run_dir / "tensorboard")).Reload()
        self.assertEqual([event.step for event in events.Scalars("train/loss")], list(range(1, 7)))
        self.assertEqual([event.step for event in events.Scalars("validation/loss")], [3, 6])
        self.assertEqual(events.Scalars("train/learning_rate")[-1].value, 0.0)
        self.assertIn("gradient_groups/attention_qkv", events.Tags()["scalars"])
        expected = copy.deepcopy(trainer.state_dict())
        trainer.train()
        self.assert_state_equal(expected, trainer.state_dict())
        with (self.root / "results.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(PretrainingConfig.from_string(rows[0]["config"]).to_dict(), trainer.config.to_dict())
        self.assertEqual(rows[0]["model_path"], str(trainer.checkpoints.latest_checkpoint))

    def test_results_csv_uses_one_physical_line_and_recognizes_old_multiline_rows(self):
        config = PretrainingConfig.from_dict({
            **self.config().to_dict(), "notes": 'città, "quoted"; first\nsecond\rthird',
        }).with_overrides({"training.max_steps": 2})
        trainer = self.trainer(config)
        trainer.train()
        text = trainer.runs_csv.read_text()
        self.assertEqual(len(text.splitlines()), 2)  # Header and one result.
        with trainer.runs_csv.open(newline="") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(PretrainingConfig.from_string(row["config"]).to_dict(), config.to_dict())
        self.assertNotIn("\n", row["config"])
        self.assertNotIn("\r", row["config"])

        # Existing valid CSV records may contain multiline TOML fields. Their
        # presence must not cause another result row for the same completed run.
        row["config"] = config.to_toml()
        with trainer.runs_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        before = trainer.runs_csv.read_bytes()
        trainer.train()
        self.assertEqual(trainer.runs_csv.read_bytes(), before)

    def test_configured_paths_are_used_and_computed_horizon_stays_outside_config(self):
        config = self.config().with_overrides({
            "training.max_steps": 2, "runtime.dataset_dir": str(self.root / "datasets"),
            "runtime.output_dir": str(self.root / "configured-runs"),
            "runtime.runs_csv": str(self.root / "configured-results.csv"),
        })
        expected = config.to_dict()
        trainer = self.trainer(config)
        trainer.train()
        self.assertEqual(trainer.dataset_dir, self.root / "datasets")
        self.assertEqual(trainer.run_dir, self.root / "configured-runs/test")
        self.assertEqual(trainer.runs_csv, self.root / "configured-results.csv")
        self.assertEqual(config.to_dict(), expected)
        self.assertEqual(trainer.config.to_dict(), expected)
        self.assertEqual(trainer.optimizer.config.to_dict(), config.optimizer.to_dict())
        self.assertEqual(PretrainingConfig.load(trainer.run_dir / "config.toml").to_dict(), expected)
        path = trainer.checkpoints.latest_checkpoint
        payload = CheckpointManager.read(path)
        self.assertEqual(PretrainingConfig.from_toml(payload["config_toml"]).to_dict(), expected)
        self.assertEqual(payload["trainer_state"]["progress"]["total_steps"], 2)
        with (self.root / "configured-results.csv").open(newline="") as file:
            self.assertEqual(PretrainingConfig.from_string(next(csv.DictReader(file))["config"]).to_dict(), expected)
        edits = {"runtime.output_dir": str(self.root / "resumed-runs"),
                 "runtime.runs_csv": str(self.root / "resumed-results.csv")}
        restored = self.resume(path, overrides=edits)
        restored.train()
        self.assertEqual(restored.config.to_dict(), config.with_overrides(edits).to_dict())
        self.assertEqual(config.to_dict(), expected)
        self.assertEqual(restored.run_dir, self.root / "resumed-runs/test")
        self.assertTrue((self.root / "resumed-results.csv").exists())

    def test_actual_loop_resume_matches_uninterrupted_with_dropout_and_validation(self):
        for optimizer in ("adamw", "muon_adamw"):
            for schedule in ("cosine", "wsd"):
                for stop_step in (3, 4):
                    with self.subTest(optimizer=optimizer, schedule=schedule, stop_step=stop_step):
                        config = self.config(optimizer, schedule).with_overrides({
                            "training.max_steps": 6, "evaluation.interval_steps": 2,
                            "checkpoint.interval_steps": 0,
                        })
                        baseline = self.trainer(config)
                        baseline.train()
                        expected = copy.deepcopy(baseline.state_dict())
                        interrupted = self.trainer(config)
                        with self.stop_at(interrupted, stop_step):
                            interrupted.train()
                        self.assertEqual(interrupted.global_step, stop_step)
                        restored = self.resume(interrupted.checkpoints.latest_checkpoint)
                        restored.train()
                        self.assert_state_equal(expected, restored.state_dict())

    def test_resume_with_prefetch_and_persistent_workers(self):
        config = self.config().with_overrides({
            "runtime.num_workers": 1, "training.max_steps": 6,
            "evaluation.interval_steps": 2, "checkpoint.interval_steps": 0,
        })
        baseline = self.trainer(config)
        baseline.train()
        expected = copy.deepcopy(baseline.state_dict())
        interrupted = self.trainer(config)
        with self.stop_at(interrupted, 3):
            interrupted.train()
        restored = self.resume(interrupted.checkpoints.latest_checkpoint)
        restored.train()
        self.assert_state_equal(expected, restored.state_dict())

    def test_same_trainer_can_continue_after_a_graceful_stop(self):
        config = self.config().with_overrides({"training.max_steps": 5, "evaluation.interval_steps": 2})
        baseline = self.trainer(config)
        baseline.train()
        expected = copy.deepcopy(baseline.state_dict())
        trainer = self.trainer(config)
        with self.stop_at(trainer, 3):
            trainer.train()
        trainer.train()
        self.assert_state_equal(expected, trainer.state_dict())

    def test_accumulation_z_loss_and_clipping_match_a_combined_batch(self):
        config = self.config().with_overrides({
            "model.dropout": 0.0, "training.max_steps": 2,
            "training.z_loss_coefficient": 0.01, "training.max_grad_norm": 0.05,
            "optimizer.schedule.warmup_steps": 0,
        })
        reference = self.trainer(config)
        reference.train_loader.set_epoch(0)
        batches = list(islice(iter(reference.train_loader), 2))
        inputs, targets = [torch.cat(parts) for parts in zip(*batches)]
        logits = reference.model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        z_loss = logits.logsumexp(dim=-1).square().mean()
        objective = loss + config.training.z_loss_coefficient * z_loss
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(reference.model.parameters(), config.training.max_grad_norm).item()
        reference.optimizer.step_optimizer()
        reference.optimizer.step_schedule()
        trainer = self.trainer(config)
        with self.stop_at(trainer, 1):
            trainer.train()
        self.assertAlmostEqual(trainer.train_loss, loss.item(), places=5)
        self.assertEqual(trainer.clipped_steps, 1)
        self.assertGreater(norm, config.training.max_grad_norm)
        for name, expected in reference.model.state_dict().items():
            torch.testing.assert_close(expected, trainer.model.state_dict()[name], rtol=1e-5, atol=1e-7)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(trainer.run_dir / "tensorboard")).Reload()
        self.assertAlmostEqual(events.Scalars("train/objective")[-1].value, objective.item(), places=5)
        self.assertAlmostEqual(events.Scalars("train/z_loss")[-1].value, z_loss.item(), places=5)
        self.assertAlmostEqual(events.Scalars("train/gradient_norm")[-1].value, norm, places=5)

    def test_epoch_tail_is_dropped_without_an_incomplete_update(self):
        self.artifact("odd-train", 145)  # Nine full batches, two per update.
        trainer = self.trainer(self.config().with_overrides({
            "train_dataset.path": "odd-train", "training.epochs": 2,
            "checkpoint.interval_steps": 0,
        }))
        with patch.object(trainer.train_model, "forward", wraps=trainer.train_model.forward) as forward:
            trainer.train()
        self.assertEqual(forward.call_count, 8 * 2 + len(trainer.validation_loader))
        self.assertEqual((trainer.global_step, trainer.epoch, trainer.next_batch, trainer.tokens_seen), (8, 2, 0, 256))

    def test_validation_weights_partial_batch_restores_mode_and_limits_batches(self):
        self.artifact("odd-validation", 41)  # Five sequences, batches of 2, 2, 1.
        trainer = self.trainer(self.config().with_overrides({"validation_dataset.path": "odd-validation"}))
        batches = list(trainer.validation_loader)
        trainer.model.eval()
        with torch.no_grad():
            expected = sum(torch.nn.functional.cross_entropy(
                trainer.model(inputs).flatten(0, 1), targets.flatten(), reduction="sum",
            ).item() for inputs, targets in batches) / 40
        rng = torch.get_rng_state().clone()
        self.assertAlmostEqual(trainer.validate(), expected, places=6)
        self.assertFalse(trainer.model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        trainer.model.train()
        trainer.config.evaluation.steps = 1
        with torch.no_grad():
            trainer.model.eval()
            expected = torch.nn.functional.cross_entropy(trainer.model(batches[0][0]).flatten(0, 1), batches[0][1].flatten()).item()
            trainer.model.train()
        self.assertAlmostEqual(trainer.validate(), expected, places=6)
        self.assertTrue(trainer.model.training)

    def test_signal_finishes_update_saves_and_restores_signal_handlers(self):
        trainer = self.trainer()
        handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        forward = trainer.train_model.forward

        def stop_during_forward(inputs):
            signal.raise_signal(signal.SIGTERM)
            with self.assertRaisesRegex(RuntimeError, "completed optimizer"):
                trainer.checkpoints.save(trainer)
            return forward(inputs)

        with patch.object(trainer.train_model, "forward", side_effect=stop_during_forward):
            trainer.train()
        self.assertEqual(trainer.global_step, 1)
        self.assertEqual(trainer.optimizer.schedulers["adamw"].last_epoch, 1)
        self.assertEqual(CheckpointManager.read(trainer.checkpoints.latest_checkpoint)["trainer_state"]["progress"]["next_batch"], 2)
        self.assertEqual({signum: signal.getsignal(signum) for signum in handlers}, handlers)
        self.assertFalse((self.root / "results.csv").exists())

    def test_failed_accumulation_or_schedule_cannot_publish_partial_state(self):
        for failure in ("loss", "gradients", "schedule"):
            with self.subTest(failure=failure):
                trainer = self.trainer(self.config().with_overrides({"training.max_steps": 2}))
                path = trainer.save_checkpoint()
                before = path.read_bytes()
                if failure == "loss":
                    def nonfinite(inputs):
                        return torch.full((*inputs.shape, 32), float("nan"), requires_grad=True)
                    mocked = patch.object(trainer.train_model, "forward", side_effect=nonfinite)
                    error = FloatingPointError
                elif failure == "gradients":
                    parameter = trainer.model.embeddings.weight
                    mocked = parameter.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
                    error = RuntimeError
                else:
                    mocked = patch.object(trainer.optimizer, "step_schedule", side_effect=RuntimeError("scheduler failed"))
                    error = RuntimeError
                with mocked, self.assertRaises(error):
                    trainer.train()
                with self.assertRaisesRegex(RuntimeError, "completed optimizer"):
                    trainer.save_checkpoint()
                with self.assertRaisesRegex(RuntimeError, "completed optimizer"):
                    trainer.train()
                self.assertEqual(path.read_bytes(), before)
                trainer.load_checkpoint(path)
                trainer.train()
                self.assertEqual(trainer.global_step, 2)

    def test_compiled_forward_marks_each_accumulation_microbatch(self):
        with patch("components.generic_trainer.torch.compile", side_effect=lambda model, **kwargs: model):
            trainer = self.trainer(self.config().with_overrides({
                "runtime.compile_model": True, "training.max_steps": 2,
            }))
        with patch("torch.compiler.cudagraph_mark_step_begin") as mark:
            trainer.train()
        self.assertEqual(mark.call_count, 4)


@unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "requires BF16 CUDA")
class TrainerCudaTests(unittest.TestCase):
    artifact = TrainerTests.artifact
    config = TrainerTests.config
    update = TrainerTests.update
    assert_state_equal = TrainerTests.assert_state_equal
    stop_at = TrainerTests.stop_at

    def test_cuda_actual_loop_resume_eager_and_compiled(self):
        with tempfile.TemporaryDirectory() as directory:
            self.root = Path(directory)
            self.artifact("train", 129)
            self.artifact("validation", 65)
            for compiled in (False, True):
                with self.subTest(compiled=compiled):
                    config = self.config("muon_adamw").with_overrides({
                        "name": f"cuda-{compiled}", "training.max_steps": 5,
                        "evaluation.interval_steps": 2, "optimizer.adamw.fused": True,
                        "runtime.compile_model": compiled, "runtime.compile_mode": "reduce-overhead",
                    })
                    with PretrainingTrainer(config) as baseline:
                        baseline.train()
                        expected = copy.deepcopy(baseline.state_dict())
                    with PretrainingTrainer(config) as interrupted:
                        with self.stop_at(interrupted, 3):
                            interrupted.train()
                        path = interrupted.checkpoints.latest_checkpoint
                    with PretrainingTrainer.from_checkpoint(path) as restored:
                        restored.train()
                        self.assert_state_equal(expected, restored.state_dict())
                        self.assertTrue(all(p.dtype == torch.float32 for p in restored.model.parameters()))

    def test_cuda_checkpoint_restores_rng_optimizer_buffers_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            self.root = Path(directory)
            self.artifact("train", 129)
            self.artifact("validation", 65)
            config = self.config("muon_adamw").with_overrides({"optimizer.adamw.fused": True})
            with PretrainingTrainer(config) as trainer:
                self.update(trainer)
                path = trainer.save_checkpoint()
                self.update(trainer)
                expected = copy.deepcopy(trainer.state_dict())
                with PretrainingTrainer.from_checkpoint(path) as restored:
                    self.update(restored)
                    self.assert_state_equal(expected, restored.state_dict())
                    self.assertTrue(all(p.dtype == torch.float32 for p in restored.model.parameters()))


if __name__ == "__main__":
    unittest.main()
