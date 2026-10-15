"""Generation from saved directories and migration of legacy weights."""

from __future__ import annotations

import io
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from components.checkpoint import CheckpointManager, model_state_from_checkpoint
from config import CheckpointConfig, ModelConfig, PretrainingConfig
from config.serialization import dump_toml
from model import DecoderTransformer
from pavullmo import generate, migrate_checkpoint_config as migration


def small_config(**overrides):
    return ModelConfig(vocab_size=32, n_blocks=1, embed_dim=16, attn_heads=4,
                       ffn_dim=32, seq_len=8, dropout=0.2, **overrides)


def save_legacy(path, model, *, omitted=(), string_values=False):
    values = {name.upper(): value for name, value in model.config.to_dict().items()
              if name not in omitted}
    if string_values:
        values = {name: str(value) for name, value in values.items()}
    # Training settings must not be mistaken for model settings or carried into
    # a fabricated PretrainingConfig for resume.
    values["LR"] = 0.001
    torch.save({"format_version": 1, "hyperparameters": values,
                "model_state_dict": model.state_dict()}, path)


class SavedDirectoryTests(unittest.TestCase):
    def test_current_full_and_weights_only_checkpoints_preserve_logits(self):
        for full_state in (True, False):
            with self.subTest(full_state=full_state), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                original = DecoderTransformer(small_config(qk_norm=True)).eval()
                run = PretrainingConfig(name="test", model=original.config)
                run.save(root / "config.toml")
                trainer = SimpleNamespace(
                    config=run, global_step=2, source_checkpoint="",
                    state_dict=lambda: {"model": original.state_dict(), "extra_training_state": 42},
                )
                manager = CheckpointManager(CheckpointConfig(save_training_state=full_state), root / "checkpoints")
                path = manager.save(trainer)
                loaded, vocab, context = generate.load_model(root, torch.device("cpu"))
                self.assertEqual((vocab, context), (32, 8))
                self.assertFalse(loaded.training)
                self.assertEqual(next(loaded.parameters()).dtype, torch.float32)
                inputs = torch.tensor([[1, 5, 7]])
                torch.testing.assert_close(loaded(inputs), original(inputs), rtol=0, atol=0)
                self.assertEqual(generate.find_checkpoint(root), path)

    def test_latest_save_order_explicit_selection_and_embedded_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            paths = [checkpoints / "step_00000002.pt", checkpoints / "step_00000010.pt"]
            for index, path in enumerate(paths):
                model = DecoderTransformer(small_config(rope_base=100.0 + index))
                torch.save({"format_version": 1, "full_state": False,
                            "config_toml": PretrainingConfig(name="saved", model=model.config).to_toml(),
                            "trainer_state": {"model": model.state_dict()}}, path)
                os.utime(path, ns=(100 - index, 100 - index))
            # An unrelated file is never considered a managed save.
            (checkpoints / "step_latest.pt").write_text("unrelated")
            (root / "config.toml").write_text("invalid config: later edited")
            self.assertEqual(generate.find_checkpoint(root), paths[0])
            latest, _, _ = generate.load_model(root, torch.device("cpu"))
            self.assertEqual(latest.config.rope_base, 100.0)
            chosen, _, _ = generate.load_model(root, torch.device("cpu"), checkpoint=Path("checkpoints") / paths[1].name)
            self.assertEqual(chosen.config.rope_base, 101.0)
            self.assertEqual(generate.find_checkpoint(root, paths[1]), paths[1])
            self.assertEqual(generate.find_checkpoint(checkpoints), paths[0])

    def test_missing_paths_and_ambiguous_legacy_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(NotADirectoryError):
                generate.find_checkpoint(root / "missing")
            with self.assertRaises(FileNotFoundError):
                generate.find_checkpoint(root)
            for name in ("one.pt", "two.pt"):
                (root / name).touch()
            with self.assertRaisesRegex(ValueError, "--checkpoint"):
                generate.find_checkpoint(root)
            self.assertEqual(generate.find_checkpoint(root, Path("two.pt")), root / "two.pt")
            with self.assertRaises(FileNotFoundError):
                generate.find_checkpoint(root, Path("missing.pt"))

    def test_invalid_checkpoint_payloads_fail(self):
        for payload in ([], {}, {"model_state_dict": {}}, {"model_state_dict": {"weight": "not a tensor"}},
                        {"trainer_state": {"model": {"weight": torch.ones(1)}}, "format_version": 100}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                model_state_from_checkpoint(payload)


class LegacyMigrationTests(unittest.TestCase):
    def test_migration_and_loading_preserve_logits_across_layouts(self):
        layouts = [
            {},
            {"split_qkv_projections": True, "qk_norm": True},
            {"canon_layers": True, "attn_bias": True},
            {"split_qkv_projections": True, "qk_norm": True, "canon_layers": True,
             "canon_kernel_size": 5, "attn_bias": True, "initialization": "gpt_scaled"},
        ]
        for settings in layouts:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                checkpoint = root / "old.pt"
                original = DecoderTransformer(small_config(**settings)).eval()
                save_legacy(checkpoint, original,
                            omitted=("split_qkv_projections", "canon_layers", "canon_kernel_size", "attn_bias", "qk_norm"),
                            string_values=True)
                before = checkpoint.read_bytes()
                config_path = migration.migrate_checkpoint_config(checkpoint)
                self.assertEqual(checkpoint.read_bytes(), before)
                data = tomllib.loads(config_path.read_text())
                self.assertEqual(set(data), {"model"})
                self.assertEqual(ModelConfig.from_dict(data["model"]), original.config)
                loaded, vocab, context = generate.load_model(root, torch.device("cpu"))
                self.assertEqual((vocab, context), (32, 8))
                inputs = torch.tensor([[1, 5, 7]])
                torch.testing.assert_close(loaded(inputs), original(inputs), rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    CheckpointManager.read(checkpoint)

    def test_string_false_flags_and_overwrite_control(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "old.pt"
            model = DecoderTransformer(small_config())
            save_legacy(checkpoint, model, string_values=True)
            migration.migrate_checkpoint_config(checkpoint)
            loaded, _, _ = generate.load_model(root, torch.device("cpu"))
            self.assertFalse(loaded.config.qk_norm)
            self.assertFalse(loaded.config.split_qkv_projections)
            self.assertFalse(loaded.config.canon_layers)
            path = root / "config.toml"
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                migration.migrate_checkpoint_config(checkpoint)
            self.assertEqual(path.read_bytes(), before)
            migration.migrate_checkpoint_config(checkpoint, force=True)
            self.assertEqual(path.read_bytes(), before)

    def test_missing_metadata_and_incompatible_weights_never_publish_config(self):
        for omitted, altered in (("attn_heads", False), (None, True)):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                checkpoint = root / "old.pt"
                model = DecoderTransformer(small_config())
                save_legacy(checkpoint, model, omitted=(omitted,) if omitted else ())
                if altered:
                    payload = torch.load(checkpoint, weights_only=True)
                    payload["hyperparameters"]["SPLIT_QKV_PROJECTIONS"] = True
                    torch.save(payload, checkpoint)
                with self.assertRaises((ValueError, RuntimeError)):
                    migration.migrate_checkpoint_config(checkpoint)
                self.assertFalse((root / "config.toml").exists())

    def test_old_checkpoint_requires_migration_and_strict_sidecar_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "old.pt"
            save_legacy(checkpoint, DecoderTransformer(small_config()))
            with self.assertRaisesRegex(FileNotFoundError, "migrate_checkpoint_config"):
                generate.load_model(root, torch.device("cpu"))
            (root / "config.toml").write_text('name="partial"\n')
            with self.assertRaisesRegex(ValueError, r"\[model\]"):
                generate.load_model(root, torch.device("cpu"))
            migration.migrate_checkpoint_config(checkpoint, force=True)
            config = small_config().to_dict()
            config["vocab_size"] = 99
            (root / "config.toml").write_text(dump_toml({"model": config}))
            with self.assertRaises(RuntimeError):
                generate.load_model(root, torch.device("cpu"))


class GenerationTests(unittest.TestCase):
    def test_cli_directory_and_migration_arguments(self):
        args = generate.parse_args(["saved", "tokenizer.model", "--checkpoint", "old.pt"])
        self.assertEqual(args.model_dir, Path("saved"))
        self.assertEqual(args.tokenizer, Path("tokenizer.model"))
        self.assertEqual(args.checkpoint, Path("old.pt"))
        self.assertTrue(migration.parse_args(["old.pt", "--force"]).force)

    def test_generation_stops_at_eos_and_checks_context(self):
        tokenizer = SimpleNamespace(
            encode=lambda prompt, out_type: [5] if prompt == "short" else list(range(8)),
            bos_id=lambda: 1, eos_id=lambda: 2, unk_id=lambda: 0, pad_id=lambda: 3,
            decode=lambda ids: str(ids),
        )
        model = DecoderTransformer(small_config()).eval()
        with patch.object(generate, "sample_next_token", side_effect=[4, 2]) as sample:
            self.assertEqual(generate.generate(model, tokenizer, "short", torch.device("cpu"), 8), "[4]")
            self.assertEqual(sample.call_count, 2)
            self.assertEqual(sample.call_args.args[1], {0, 1, 3})
        with self.assertRaisesRegex(ValueError, "shorten the prompt"):
            generate.generate(model, tokenizer, "long", torch.device("cpu"), 8)

    def test_interactive_main_loads_directory_and_validates_tokenizer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_legacy(root / "old.pt", DecoderTransformer(small_config()))
            migration.migrate_checkpoint_config(root / "old.pt")
            tokenizer = SimpleNamespace(vocab_size=lambda: 32, bos_id=lambda: 1, eos_id=lambda: 2)
            with patch("sys.argv", ["generate", str(root), "tokenizer.model"]), \
                 patch.object(generate.torch.cuda, "is_available", return_value=False), \
                 patch.object(generate.spm, "SentencePieceProcessor", return_value=tokenizer), \
                 patch("builtins.input", side_effect=EOFError), patch("sys.stdout", new_callable=io.StringIO):
                generate.main()
                tokenizer.vocab_size = lambda: 31
                with self.assertRaisesRegex(ValueError, "does not match"):
                    generate.main()


if __name__ == "__main__":
    unittest.main()
