"""TOML presets, CLI parsing, and provider wrappers without submitting jobs."""

from __future__ import annotations

import csv
import io
import tempfile
import tomllib
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from config import PretrainingConfig
from pavullmo import pretrain

import test_trainer as fixtures

ROOT = Path(__file__).resolve().parents[1]


class OverrideTests(unittest.TestCase):
    def test_semicolon_extends_and_bundled_preset_names(self):
        self.assertEqual(pretrain.parse_extends(["modal_config;second.toml", " third; "]),
                         ["modal_config", "second.toml", "third"])
        config = pretrain.prepare_config("10m_tuned_config.toml", None, {"name": "trial"},
                                      extends=["modal_config"])
        self.assertEqual(config.model.n_blocks, 8)
        self.assertEqual(config.model.vocab_size, 16384)
        self.assertEqual(config.runtime.dataset_dir, "/datasets")
        self.assertEqual(config.runtime.output_dir, "/outputs/runs")
        self.assertEqual(config.runtime.runs_csv, "/outputs/pretrain_runs.csv")
        self.assertNotIn("extends", config.to_dict())

    def test_appended_presets_resolve_beside_config_and_preserve_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "base.toml").write_text('[runtime]\noutput_dir="/base"\nruns_csv="/base.csv"\n')
            (root / "first.toml").write_text('[runtime]\noutput_dir="/first"\nruns_csv="/first.csv"\n')
            (root / "second.toml").write_text('[runtime]\noutput_dir="/second"\nruns_csv="/second.csv"\n')
            source = 'extends=["base.toml"]\nname="trial"\n[runtime]\noutput_dir="/own"\n'
            path = root / "run.toml"
            path.write_text(source)
            config = pretrain.prepare_config(path, None, {}, extends=["first", "second.toml"])
            self.assertEqual(config.runtime.output_dir, "/own")
            self.assertEqual(config.runtime.runs_csv, "/second.csv")
            self.assertEqual(path.read_text(), source)
            config = pretrain.prepare_config(path, None, {"runtime.output_dir": "/cli"}, extends=["first"])
            self.assertEqual(config.runtime.output_dir, "/cli")
            with self.assertRaises(FileNotFoundError):
                pretrain.prepare_config(path, None, {}, extends=["missing"])
        with self.assertRaisesRegex(ValueError, "--extends requires --config"):
            pretrain.prepare_config(None, "checkpoint.pt", {}, extends=["modal_config"])

    def test_typed_semicolon_values_and_repeated_flags(self):
        parsed = pretrain.parse_overrides([
            'name=trial;training.batch_size=16;optimizer.lr=1.5e-3;model.qk_norm=true;'
            'optimizer.muon.ns_coefficients=[3.4445, -4.775, 2.0315]',
            'training.batch_size=8;optimizer.kind="muon_adamw";',
        ])
        config = PretrainingConfig.from_dict({"name": "base"}).with_overrides(parsed)
        self.assertEqual(config.name, "trial")
        self.assertEqual(config.training.batch_size, 8)
        self.assertEqual(config.optimizer.lr, 0.0015)
        self.assertTrue(config.model.qk_norm)
        self.assertEqual(config.optimizer.kind, "muon_adamw")
        self.assertEqual(config.optimizer.muon.ns_coefficients, [3.4445, -4.775, 2.0315])

    def test_quoted_semicolons_equals_escapes_and_inline_tables(self):
        parsed = pretrain.parse_overrides([
            'logging.file="a;b=c.log";train_dataset.path=\'a;b\';'
            'optimizer.adamw={beta1=0.8, beta2=0.99};name="a\\\"b"',
        ])
        self.assertEqual(parsed["logging.file"], "a;b=c.log")
        self.assertEqual(parsed["train_dataset.path"], "a;b")
        self.assertEqual(parsed["optimizer.adamw"], {"beta1": 0.8, "beta2": 0.99})
        self.assertEqual(parsed["name"], 'a"b')
        self.assertEqual(pretrain.parse_overrides([" ; "]), {})

    def test_malformed_unknown_and_mistyped_overrides_fail(self):
        for text in ("name", "=value", "name=", 'name="unfinished', "optimizer.muon.ns_coefficients=[1;name=a"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                pretrain.parse_overrides([text])
        config = PretrainingConfig.from_dict({"name": "base"})
        for text in ("training.batch_szie=8", 'training.batch_size="8"', "model.qk_norm=True"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                config.with_overrides(pretrain.parse_overrides([text]))

    def test_all_presets_compose_with_expected_parameters_without_data_selectors(self):
        directory = ROOT / "src/pavullmo/configs"
        architecture = {"10m": (8, 256, 4, 704), "20m": (8, 384, 6, 1024),
                        "35m": (11, 448, 7, 1216), "50m": (16, 512, 8, 1344), "100m": (18, 640, 10, 1728)}
        self.assertFalse(list(directory.glob("*.sh")))
        for size, expected in architecture.items():
            config = PretrainingConfig.load(directory / f"{size}_config.toml", overrides={"name": "test"})
            self.assertEqual((config.model.n_blocks, config.model.embed_dim, config.model.attn_heads, config.model.ffn_dim), expected)
        for size, lr, adam_decay, muon_lr, muon_decay, beta2, decay in (
            ("10m", 0.0024, 0.05, 0.5, 0.05, 0.99, 0.4),
            ("20m", 0.0023, 0.1, 0.5, 0.1, 0.99, 0.4),
            ("35m", 0.00242, 0.05, 1.0, 0.0375, 0.99, 0.4),
            ("50m", 0.0021, 0.05, 1.0, 0.05, 0.9025, 0.2),
        ):
            config = PretrainingConfig.load(directory / f"{size}_tuned_config.toml", overrides={"name": "test"})
            self.assertEqual(config.model.vocab_size, 16384)
            self.assertEqual(config.model.initialization, "gpt_scaled")
            self.assertTrue(config.model.qk_norm)
            self.assertTrue(config.model.split_qkv_projections)
            self.assertEqual((config.optimizer.lr, config.optimizer.adamw.weight_decay,
                              config.optimizer.muon.lr_multiplier, config.optimizer.muon.weight_decay,
                              config.optimizer.adamw.beta2, config.optimizer.schedule.decay_fraction),
                             (lr, adam_decay, muon_lr, muon_decay, beta2, decay))
            self.assertEqual(config.optimizer.schedule.warmup_steps, 219)
        for path in directory.glob("*.toml"):
            data = tomllib.loads(path.read_text())
            self.assertNotIn("data", data)
            self.assertNotIn("train_dataset", data)
            self.assertNotIn("validation_dataset", data)
            self.assertNotIn("DATASET_PREFIX", path.read_text())
            self.assertNotIn("DATASET_VARIANT", path.read_text())
            with self.assertRaises(ValueError):
                PretrainingConfig.load(path)  # The run name must be supplied.


class LocalEntrypointTests(unittest.TestCase):
    setUp = fixtures.TrainerTests.setUp
    tearDown = fixtures.TrainerTests.tearDown
    artifact = fixtures.TrainerTests.artifact
    config = fixtures.TrainerTests.config
    trainer = fixtures.TrainerTests.trainer

    def test_local_main_runs_from_composed_config_and_records_cli_overrides(self):
        base = self.root / "base.toml"
        self.config().save(base)
        run = self.root / "run.toml"
        run.write_text('extends=["base.toml"]\nname="cli"\n')
        (self.root / "lr.toml").write_text('[optimizer]\nlr=0.003\n')
        (self.root / "duration.toml").write_text('[training]\nmax_steps=2\n')
        pretrain.main(["--config", str(run), "--extends", "lr;duration.toml",
                    "--override", "optimizer.lr=0.002"])
        with (self.root / "results.csv").open(newline="") as file:
            row = next(csv.DictReader(file))
        resolved = PretrainingConfig.from_string(row["config"])
        self.assertEqual((resolved.name, resolved.training.max_steps, resolved.optimizer.lr), ("cli", 2, 0.002))
        self.assertTrue(Path(row["model_path"]).is_file())

    def test_local_main_resumes_embedded_config_without_source_files(self):
        trainer = self.trainer(self.config().with_overrides({"training.max_steps": 2}))
        path = trainer.save_checkpoint()
        overrides = f"name=resumed;runtime.output_dir={self.root / 'relocated'}"
        pretrain.main(["--resume", str(path), "--override", overrides])
        self.assertTrue((self.root / "relocated/resumed/checkpoints/step_00000002.pt").exists())

    def test_cli_requires_exactly_one_source_and_rejects_bad_config_before_training(self):
        for arguments in ([], ["--config", "a", "--resume", "b"],
                          ["--config", "missing.toml"], ["--config", "missing.toml", "--override", "name"]):
            with self.subTest(arguments=arguments), patch("sys.stderr", new=io.StringIO()), \
                    patch.object(pretrain, "run_training") as execute, self.assertRaises(SystemExit) as error:
                pretrain.main(arguments)
            self.assertEqual(error.exception.code, 2)
            execute.assert_not_called()

    def test_checkpoint_persistence_hook_runs_after_each_published_checkpoint(self):
        config = self.config().with_overrides({
            "training.max_steps": 2, "checkpoint.interval_steps": 1, "checkpoint.keep_last": 2,
        })
        files = []

        def persist():
            files.append(sorted((self.root / "runs/test/checkpoints").glob("step_*.pt")))

        result = pretrain.run_training(config, checkpoint_saved=persist)
        self.assertTrue(result["completed"])
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0][-1].name, "step_00000001.pt")
        self.assertEqual(files[1][-1].name, "step_00000002.pt")

    def test_dashboard_context_uses_configured_output_root_on_fresh_and_resumed_runs(self):
        config = self.config().with_overrides({
            "training.max_steps": 2, "runtime.output_dir": str(self.root / "custom-runs"),
        })
        observed = []

        @contextmanager
        def dashboard(output_root):
            observed.append(output_root)
            yield

        result = pretrain.run_training(config, training_context=dashboard)
        pretrain.run_training(None, resume=result["checkpoint"],
                           overrides={"runtime.output_dir": str(self.root / "resumed-runs")},
                           training_context=dashboard)
        self.assertEqual(observed, [self.root / "custom-runs", self.root / "resumed-runs"])


try:
    from pavullmo import pretrain_modal
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    pretrain_modal = None


@unittest.skipIf(pretrain_modal is None, "requires the cloud extra")
class ModalEntrypointTests(unittest.TestCase):
    def test_live_outputs_are_published_without_checkpoints_and_worker_stops_on_failure(self):
        from threading import Event

        published = Event()
        volume = MagicMock()
        volume.commit.side_effect = lambda: published.set()
        with patch.object(pretrain_modal, "OUTPUT_COMMIT_SECONDS", 0.01):
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with pretrain_modal._publish_outputs(volume):
                    self.assertTrue(published.wait(2), "outputs were not published during training")
                    raise RuntimeError("training failed")
            count = volume.commit.call_count
            # The context joins the worker, so no later publication can race
            # the wrapper's final commit or continue after training has failed.
            self.assertGreaterEqual(count, 1)
            published.clear()
            self.assertFalse(published.wait(0.05))
            self.assertEqual(volume.commit.call_count, count)

    def test_live_output_publication_retries_after_transient_errors(self):
        from threading import Event

        published = Event()
        volume = MagicMock()
        def commit():
            if volume.commit.call_count == 1:
                raise RuntimeError("temporary")
            published.set()

        volume.commit.side_effect = commit
        with patch.object(pretrain_modal, "OUTPUT_COMMIT_SECONDS", 0.01), \
                patch.object(pretrain_modal.logger, "exception") as warning:
            with pretrain_modal._publish_outputs(volume):
                self.assertTrue(published.wait(2), "publication did not recover")
            warning.assert_called_once()

    def test_local_dispatch_serializes_full_resolved_config_and_infrastructure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.toml"
            path.write_text('name="cloud"\n[profiling]\nenabled=true\n')
            function = MagicMock()
            function.remote.return_value = {"name": "cloud", "global_step": 2, "total_steps": 2, "checkpoint": "/outputs/run.pt"}
            datasets, outputs = MagicMock(), MagicMock()
            with patch.object(pretrain_modal.train_remote, "with_options", return_value=function) as options, \
                    patch.object(pretrain_modal.modal.Volume, "from_name", side_effect=[datasets, outputs]) as volume, \
                    patch.object(pretrain_modal, "ensure_tensorboard") as dashboard:
                pretrain_modal.main(config=str(path), override="training.batch_size=16;optimizer.kind=muon_adamw",
                                 extends="modal_config", gpu="A10G", dataset_volume="tokens",
                                 output_volume="results", tensorboard=False)
            args = function.remote.call_args.args
            received = PretrainingConfig.from_toml(args[0])
            self.assertEqual(received.training.batch_size, 16)
            self.assertEqual(received.optimizer.kind, "muon_adamw")
            self.assertEqual(received.runtime.dataset_dir, "/datasets")
            self.assertEqual(received.runtime.output_dir, "/outputs/runs")
            self.assertEqual(received.runtime.runs_csv, "/outputs/pretrain_runs.csv")
            self.assertEqual(received.model_extra["profiling"], {"enabled": True})
            self.assertNotIn("extends", received.to_dict())
            self.assertEqual(args[1:], ("", {}, "results"))
            dashboard.assert_not_called()
            self.assertEqual(options.call_args.kwargs["gpu"], "A10G")
            datasets.with_mount_options.assert_called_once_with(read_only=True)
            self.assertEqual(options.call_args.kwargs["volumes"]["/outputs"], outputs)
            self.assertEqual(volume.call_args_list[-1].kwargs, {"create_if_missing": True})

    def test_remote_uses_authored_paths_and_commits_even_on_failure(self):
        volume = MagicMock()
        config = PretrainingConfig.from_dict({"name": "cloud", "runtime": {
            "dataset_dir": "/datasets/custom", "output_dir": "/outputs/custom-runs",
            "runs_csv": "/outputs/custom-results.csv",
        }})
        for fail in (False, True):
            with self.subTest(fail=fail), patch.object(pretrain_modal.modal.Volume, "from_name", return_value=volume), \
                    patch.object(pretrain_modal, "run_training", side_effect=RuntimeError("failed") if fail else None,
                                 return_value={"completed": True}) as execute:
                volume.reset_mock()
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        pretrain_modal.train_remote.local(config.to_toml(), "", {}, "results")
                else:
                    pretrain_modal.train_remote.local(config.to_toml(), "", {}, "results")
                kwargs = execute.call_args.kwargs
                self.assertEqual(execute.call_args.args[0].to_dict(), config.to_dict())
                self.assertEqual(kwargs["overrides"], {})
                self.assertEqual(set(kwargs), {"resume", "overrides", "checkpoint_saved"})
                self.assertEqual(kwargs["checkpoint_saved"], volume.commit)
                volume.commit.assert_called_once_with()

    def test_resume_checkpoint_stays_remote_and_cli_overrides_are_forwarded(self):
        with patch.object(pretrain_modal.modal.Volume, "from_name"), patch.object(pretrain_modal, "run_training") as execute:
            pretrain_modal.train_remote.local("", "runs/test/checkpoints/step_00000002.pt", {"name": "continued"}, "results")
        self.assertIsNone(execute.call_args.args[0])
        self.assertEqual(execute.call_args.kwargs["resume"], Path("runs/test/checkpoints/step_00000002.pt"))
        self.assertEqual(execute.call_args.kwargs["overrides"]["name"], "continued")

    def test_training_ensures_independent_dashboard_for_fresh_and_resumed_runs(self):
        function = MagicMock()
        function.remote.return_value = {"name": "cloud", "global_step": 2, "total_steps": 2,
                                        "checkpoint": "/outputs/run.pt"}
        for source in ({"config": "10m_tuned_config", "override": "name=cloud"},
                       {"resume": "/outputs/run.pt"}):
            with self.subTest(source=source), \
                    patch.object(pretrain_modal.train_remote, "with_options", return_value=function), \
                    patch.object(pretrain_modal.modal.Volume, "from_name"), \
                    patch.object(pretrain_modal, "ensure_tensorboard", return_value="https://dashboard") as dashboard:
                pretrain_modal.main(**source, output_volume="results")
                dashboard.assert_called_once_with("results")

    def test_remote_module_can_import_from_modal_root_without_local_image_operations(self):
        from importlib.util import module_from_spec, spec_from_file_location

        spec = spec_from_file_location("remote_pretrain_modal", ROOT / "src/pavullmo/pretrain_modal.py")
        module = module_from_spec(spec)
        module.__file__ = "/root/pretrain_modal.py"
        with patch.object(pretrain_modal.modal, "is_local", return_value=False), \
                patch.object(pretrain_modal.modal.Image, "uv_sync") as sync:
            spec.loader.exec_module(module)
        sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
