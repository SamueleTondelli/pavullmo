"""Colab session wrapper, pretrain launcher, archive merge, and TensorBoard sync tests."""

from __future__ import annotations

import csv
import io
import json
import os
import signal
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config import PretrainingConfig
from pavullmo import colab_session, pretrain, pretrain_colab


def _write_tar_gz(archive_path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class ColabHelperTests(unittest.TestCase):
    def test_colab_config_preset_composes_expected_runtime_paths(self):
        config = pretrain.prepare_config(
            "10m_tuned_config.toml", None, {"name": "colab_trial"}, extends=["colab_config"],
        )
        self.assertEqual(config.runtime.dataset_dir, "/content/datasets")
        self.assertEqual(config.runtime.output_dir, "/content/outputs/runs")
        self.assertEqual(config.runtime.runs_csv, "/content/outputs/pretrain_runs.csv")
        self.assertEqual(config.model.n_blocks, 8)

    def test_parse_datasets_normalizes_deduplicates_and_rejects_unsafe_paths(self):
        parsed = colab_session.parse_datasets([
            " train_balanced_100m/ ; validation ",
            "train_balanced_100m; nested/artifact/ ",
        ])
        self.assertEqual(parsed, ["train_balanced_100m", "validation", "nested/artifact"])

        for bad in ("", "a;;b", "/abs/path", "../escape", "sub/../escape"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                colab_session.parse_datasets([bad])

    def test_validate_gpu_allows_bf16_gpus_and_rejects_t4_or_unknown(self):
        self.assertEqual(colab_session.validate_gpu("l4"), "L4")
        self.assertEqual(colab_session.validate_gpu("A100"), "A100")
        self.assertEqual(colab_session.validate_gpu("H100"), "H100")
        self.assertEqual(colab_session.validate_gpu("g4"), "G4")
        with self.assertRaisesRegex(ValueError, "BF16"):
            colab_session.validate_gpu("t4")
        with self.assertRaises(ValueError):
            colab_session.validate_gpu("V100")

    def test_ssh_config_uses_short_control_socket_and_optional_flags(self):
        sock = colab_session.control_socket_path("very-long-session-name-" + "x" * 120)
        self.assertLess(len(str(sock)), 108)

        with tempfile.TemporaryDirectory() as folder:
            priv_key = Path(folder) / "id_ed25519"
            pub_key = Path(folder) / "id_ed25519.pub"
            priv_key.write_text("private")
            pub_key.write_text("public")
            with patch.object(colab_session.shutil, "which", return_value="/usr/local/bin/colab"):
                text = colab_session.build_ssh_config_text(
                    "sweep-1", colab_auth="adc", identity=str(pub_key),
                )
        self.assertIn("Host colab-runtime", text)
        self.assertIn(
            "ProxyCommand /usr/local/bin/colab --auth adc ssh --proxy-mode -s sweep-1 --identity",
            text,
        )
        self.assertIn(f"IdentityFile {priv_key}", text)
        self.assertIn("ControlMaster auto", text)
        self.assertIn(f"ControlPath {colab_session.control_socket_path('sweep-1')}", text)


class ArchiveAndCsvMergeTests(unittest.TestCase):
    def test_extract_and_merge_preserves_existing_runs_and_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local_runs = root / "artifacts/runs"
            older = local_runs / "local_run/tensorboard"
            older.mkdir(parents=True)
            (older / "events.out.tfevents.old").write_bytes(b"old-local")

            pre_synced = local_runs / "sweep_1/tensorboard"
            pre_synced.mkdir(parents=True)
            (pre_synced / "events.out.tfevents.1").write_bytes(b"partial")

            archive = root / "runs.tar.gz"
            _write_tar_gz(archive, {
                "sweep_1/config.toml": b'name = "sweep_1"\n',
                "sweep_1/tensorboard/events.out.tfevents.1": b"complete-events",
                "sweep_1/checkpoints/step_00000010.pt": b"ckpt-1",
                "sweep_2/config.toml": b'name = "sweep_2"\n',
                "sweep_2/checkpoints/step_00000020.pt": b"ckpt-2",
            })
            extracted = root / "extracted"
            colab_session.extract_tar_archive(archive, extracted)
            runs_count, files_count = colab_session.merge_directory_tree(extracted, local_runs)

            self.assertEqual((runs_count, files_count), (2, 5))
            self.assertEqual((older / "events.out.tfevents.old").read_bytes(), b"old-local")
            self.assertEqual((pre_synced / "events.out.tfevents.1").read_bytes(), b"complete-events")
            self.assertEqual(
                (local_runs / "sweep_1/checkpoints/step_00000010.pt").read_bytes(),
                b"ckpt-1",
            )
            self.assertEqual(
                (local_runs / "sweep_2/checkpoints/step_00000020.pt").read_bytes(),
                b"ckpt-2",
            )

            unsafe_archive = root / "unsafe.tar.gz"
            _write_tar_gz(unsafe_archive, {"../escaped.txt": b"evil"})
            with self.assertRaises(tarfile.TarError):
                colab_session.extract_tar_archive(unsafe_archive, root / "safe_dest")

    def test_merge_runs_csv_remaps_paths_and_deduplicates_compact_and_multiline_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local_runs = root / "artifacts/runs"
            local_csv = root / "artifacts/results/pretrain_runs.csv"

            cfg1 = PretrainingConfig.from_dict({"name": "sweep_1"})
            cfg2 = PretrainingConfig.from_dict({"name": "sweep_2"})
            cfg3 = PretrainingConfig.from_dict({"name": "sweep_3"})
            remote_model_1 = "/content/outputs/runs/sweep_1/checkpoints/step_00000010.pt"
            remote_model_2 = "/content/outputs/runs/sweep_2/checkpoints/step_00000010.pt"
            unrelated_model = "/elsewhere/custom.pt"
            local_model_1 = str(local_runs.resolve() / "sweep_1/checkpoints/step_00000010.pt")

            # Seed local CSV with an older multiline to_toml() row for sweep_1.
            local_csv.parent.mkdir(parents=True)
            with local_csv.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=colab_session.RUNS_CSV_FIELDS)
                writer.writeheader()
                writer.writerow({
                    "experiment_name": "sweep_1",
                    "dataset_variant": "train_10m",
                    "train_loss": "3.1",
                    "validation_loss": "3.2",
                    "train_script": "components.pretraining_trainer.PretrainingTrainer",
                    "model_path": local_model_1,
                    "config": cfg1.to_toml(),
                })

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=colab_session.RUNS_CSV_FIELDS)
            writer.writeheader()
            for name, model_path, cfg in (
                ("sweep_1", remote_model_1, cfg1),
                ("sweep_2", remote_model_2, cfg2),
                ("sweep_2", remote_model_2, cfg2),
                ("sweep_3", unrelated_model, cfg3),
            ):
                writer.writerow({
                    "experiment_name": name,
                    "dataset_variant": "train_10m",
                    "train_loss": "2.5",
                    "validation_loss": "2.6",
                    "train_script": "components.pretraining_trainer.PretrainingTrainer",
                    "model_path": model_path,
                    "config": cfg.to_string(),
                })

            added = colab_session.merge_runs_csv(
                buf.getvalue(),
                local_csv,
                remote_output_dir="/content/outputs/runs",
                local_output_dir=local_runs,
            )
            self.assertEqual(added, 2)

            with local_csv.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r["experiment_name"] for r in rows], ["sweep_1", "sweep_2", "sweep_3"])
            self.assertEqual(
                rows[1]["model_path"],
                str(local_runs.resolve() / "sweep_2/checkpoints/step_00000010.pt"),
            )
            self.assertEqual(rows[2]["model_path"], unrelated_model)

            # Merging into a brand new CSV creates the header automatically.
            fresh_csv = root / "fresh/pretrain_runs.csv"
            added_fresh = colab_session.merge_runs_csv(
                buf.getvalue(),
                fresh_csv,
                remote_output_dir="/content/outputs/runs",
                local_output_dir=local_runs,
            )
            self.assertEqual(added_fresh, 3)

            # Mismatched local CSV header raises ValueError.
            bad_csv = root / "bad.csv"
            bad_csv.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "columns differ"):
                colab_session.merge_runs_csv(
                    buf.getvalue(),
                    bad_csv,
                    remote_output_dir="/content/outputs/runs",
                    local_output_dir=local_runs,
                )


class LiveTensorBoardSyncTests(unittest.TestCase):
    def test_live_tensorboard_sync_polls_retries_errors_and_stops_on_exit(self):
        from threading import Event

        synced = Event()
        calls = 0

        def fake_sync(_ssh_config, *, remote_output_dir, local_output_dir):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient")
            synced.set()

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(colab_session, "sync_tensorboard_events_once", side_effect=fake_sync), \
                patch.object(colab_session.logger, "exception") as logged:
            ssh_cfg = Path(folder) / "ssh.cfg"
            with self.assertRaisesRegex(RuntimeError, "sweep aborted"):
                with colab_session.live_tensorboard_sync(
                    ssh_cfg,
                    remote_output_dir="/content/outputs/runs",
                    local_output_dir=Path(folder) / "runs",
                    interval_seconds=0.01,
                ):
                    self.assertTrue(synced.wait(2), "TensorBoard sync worker did not recover")
                    raise RuntimeError("sweep aborted")
            logged.assert_called_once()
            count_after_exit = calls
            synced.clear()
            self.assertFalse(synced.wait(0.05))
            self.assertEqual(calls, count_after_exit)

    def test_sync_tensorboard_events_once_invokes_filtered_rsync(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(
                    colab_session.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stderr=""),
                ) as run:
            local_runs = Path(folder) / "runs"
            ssh_cfg = Path(folder) / "ssh.cfg"
            colab_session.sync_tensorboard_events_once(
                ssh_cfg,
                remote_output_dir="/content/outputs/runs",
                local_output_dir=local_runs,
            )
            cmd = run.call_args.args[0]
            self.assertEqual(cmd[0], "rsync")
            self.assertIn("events.out.tfevents.*", cmd)
            self.assertIn("--prune-empty-dirs", cmd)
            self.assertEqual(cmd[-2], "colab-runtime:/content/outputs/runs/")
            self.assertEqual(cmd[-1], f"{local_runs}/")


class ColabSessionEntrypointTests(unittest.TestCase):
    def test_happy_path_provisions_downloads_and_stops_session(self):
        observed_env: dict[str, str] = {}

        def fake_run_command(cmd, *, env):
            observed_env.update(env)
            self.assertEqual(cmd, ["bash", "sweeps/lr.sh"])
            return 0

        with patch.object(colab_session, "create_session") as create, \
                patch.object(colab_session, "sync_workspace") as sync, \
                patch.object(colab_session, "ensure_hf_datasets") as ensure_ds, \
                patch.object(colab_session, "_run_managed_command", side_effect=fake_run_command), \
                patch.object(colab_session, "download_and_merge_outputs") as download, \
                patch.object(colab_session, "close_ssh_master") as close_master, \
                patch.object(colab_session, "stop_session") as stop:
            colab_session.main([
                "--session", "sweep-s1",
                "--gpu", "L4",
                "--hf-repo", "user/pavullmo-ds",
                "--dataset", "train_100m;validation",
                "--no-tensorboard-sync",
                "--", "bash", "sweeps/lr.sh",
            ])

        create.assert_called_once_with("sweep-s1", gpu="L4", high_mem=False, colab_auth="")
        sync.assert_called_once()
        self.assertEqual(ensure_ds.call_args.kwargs["datasets"], ["train_100m", "validation"])
        self.assertEqual(observed_env[colab_session.ENV_SESSION], "sweep-s1")
        self.assertEqual(observed_env[colab_session.ENV_MANAGED_SESSION], "1")
        self.assertEqual(observed_env[colab_session.ENV_HF_REPO], "user/pavullmo-ds")
        download.assert_called_once()
        close_master.assert_called_once()
        stop.assert_called_once_with("sweep-s1", colab_auth="")

    def test_failure_still_downloads_closes_master_and_stops_session(self):
        with patch.object(colab_session, "create_session"), \
                patch.object(colab_session, "sync_workspace"), \
                patch.object(colab_session, "_run_managed_command", return_value=7), \
                patch.object(colab_session, "download_and_merge_outputs") as download, \
                patch.object(colab_session, "close_ssh_master") as close_master, \
                patch.object(colab_session, "stop_session") as stop, \
                self.assertRaises(SystemExit) as error:
            colab_session.main(["--session", "s2", "--no-tensorboard-sync", "--", "false"])

        self.assertEqual(error.exception.code, 7)
        download.assert_called_once()
        close_master.assert_called_once()
        stop.assert_called_once_with("s2", colab_auth="")

    def test_keep_reuse_and_no_download_flags(self):
        with patch.object(colab_session, "create_session") as create, \
                patch.object(colab_session, "sync_workspace"), \
                patch.object(colab_session, "_run_managed_command", return_value=0), \
                patch.object(colab_session, "download_and_merge_outputs") as download, \
                patch.object(colab_session, "close_ssh_master"), \
                patch.object(colab_session, "stop_session") as stop:
            colab_session.main([
                "--session", "s3", "--reuse", "--no-download", "--no-tensorboard-sync", "--", "true",
            ])
        create.assert_not_called()
        download.assert_not_called()
        stop.assert_not_called()

        with patch.object(colab_session, "create_session") as create, \
                patch.object(colab_session, "sync_workspace"), \
                patch.object(colab_session, "download_and_merge_outputs"), \
                patch.object(colab_session, "close_ssh_master"), \
                patch.object(colab_session, "stop_session") as stop:
            colab_session.main(["--session", "s4", "--keep", "--no-tensorboard-sync"])
        create.assert_called_once()
        stop.assert_not_called()

    def test_invalid_arguments_fail_before_creating_session(self):
        for args in (
            ["--gpu", "T4", "--", "true"],
            ["--dataset", "train_10m", "--", "true"],
            ["--reuse", "--", "true"],
            ["--session", "s5"],
        ):
            with self.subTest(args=args), \
                    patch("sys.stderr", new=io.StringIO()), \
                    patch.object(colab_session, "create_session") as create, \
                    self.assertRaises(SystemExit) as error:
                colab_session.main(args)
            self.assertEqual(error.exception.code, 2)
            create.assert_not_called()


class PretrainColabEntrypointTests(unittest.TestCase):
    def test_validation_and_missing_session_fail_fast_without_ssh(self):
        with patch.dict(os.environ, {}, clear=True):
            for args in (
                ["--config", "missing.toml", "--session", "s1"],
                ["--config", "10m_tuned_config", "--override", "bad", "--session", "s1"],
                ["--config", "10m_tuned_config", "--override", "name=trial"],
            ):
                with self.subTest(args=args), \
                        patch("sys.stderr", new=io.StringIO()), \
                        patch.object(pretrain_colab, "run_remote_pretrain") as remote, \
                        self.assertRaises(SystemExit) as error:
                    pretrain_colab.main(args)
                self.assertEqual(error.exception.code, 2)
                remote.assert_not_called()

    def test_managed_session_skips_sync_and_stages_relative_datasets_only(self):
        fake_proc = MagicMock()
        fake_proc.wait.return_value = 0

        with tempfile.TemporaryDirectory() as folder:
            ssh_cfg = Path(folder) / "managed.sshconfig"
            ssh_cfg.write_text("Host colab-runtime\n")
            env = {
                colab_session.ENV_SESSION: "sweep-1",
                colab_session.ENV_MANAGED_SESSION: "1",
                colab_session.ENV_SSH_CONFIG: str(ssh_cfg),
                colab_session.ENV_HF_REPO: "user/ds",
            }
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(pretrain_colab, "sync_workspace") as sync, \
                    patch.object(pretrain_colab, "ensure_hf_datasets") as ensure_ds, \
                    patch.object(pretrain_colab.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as sub_run, \
                    patch.object(pretrain_colab.subprocess, "Popen", return_value=fake_proc) as popen:
                pretrain_colab.main([
                    "--config", "10m_tuned_config",
                    "--extends", "colab_config",
                    "--override", "name=trial1;train_dataset.path=train_100m;validation_dataset.path=/custom/val",
                ])

        sync.assert_not_called()
        ensure_ds.assert_called_once()
        self.assertEqual(ensure_ds.call_args.kwargs["datasets"], ["train_100m"])
        self.assertEqual(ensure_ds.call_args.kwargs["remote_dataset_dir"], "/content/datasets")
        upload_call = sub_run.call_args_list[0]
        self.assertIn("cat > /tmp/pavullmo-config-", upload_call.args[0][-1])
        uploaded_config = PretrainingConfig.from_toml(upload_call.kwargs["input"])
        self.assertEqual(uploaded_config.name, "trial1")
        remote_cmd = popen.call_args.args[0][-1]
        self.assertIn("TORCHINDUCTOR_CACHE_DIR=/content/cache/torchinductor", remote_cmd)
        self.assertIn("-m pavullmo.pretrain --config /tmp/pavullmo-config-", remote_cmd)

    def test_standalone_session_syncs_workspace_and_resume_forwards_overrides(self):
        fake_proc = MagicMock()
        fake_proc.wait.return_value = 0

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(pretrain_colab, "sync_workspace") as sync, \
                patch.object(pretrain_colab, "ensure_hf_datasets") as ensure_ds, \
                patch.object(pretrain_colab.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                patch.object(pretrain_colab.subprocess, "Popen", return_value=fake_proc) as popen:
            pretrain_colab.main([
                "--resume", "/content/outputs/runs/r1/checkpoints/step_00000010.pt",
                "--override", "name=r1_cont;training.max_steps=20",
                "--session", "manual-s1",
            ])

        sync.assert_called_once()
        self.assertFalse(sync.call_args.kwargs["run_check"])
        ensure_ds.assert_not_called()
        remote_cmd = popen.call_args.args[0][-1]
        self.assertIn("--resume /content/outputs/runs/r1/checkpoints/step_00000010.pt", remote_cmd)
        self.assertIn("--override 'name=r1_cont;training.max_steps=20'", remote_cmd)

    def test_sigint_forwards_kill_int_to_remote_pid(self):
        fake_proc = MagicMock()

        def wait_with_interrupt():
            os.kill(os.getpid(), signal.SIGINT)
            return 0

        fake_proc.wait.side_effect = wait_with_interrupt

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(pretrain_colab, "sync_workspace"), \
                patch.object(pretrain_colab, "_interrupt_remote_training") as interrupt_remote, \
                patch.object(pretrain_colab.subprocess, "run", return_value=SimpleNamespace(returncode=0)), \
                patch.object(pretrain_colab.subprocess, "Popen", return_value=fake_proc):
            pretrain_colab.main([
                "--config", "10m_tuned_config",
                "--extends", "colab_config",
                "--override", "name=interrupt_test",
                "--session", "manual-s2",
            ])

        interrupt_remote.assert_called_once()


if __name__ == "__main__":
    unittest.main()
