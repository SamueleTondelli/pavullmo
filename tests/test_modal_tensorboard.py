"""Dashboard lifetime and cross-container event refresh without cloud calls."""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from pavullmo import modal_start_tensorboard as dashboard
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    dashboard = None


@unittest.skipIf(dashboard is None, "requires the cloud extra")
class ModalTensorBoardTests(unittest.TestCase):
    def test_dashboard_can_be_registered_with_real_modal_handles_without_cloud_calls(self):
        service = dashboard._create_dashboard("results")
        self.assertEqual(service.name, dashboard.tensorboard_app_name("results"))
        self.assertEqual(set(service.registered_functions), {"tensorboard"})
        startup = service.registered_functions["tensorboard"].get_raw_f()
        self.assertEqual((startup.__module__, startup.__qualname__),
                         (dashboard.__name__, "_serve_tensorboard"))

    def test_existing_dashboard_is_reused_without_deployment(self):
        with patch.object(dashboard.modal.Function, "from_name") as lookup, \
                patch.object(dashboard, "_create_dashboard") as create:
            lookup.return_value.get_web_url.return_value = "https://existing"
            self.assertEqual(dashboard.ensure_tensorboard("results"), "https://existing")
            lookup.assert_called_once_with(dashboard.tensorboard_app_name("results"), "tensorboard")
            create.assert_not_called()

    def test_missing_dashboard_is_deployed_once_and_then_reused(self):
        with patch.object(dashboard.modal.Function, "from_name") as lookup, \
                patch.object(dashboard, "_create_dashboard") as create:
            lookup.return_value.get_web_url.side_effect = [
                dashboard.modal.exception.NotFoundError("missing"), "https://new", "https://new",
            ]
            self.assertEqual(dashboard.ensure_tensorboard("results"), "https://new")
            self.assertEqual(dashboard.ensure_tensorboard("results"), "https://new")
            create.assert_called_once_with("results")
            create.return_value.deploy.assert_called_once_with()

    def test_restart_redeploys_updated_code_at_the_existing_identity(self):
        with patch.object(dashboard.modal.Function, "from_name") as lookup, \
                patch.object(dashboard, "_create_dashboard") as create:
            lookup.return_value.get_web_url.return_value = "https://existing"
            self.assertEqual(dashboard.ensure_tensorboard("results", restart=True), "https://existing")
            create.assert_called_once_with("results")
            create.return_value.deploy.assert_called_once_with(strategy="recreate")

    def test_unrelated_lookup_errors_do_not_trigger_deployment(self):
        with patch.object(dashboard.modal.Function, "from_name", side_effect=RuntimeError("unavailable")), \
                patch.object(dashboard, "_create_dashboard") as create, \
                self.assertRaisesRegex(RuntimeError, "unavailable"):
            dashboard.ensure_tensorboard("results")
        create.assert_not_called()

    def test_volumes_have_distinct_bounded_app_names(self):
        self.assertNotEqual(dashboard.tensorboard_app_name("first"), dashboard.tensorboard_app_name("second"))
        self.assertLess(len(dashboard.tensorboard_app_name("long" * 100)), 64)

    def test_snapshot_refresh_loads_growing_events_and_discovers_later_runs(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        with tempfile.TemporaryDirectory() as folder:
            source, cache = Path(folder) / "outputs", Path(folder) / "cache"
            run = source / "custom-runs/first/tensorboard"
            volume = MagicMock()
            writer = EventFileWriter(str(run))
            self.addCleanup(writer.close)

            def record(step):
                writer.add_event(Event(step=step, summary=Summary(value=[
                    Summary.Value(tag="loss", simple_value=1.0 / step),
                ])))
                writer.flush()

            record(1)
            (run / "checkpoint.pt").write_bytes(b"never copied")
            dashboard._sync_events(volume, source, cache)
            reader = EventAccumulator(str(cache / run.relative_to(source)))
            reader.Reload()
            self.assertEqual([event.step for event in reader.Scalars("loss")], [1])
            record(2)
            later = source / "other-runs/second/tensorboard"
            later.mkdir(parents=True)
            (later / "events.out.tfevents.later").write_bytes(b"new run")
            dashboard._sync_events(volume, source, cache)
            reader.Reload()
            self.assertEqual([event.step for event in reader.Scalars("loss")], [1, 2])
            self.assertTrue((cache / later.relative_to(source) / "events.out.tfevents.later").is_file())
            self.assertEqual(list(cache.rglob("*.pt")), [])
            volume.reload.assert_called_with()

    def test_http_server_discovers_a_later_run_and_reads_appended_metrics(self):
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        with tempfile.TemporaryDirectory() as folder, socket.socket() as reservation:
            source, cache = Path(folder) / "outputs", Path(folder) / "cache"
            source.mkdir()
            cache.mkdir()
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
            reservation.close()
            base = f"http://127.0.0.1:{port}"
            with (Path(folder) / "server.log").open("w+") as log:
                process = subprocess.Popen(dashboard._tensorboard_command(cache, port), stdout=log, stderr=log)
                try:
                    def wait_for(path, expected):
                        deadline = time.monotonic() + 30
                        while time.monotonic() < deadline and process.poll() is None:
                            try:
                                with urllib.request.urlopen(base + path, timeout=1) as response:
                                    value = json.load(response)
                                if expected(value):
                                    return value
                            except (urllib.error.URLError, TimeoutError):
                                pass
                            time.sleep(0.1)
                        log.flush()
                        log.seek(0)
                        self.fail(f"TensorBoard did not update {path}: {log.read()}")

                    wait_for("/data/runs", lambda runs: not runs)
                    run_name = "runs/new/tensorboard"
                    writer = EventFileWriter(str(source / run_name))
                    try:
                        def record(step):
                            writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=[
                                Summary.Value(tag="loss", simple_value=1.0 / step),
                            ])))
                            writer.flush()
                            dashboard._sync_events(MagicMock(), source, cache)

                        record(1)
                        wait_for("/data/runs", lambda runs: run_name in runs)
                        scalars = "/data/plugin/scalars/scalars?" + urllib.parse.urlencode({
                            "run": run_name, "tag": "loss",
                        })
                        wait_for(scalars, lambda points: points and points[-1][1] == 1)
                        record(2)
                        wait_for(scalars, lambda points: points and points[-1][1] == 2)
                    finally:
                        writer.close()
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
