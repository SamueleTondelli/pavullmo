"""Start or reuse a persistent TensorBoard deployment for a Modal output Volume."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

OUTPUT_MOUNT = "/outputs"
EVENT_CACHE = "/tmp/tensorboard-events"
TENSORBOARD_PORT = 6006
REFRESH_SECONDS = 15
FUNCTION_NAME = "tensorboard"

image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    image = image.pip_install("tensorboard==2.21.0")

app = modal.App("pavullmo-tensorboard-launcher")


def tensorboard_app_name(output_volume: str) -> str:
    """Use a bounded, deterministic deployment name for each output Volume."""
    identity = hashlib.sha256(output_volume.encode()).hexdigest()[:16]
    return f"pavullmo-tensorboard-{identity}"


def _sync_events(volume: modal.Volume, source_root: Path, cache_root: Path) -> None:
    # TensorBoard keeps event files open. Read a local mirror so those handles
    # cannot prevent Volume.reload(), and never copy checkpoints or other outputs.
    print("TensorBoard: reloading output Volume", flush=True)
    volume.reload()
    cache_root.mkdir(parents=True, exist_ok=True)
    files = copied_bytes = 0
    for source in source_root.rglob("events.out.tfevents.*"):
        if not source.is_file():
            continue
        target = cache_root / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        offset = target.stat().st_size if target.exists() else 0
        if offset > source.stat().st_size:
            offset = 0
        with source.open("rb") as incoming, target.open("ab" if offset else "wb") as outgoing:
            incoming.seek(offset)
            shutil.copyfileobj(incoming, outgoing)
            copied_bytes += outgoing.tell() - offset
        files += 1
    print(f"TensorBoard: refreshed {files} event files, copied {copied_bytes} bytes", flush=True)


def _tensorboard_command(cache: Path, port: int) -> list[str]:
    # The native data server avoids scanning large histories in pure Python on
    # every reload. Open event handles remain on the local mirror, not the Volume.
    return [
        sys.executable, "-m", "tensorboard.main", "--logdir", str(cache),
        "--host", "0.0.0.0", "--port", str(port),
        "--reload_interval", "5", "--load_fast=true",
    ]


def _serve_tensorboard() -> None:
    """Modal web-server startup; its container remains warm between training runs."""
    volume = modal.Volume.from_name(os.environ["TENSORBOARD_OUTPUT_VOLUME"])
    source, cache = Path(OUTPUT_MOUNT), Path(EVENT_CACHE)
    _sync_events(volume, source, cache)
    process = subprocess.Popen(_tensorboard_command(cache, TENSORBOARD_PORT))

    def refresh() -> None:
        print("TensorBoard: refresh worker started", flush=True)
        while process.poll() is None:
            time.sleep(REFRESH_SECONDS)
            try:
                _sync_events(volume, source, cache)
            except Exception:
                logging.exception("Unable to refresh TensorBoard events; retrying next interval")
        print(f"TensorBoard: process exited with code {process.returncode}", flush=True)

    threading.Thread(target=refresh, daemon=True).start()


def _create_dashboard(output_volume: str) -> modal.App:
    volume = modal.Volume.from_name(output_volume, create_if_missing=True)
    dashboard = modal.App(tensorboard_app_name(output_volume), image=image)
    dashboard.function(
        name=FUNCTION_NAME, cpu=1.0, memory=2048,
        min_containers=1, max_containers=1,
        volumes={OUTPUT_MOUNT: volume.with_mount_options(read_only=True)},
        env={"TENSORBOARD_OUTPUT_VOLUME": output_volume},
    )(modal.concurrent(max_inputs=32)(
        modal.web_server(TENSORBOARD_PORT, startup_timeout=120)(_serve_tensorboard)
    ))
    return dashboard


def ensure_tensorboard(output_volume: str, *, restart: bool = False) -> str:
    """Deploy once per Volume, independently of the lifetime of a training App."""
    name = tensorboard_app_name(output_volume)
    if restart:
        _create_dashboard(output_volume).deploy(strategy="recreate")
    try:
        url = modal.Function.from_name(name, FUNCTION_NAME).get_web_url()
    except modal.exception.NotFoundError:
        _create_dashboard(output_volume).deploy()
        url = modal.Function.from_name(name, FUNCTION_NAME).get_web_url()
    if not url:
        raise RuntimeError(f"Modal TensorBoard deployment '{name}' has no web URL")
    return url


@app.local_entrypoint()
def main(output_volume: str = "pavullmo-training", restart: bool = False) -> None:
    """Start independently with `modal run .../modal_start_tensorboard.py`."""
    print(f"TensorBoard (public): {ensure_tensorboard(output_volume, restart=restart)}")
    print(f"Stop with: modal app stop {tensorboard_app_name(output_volume)}")
