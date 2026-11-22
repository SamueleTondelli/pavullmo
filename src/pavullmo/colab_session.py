"""Manage a Google Colab VM lifecycle for bash training sweeps and shared SSH helpers."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import errno
import hashlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from threading import Event
from threading import Thread
from typing import Any

from loguru import logger

from config import PretrainingConfig, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_WORKSPACE = "/content/pavullmo"
REMOTE_DATASET_DIR = "/content/datasets"
REMOTE_OUTPUT_DIR = "/content/outputs/runs"
REMOTE_RUNS_CSV = "/content/outputs/pretrain_runs.csv"
REMOTE_CACHE_DIR = "/content/cache"
DEFAULT_LOCAL_OUTPUT_DIR = Path("artifacts/runs")
DEFAULT_LOCAL_RUNS_CSV = Path("artifacts/results/pretrain_runs.csv")
TENSORBOARD_SYNC_SECONDS = 15.0
SUPPORTED_BF16_GPUS = ("L4", "A100", "H100", "G4")
SSH_HOST_ALIAS = "colab-runtime"
RUNS_CSV_FIELDS = [
    "experiment_name",
    "dataset_variant",
    "train_loss",
    "validation_loss",
    "train_script",
    "model_path",
    "config",
]

ENV_SESSION = "PAVULLMO_COLAB_SESSION"
ENV_MANAGED_SESSION = "PAVULLMO_COLAB_MANAGED_SESSION"
ENV_HF_REPO = "PAVULLMO_COLAB_HF_REPO"
ENV_HF_REVISION = "PAVULLMO_COLAB_HF_REVISION"
ENV_COLAB_AUTH = "PAVULLMO_COLAB_AUTH"
ENV_SSH_IDENTITY = "PAVULLMO_COLAB_SSH_IDENTITY"
ENV_REMOTE_DIR = "PAVULLMO_COLAB_REMOTE_DIR"
ENV_SSH_CONFIG = "PAVULLMO_COLAB_SSH_CONFIG"

_HF_DOWNLOAD_SCRIPT = """
import json
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

payload = json.load(sys.stdin)
dataset_dir = Path(payload["dataset_dir"])
dataset_dir.mkdir(parents=True, exist_ok=True)
requested = payload["datasets"]
missing = [name for name in requested if not (dataset_dir / name / "metadata.json").is_file()]
if not missing:
    print(f"All requested datasets already staged in {dataset_dir}: {', '.join(requested)}", flush=True)
else:
    print(f"Downloading datasets from {payload['repo_id']} into {dataset_dir}: {', '.join(missing)}", flush=True)
    snapshot_download(
        repo_id=payload["repo_id"],
        repo_type="dataset",
        revision=payload["revision"],
        local_dir=str(dataset_dir),
        allow_patterns=[f"{name}/*" for name in missing],
        token=payload["token"],
    )
still_missing = [name for name in requested if not (dataset_dir / name / "metadata.json").is_file()]
if still_missing:
    raise SystemExit(f"Datasets missing metadata.json after HF download: {', '.join(still_missing)}")
""".strip()


def parse_datasets(groups: Sequence[str]) -> list[str]:
    """Parse repeatable, semicolon-separated dataset paths relative to dataset_dir."""
    seen: set[str] = set()
    datasets: list[str] = []
    for group in groups:
        for raw_item in group.split(";"):
            item = raw_item.strip()
            if not item:
                raise ValueError("dataset name cannot be empty")
            if item.startswith("/"):
                raise ValueError(f"dataset path must be relative, got {item!r}")
            cleaned = item.strip("/")
            if not cleaned or ".." in PurePosixPath(cleaned).parts:
                raise ValueError(f"invalid relative dataset path {item!r}")
            if cleaned not in seen:
                seen.add(cleaned)
                datasets.append(cleaned)
    return datasets


def validate_gpu(gpu: str) -> str:
    """Validate and normalize a Colab GPU type compatible with BF16 training."""
    normalized = gpu.strip().upper()
    if normalized == "T4":
        raise ValueError(
            "GPU 'T4' does not support BF16/FlashAttention-2 required by pavullmo; "
            f"use one of: {', '.join(SUPPORTED_BF16_GPUS)}"
        )
    if normalized not in SUPPORTED_BF16_GPUS:
        raise ValueError(
            f"Unsupported GPU {gpu!r}; expected one of: {', '.join(SUPPORTED_BF16_GPUS)}"
        )
    return normalized


def colab_base_cmd(colab_auth: str = "") -> list[str]:
    """Build the base `colab` CLI command, preserving an optional auth strategy."""
    cmd = [shutil.which("colab") or "colab"]
    if colab_auth:
        cmd.extend(["--auth", colab_auth])
    return cmd


def resolve_hf_token() -> str | None:
    """Resolve a Hugging Face token from environment variables or local CLI login."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        import huggingface_hub
    except ImportError:
        return None
    get_token = getattr(huggingface_hub, "get_token", None)
    return get_token() if callable(get_token) else None


def control_socket_path(session: str) -> Path:
    """Return a short Unix domain socket path well below the 108-byte sun_path limit."""
    digest = hashlib.sha256(session.encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"pvl-colab-{digest}.sock"


def build_ssh_config_text(
    session: str, *, colab_auth: str = "", identity: str = "",
) -> str:
    """Build a dedicated OpenSSH config for multiplexing traffic over one Colab session."""
    proxy_argv = [*colab_base_cmd(colab_auth), "ssh", "--proxy-mode", "-s", session]
    if identity:
        proxy_argv.extend(["--identity", identity])

    lines = [
        f"Host {SSH_HOST_ALIAS}",
        "  User root",
        f"  ProxyCommand {shlex.join(proxy_argv)}",
    ]
    if identity:
        expanded = Path(identity).expanduser()
        if expanded.suffix == ".pub":
            private = expanded.with_suffix("")
            identity_file = str(private if private.exists() else expanded)
        else:
            identity_file = str(expanded)
        lines.append(f"  IdentityFile {identity_file}")

    lines.extend([
        "  StrictHostKeyChecking no",
        "  UserKnownHostsFile /dev/null",
        "  LogLevel ERROR",
        "  ControlMaster auto",
        f"  ControlPath {control_socket_path(session)}",
        "  ControlPersist 600",
        "  ServerAliveInterval 30",
        "  ServerAliveCountMax 3",
    ])
    return "\n".join(lines) + "\n"


@contextmanager
def ssh_config_file(
    session: str,
    *,
    colab_auth: str = "",
    identity: str = "",
    existing_path: str = "",
) -> Iterator[Path]:
    """Yield an existing SSH config file or materialize a temporary one with mode 0600."""
    if existing_path:
        candidate = Path(existing_path)
        if candidate.is_file():
            yield candidate
            return

    temp_dir = "/tmp" if " " in tempfile.gettempdir() else None
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix="pavullmo-colab-",
        suffix=".sshconfig",
        dir=temp_dir,
        delete=False,
        encoding="utf-8",
    ) as file:
        path = Path(file.name)
        file.write(build_ssh_config_text(session, colab_auth=colab_auth, identity=identity))
    os.chmod(path, 0o600)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def ssh_command(ssh_config: Path, remote_cmd: str, *, tty: bool = False) -> list[str]:
    """Build an ssh command targeting the multiplexed Colab host alias."""
    return ["ssh", "-F", str(ssh_config), "-t" if tty else "-T", SSH_HOST_ALIAS, remote_cmd]


def close_ssh_master(ssh_config: Path) -> None:
    """Close any active ControlMaster socket cleanly without failing teardown."""
    try:
        subprocess.run(
            ["ssh", "-F", str(ssh_config), "-O", "exit", SSH_HOST_ALIAS],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def create_session(
    session: str, *, gpu: str, high_mem: bool = False, colab_auth: str = "",
) -> None:
    """Provision a Colab runtime with the requested GPU."""
    cmd = [*colab_base_cmd(colab_auth), "new", "-s", session, "--gpu", gpu]
    if high_mem:
        cmd.append("--high-mem")
    logger.info("Creating Colab session '{}' with GPU {}", session, gpu)
    subprocess.run(cmd, check=True)


def stop_session(session: str, *, colab_auth: str = "") -> None:
    """Release a Colab runtime, logging rather than raising on teardown errors."""
    cmd = [*colab_base_cmd(colab_auth), "stop", "-s", session]
    logger.info("Stopping Colab session '{}'", session)
    try:
        result = subprocess.run(cmd, check=False)
    except OSError as error:
        logger.warning("Failed to invoke colab stop for '{}': {}", session, error)
        return
    if result.returncode != 0:
        logger.warning("colab stop for '{}' exited with code {}", session, result.returncode)


def sync_workspace(
    ssh_config: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    remote_dir: str = REMOTE_WORKSPACE,
    remote_dataset_dir: str = REMOTE_DATASET_DIR,
    remote_output_dir: str = REMOTE_OUTPUT_DIR,
    run_uv_sync: bool = True,
    run_check: bool = True,
) -> None:
    """Sync repository metadata/source and ensure remote .venv and caches are ready."""
    remote_base = remote_dir.rstrip("/") or "/"
    mkdir_cmd = "mkdir -p " + " ".join(
        shlex.quote(path)
        for path in (
            remote_base,
            remote_dataset_dir,
            remote_output_dir,
            f"{REMOTE_CACHE_DIR}/torchinductor",
            f"{REMOTE_CACHE_DIR}/triton",
            f"{REMOTE_CACHE_DIR}/uv",
        )
    )
    subprocess.run(ssh_command(ssh_config, mkdir_cmd), check=True)

    logger.info("Syncing project files to {}", remote_base)
    rsync_cmd = [
        "rsync",
        "-az",
        "--delete",
        "--exclude",
        ".venv",
        "--exclude",
        "__pycache__",
        "--exclude",
        "*.pyc",
        "-e",
        shlex.join(["ssh", "-F", str(ssh_config)]),
        str(project_root / "pyproject.toml"),
        str(project_root / "uv.lock"),
        str(project_root / "src"),
        f"{SSH_HOST_ALIAS}:{remote_base}/",
    ]
    subprocess.run(rsync_cmd, check=True)

    if run_uv_sync:
        logger.info("Running uv sync --frozen on Colab runtime")
        sync_script = "\n".join([
            "set -euo pipefail",
            'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"',
            "if ! command -v uv >/dev/null 2>&1; then",
            "  curl -LsSf https://astral.sh/uv/install.sh | sh",
            "fi",
            f"cd {shlex.quote(remote_base)}",
            f"UV_CACHE_DIR={shlex.quote(f'{REMOTE_CACHE_DIR}/uv')} uv sync --frozen",
        ])
        subprocess.run(
            ssh_command(ssh_config, f"bash -c {shlex.quote(sync_script)}"),
            check=True,
        )

    if run_check:
        logger.info("Running pavullmo-check on Colab runtime")
        subprocess.run(
            ssh_command(ssh_config, shlex.quote(f"{remote_base}/.venv/bin/pavullmo-check")),
            check=True,
        )


def ensure_hf_datasets(
    ssh_config: Path,
    *,
    hf_repo: str,
    datasets: Sequence[str],
    remote_dataset_dir: str = REMOTE_DATASET_DIR,
    remote_dir: str = REMOTE_WORKSPACE,
    revision: str | None = None,
    token: str | None = None,
) -> None:
    """Download any missing dataset artifacts from a Hugging Face dataset repo onto the VM."""
    normalized = parse_datasets(datasets)
    if not normalized:
        return
    resolved_token = token if token is not None else resolve_hf_token()
    remote_python = f"{remote_dir.rstrip('/')}/.venv/bin/python"
    payload = {
        "repo_id": hf_repo,
        "revision": revision or None,
        "datasets": normalized,
        "dataset_dir": remote_dataset_dir,
        "token": resolved_token,
    }
    logger.info(
        "Ensuring Hugging Face datasets from '{}' in {}: {}",
        hf_repo,
        remote_dataset_dir,
        ", ".join(normalized),
    )
    subprocess.run(
        ssh_command(
            ssh_config,
            f"{shlex.quote(remote_python)} -c {shlex.quote(_HF_DOWNLOAD_SCRIPT)}",
        ),
        input=json.dumps(payload),
        text=True,
        check=True,
    )


def sync_tensorboard_events_once(
    ssh_config: Path, *, remote_output_dir: str, local_output_dir: Path,
) -> None:
    """Mirror only TensorBoard event files from the remote output directory to local runs."""
    local_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-az",
        "--include",
        "*/",
        "--include",
        "events.out.tfevents.*",
        "--exclude",
        "*",
        "--prune-empty-dirs",
        "-e",
        shlex.join(["ssh", "-F", str(ssh_config)]),
        f"{SSH_HOST_ALIAS}:{remote_output_dir.rstrip('/')}/",
        f"{str(local_output_dir).rstrip('/')}/",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return
    if result.returncode in (23, 24):
        logger.debug(
            "Transient rsync status {} while refreshing TensorBoard events: {}",
            result.returncode,
            result.stderr.strip(),
        )
        return
    raise RuntimeError(
        f"TensorBoard event rsync failed (exit {result.returncode}): {result.stderr.strip()}"
    )


@contextmanager
def live_tensorboard_sync(
    ssh_config: Path,
    *,
    remote_output_dir: str,
    local_output_dir: Path,
    interval_seconds: float = TENSORBOARD_SYNC_SECONDS,
) -> Iterator[None]:
    """Periodically mirror TensorBoard event files while a sweep runs."""
    if interval_seconds <= 0:
        yield
        return

    stopped = Event()

    def worker() -> None:
        while not stopped.wait(interval_seconds):
            try:
                sync_tensorboard_events_once(
                    ssh_config,
                    remote_output_dir=remote_output_dir,
                    local_output_dir=local_output_dir,
                )
            except Exception:
                logger.exception("Unable to refresh TensorBoard events; retrying next interval")

    thread = Thread(target=worker, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def extract_tar_archive(archive_path: Path, destination: Path) -> None:
    """Safely extract a downloaded run archive into a staging directory."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(destination, filter="data")


def merge_directory_tree(source: Path, destination: Path) -> tuple[int, int]:
    """Merge extracted run directories into the local runs folder without removing older runs."""
    destination.mkdir(parents=True, exist_ok=True)
    top_level_dirs = sum(1 for item in source.iterdir() if item.is_dir())
    files_merged = 0
    for item in source.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        relative = item.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(item, target)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
            try:
                shutil.copy2(item, temp_path)
                os.replace(temp_path, target)
            finally:
                temp_path.unlink(missing_ok=True)
            item.unlink(missing_ok=True)
        files_merged += 1
    return top_level_dirs, files_merged


def remap_model_path(
    model_path: str, *, remote_output_dir: str, local_output_dir: Path,
) -> str:
    """Translate checkpoint paths rooted under remote_output_dir to local_output_dir."""
    remote_root = PurePosixPath(remote_output_dir)
    remote_path = PurePosixPath(model_path)
    try:
        relative = remote_path.relative_to(remote_root)
    except ValueError:
        return model_path
    return str(local_output_dir.resolve() / relative)


def merge_runs_csv(
    remote_csv_text: str,
    local_runs_csv: Path,
    *,
    remote_output_dir: str,
    local_output_dir: Path,
) -> int:
    """Append new run registry rows locally, remapping checkpoint paths and deduplicating."""
    if not remote_csv_text.strip():
        return 0
    reader = csv.DictReader(io.StringIO(remote_csv_text))
    if reader.fieldnames is None:
        return 0
    if reader.fieldnames != RUNS_CSV_FIELDS:
        raise ValueError(f"remote run registry columns differ: {reader.fieldnames}")
    incoming_rows = list(reader)
    if not incoming_rows:
        return 0

    existing = local_runs_csv.exists() and local_runs_csv.stat().st_size > 0
    seen_rows: set[tuple[str, str]] = set()
    if existing:
        with local_runs_csv.open(encoding="utf-8", newline="") as file:
            local_reader = csv.DictReader(file)
            if local_reader.fieldnames != RUNS_CSV_FIELDS:
                raise ValueError(f"run registry columns differ: {local_runs_csv}")
            for item in local_reader:
                seen_rows.add((item["model_path"], item["config"]))

    new_rows: list[dict[str, str]] = []
    for row in incoming_rows:
        updated = dict(row)
        updated["model_path"] = remap_model_path(
            updated["model_path"],
            remote_output_dir=remote_output_dir,
            local_output_dir=local_output_dir,
        )
        config_candidates = {updated["config"]}
        try:
            config_candidates.add(PretrainingConfig.from_string(updated["config"]).to_toml())
        except Exception:
            pass
        if any((updated["model_path"], candidate) in seen_rows for candidate in config_candidates):
            continue
        seen_rows.add((updated["model_path"], updated["config"]))
        new_rows.append(updated)

    if not new_rows:
        return 0

    local_runs_csv.parent.mkdir(parents=True, exist_ok=True)
    with local_runs_csv.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUNS_CSV_FIELDS)
        if not existing:
            writer.writeheader()
        writer.writerows(new_rows)
    return len(new_rows)


def download_and_merge_outputs(
    ssh_config: Path,
    *,
    remote_output_dir: str,
    remote_runs_csv: str,
    local_output_dir: Path,
    local_runs_csv: Path,
) -> dict[str, int]:
    """Fetch a single compressed archive of Colab runs and merge outputs/CSV locally."""
    probe_code = (
        "import json, os, sys; "
        "out, csv_p = sys.argv[1], sys.argv[2]; "
        "has_runs = os.path.isdir(out) and any(os.scandir(out)); "
        "has_csv = os.path.isfile(csv_p); "
        "print(json.dumps({'has_runs': has_runs, 'has_csv': has_csv}))"
    )
    probe_cmd = (
        f"python3 -c {shlex.quote(probe_code)} "
        f"{shlex.quote(remote_output_dir)} {shlex.quote(remote_runs_csv)}"
    )
    probe = subprocess.run(
        ssh_command(ssh_config, probe_cmd),
        check=True,
        capture_output=True,
        text=True,
    )
    availability = json.loads(probe.stdout)

    runs_merged = files_merged = csv_rows = 0
    if availability.get("has_runs"):
        local_output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".colab-download-", dir=local_output_dir.parent,
        ) as staging_root:
            stage = Path(staging_root)
            archive_path = stage / "runs.tar.gz"
            extracted_dir = stage / "extracted"
            tar_shell = f"tar -C {shlex.quote(remote_output_dir)} -cf - . | gzip -1"
            remote_tar_cmd = f"bash -o pipefail -c {shlex.quote(tar_shell)}"
            logger.info("Downloading sweep outputs archive from {}", remote_output_dir)
            with archive_path.open("wb") as archive_file:
                tar_result = subprocess.run(
                    ssh_command(ssh_config, remote_tar_cmd),
                    stdout=archive_file,
                    stderr=subprocess.PIPE,
                )
            if tar_result.returncode != 0:
                stderr_text = tar_result.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Failed to download Colab runs archive from {remote_output_dir}: "
                    f"{stderr_text or f'exit code {tar_result.returncode}'}"
                )
            extract_tar_archive(archive_path, extracted_dir)
            runs_merged, files_merged = merge_directory_tree(extracted_dir, local_output_dir)

    if availability.get("has_csv"):
        csv_result = subprocess.run(
            ssh_command(ssh_config, f"cat {shlex.quote(remote_runs_csv)}"),
            check=True,
            capture_output=True,
            text=True,
        )
        csv_rows = merge_runs_csv(
            csv_result.stdout,
            local_runs_csv,
            remote_output_dir=remote_output_dir,
            local_output_dir=local_output_dir,
        )

    logger.info(
        "Merged Colab outputs into {} ({} run dirs, {} files) and {} ({} new CSV rows)",
        local_output_dir,
        runs_merged,
        files_merged,
        local_runs_csv,
        csv_rows,
    )
    return {"runs": runs_merged, "files": files_merged, "csv_rows": csv_rows}


def _generate_session_name() -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"pavullmo-{timestamp}-{uuid.uuid4().hex[:4]}"


def _run_managed_command(command: Sequence[str], *, env: dict[str, str]) -> int:
    process = subprocess.Popen(list(command), env=env)
    signal_count = 0
    previous_handlers: dict[int, Any] = {}

    def on_signal(signum: int, _frame: Any) -> None:
        nonlocal signal_count
        signal_count += 1
        if signal_count == 1:
            logger.warning(
                "Received signal {}; waiting for sweep command to finish graceful stop "
                "(send signal again to force terminate)",
                signum,
            )
            if signum == signal.SIGTERM and process.poll() is None:
                process.send_signal(signal.SIGTERM)
        else:
            logger.warning("Received second signal; terminating sweep command")
            if process.poll() is None:
                process.terminate()

    install_handlers = True
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, on_signal)
    except ValueError:
        install_handlers = False

    try:
        return process.wait()
    finally:
        if install_handlers:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision one Colab VM, run a sweep command/script, sync TensorBoard live, archive-download outputs, and tear down.",
    )
    parser.add_argument("-s", "--session", default=None, help="Colab session name (auto-generated if omitted)")
    parser.add_argument("--gpu", default="L4", help="Colab GPU for a new session (L4, A100, H100, G4)")
    parser.add_argument("--high-mem", action="store_true", help="Request high-RAM machine shape when supported")
    parser.add_argument("--reuse", action="store_true", help="Attach to an existing --session instead of creating one")
    parser.add_argument(
        "--keep",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the Colab VM alive after exit (defaults to true with --reuse, false otherwise)",
    )
    parser.add_argument("--hf-repo", default=None, help=f"Hugging Face dataset repo (default: {ENV_HF_REPO})")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME;NAME",
        help="Semicolon-separated dataset artifact directories to download from --hf-repo; repeatable",
    )
    parser.add_argument("--hf-revision", default=None, help=f"Optional HF revision (default: {ENV_HF_REVISION})")
    parser.add_argument("--remote-dir", default=REMOTE_WORKSPACE, help="Remote repo directory on Colab")
    parser.add_argument("--remote-dataset-dir", default=REMOTE_DATASET_DIR, help="Remote dataset staging directory")
    parser.add_argument("--remote-output-dir", default=REMOTE_OUTPUT_DIR, help="Remote training runs directory")
    parser.add_argument("--remote-runs-csv", default=REMOTE_RUNS_CSV, help="Remote pretrain_runs.csv path")
    parser.add_argument("--local-output-dir", type=Path, default=DEFAULT_LOCAL_OUTPUT_DIR, help="Local runs directory to merge into")
    parser.add_argument("--local-runs-csv", type=Path, default=DEFAULT_LOCAL_RUNS_CSV, help="Local runs CSV to append into")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zip, download, and merge remote outputs/CSV after the command finishes",
    )
    parser.add_argument(
        "--tensorboard-sync-seconds",
        type=float,
        default=TENSORBOARD_SYNC_SECONDS,
        help="Interval for mirroring TensorBoard events locally (0 disables)",
    )
    parser.add_argument(
        "--no-tensorboard-sync",
        action="store_true",
        help="Disable background TensorBoard event mirroring",
    )
    parser.add_argument("--skip-check", action="store_true", help="Skip pavullmo-check during remote environment setup")
    parser.add_argument("--colab-auth", default=None, help=f"Optional colab --auth strategy (default: {ENV_COLAB_AUTH})")
    parser.add_argument("-i", "--identity", default=None, help=f"SSH key path (default: {ENV_SSH_IDENTITY})")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command or bash script to run against the session")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]

    if args.reuse and not args.session:
        parser.error("--reuse requires --session")

    keep = args.reuse if args.keep is None else args.keep
    if not command and not keep:
        parser.error("provide a command to run (or pass --keep to provision and keep a session)")

    hf_repo = (args.hf_repo if args.hf_repo is not None else os.environ.get(ENV_HF_REPO, "")).strip()
    hf_revision = (args.hf_revision if args.hf_revision is not None else os.environ.get(ENV_HF_REVISION, "")).strip()
    colab_auth = (args.colab_auth if args.colab_auth is not None else os.environ.get(ENV_COLAB_AUTH, "")).strip()
    identity = (args.identity if args.identity is not None else os.environ.get(ENV_SSH_IDENTITY, "")).strip()

    try:
        datasets = parse_datasets(args.dataset)
        if datasets and not hf_repo:
            raise ValueError("--dataset requires --hf-repo")
        if not args.reuse:
            gpu = validate_gpu(args.gpu)
        else:
            gpu = args.gpu.strip().upper()
    except ValueError as error:
        parser.error(str(error))

    session = args.session.strip() if args.session else _generate_session_name()
    tb_interval = 0.0 if args.no_tensorboard_sync else args.tensorboard_sync_seconds
    if tb_interval < 0:
        parser.error("--tensorboard-sync-seconds must be non-negative")

    session_active = args.reuse
    setup_done = False
    returncode = 0
    download_error: Exception | None = None

    with ssh_config_file(session, colab_auth=colab_auth, identity=identity) as ssh_config_path:
        try:
            if not args.reuse:
                create_session(session, gpu=gpu, high_mem=args.high_mem, colab_auth=colab_auth)
                session_active = True

            sync_workspace(
                ssh_config_path,
                remote_dir=args.remote_dir,
                remote_dataset_dir=args.remote_dataset_dir,
                remote_output_dir=args.remote_output_dir,
                run_uv_sync=True,
                run_check=not args.skip_check,
            )
            if datasets:
                ensure_hf_datasets(
                    ssh_config_path,
                    hf_repo=hf_repo,
                    datasets=datasets,
                    remote_dataset_dir=args.remote_dataset_dir,
                    remote_dir=args.remote_dir,
                    revision=hf_revision or None,
                )
            setup_done = True

            child_env = os.environ.copy()
            child_env[ENV_SESSION] = session
            child_env[ENV_MANAGED_SESSION] = "1"
            child_env[ENV_SSH_CONFIG] = str(ssh_config_path)
            child_env[ENV_REMOTE_DIR] = args.remote_dir
            for env_key, value in (
                (ENV_HF_REPO, hf_repo),
                (ENV_HF_REVISION, hf_revision),
                (ENV_COLAB_AUTH, colab_auth),
                (ENV_SSH_IDENTITY, identity),
            ):
                if value:
                    child_env[env_key] = value

            with live_tensorboard_sync(
                ssh_config_path,
                remote_output_dir=args.remote_output_dir,
                local_output_dir=args.local_output_dir,
                interval_seconds=tb_interval,
            ):
                if command:
                    returncode = _run_managed_command(command, env=child_env)
                else:
                    logger.info(
                        "Session '{}' is ready and kept alive (export {}={})",
                        session,
                        ENV_SESSION,
                        session,
                    )
        finally:
            if setup_done and args.download:
                try:
                    download_and_merge_outputs(
                        ssh_config_path,
                        remote_output_dir=args.remote_output_dir,
                        remote_runs_csv=args.remote_runs_csv,
                        local_output_dir=args.local_output_dir,
                        local_runs_csv=args.local_runs_csv,
                    )
                except Exception as error:
                    download_error = error
                    logger.exception("Failed to download Colab outputs from '{}'", session)
            close_ssh_master(ssh_config_path)
            if session_active and not keep:
                stop_session(session, colab_auth=colab_auth)

    if returncode != 0:
        raise SystemExit(returncode)
    if download_error is not None:
        raise download_error


if __name__ == "__main__":
    main()
