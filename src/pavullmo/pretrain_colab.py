"""Launch a single TOML-configured pretraining run on an existing Colab session."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from config import PretrainingConfig, configure_logging
from pavullmo.colab_session import (
    ENV_COLAB_AUTH,
    ENV_HF_REPO,
    ENV_HF_REVISION,
    ENV_MANAGED_SESSION,
    ENV_REMOTE_DIR,
    ENV_SESSION,
    ENV_SSH_CONFIG,
    ENV_SSH_IDENTITY,
    REMOTE_CACHE_DIR,
    REMOTE_WORKSPACE,
    ensure_hf_datasets,
    parse_datasets,
    ssh_command,
    ssh_config_file,
    sync_workspace,
)
from pavullmo.pretrain import parse_extends, parse_overrides, prepare_config


def _relative_datasets(config: PretrainingConfig) -> list[str]:
    candidates = [
        path
        for path in (config.train_dataset.path, config.validation_dataset.path)
        if not path.startswith("/")
    ]
    return parse_datasets(candidates) if candidates else []


def _interrupt_remote_training(ssh_config: Path, pid_file: str) -> None:
    remote_kill = (
        f"test -s {shlex.quote(pid_file)} && "
        f"kill -INT $(cat {shlex.quote(pid_file)}) 2>/dev/null || true"
    )
    try:
        subprocess.run(
            ssh_command(ssh_config, remote_kill),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def run_remote_pretrain(
    config: PretrainingConfig | None,
    *,
    resume: str | Path | None = None,
    override_groups: Sequence[str] = (),
    session: str,
    hf_repo: str = "",
    hf_revision: str = "",
    sync_code: bool,
    remote_dir: str = REMOTE_WORKSPACE,
    colab_auth: str = "",
    identity: str = "",
    existing_ssh_config: str = "",
) -> int:
    """Execute one pretraining run on an existing Colab VM without managing VM lifetime."""
    if bool(config) == bool(resume):
        raise ValueError("provide a run config or a resume checkpoint, exclusively")

    remote_base = remote_dir.rstrip("/") or "/"
    run_id = uuid.uuid4().hex[:12]
    pid_file = f"/tmp/pavullmo-train-{run_id}.pid"
    remote_config_path = f"/tmp/pavullmo-config-{run_id}.toml" if config is not None else ""

    with ssh_config_file(
        session,
        colab_auth=colab_auth,
        identity=identity,
        existing_path=existing_ssh_config,
    ) as ssh_config:
        if sync_code:
            sync_workspace(
                ssh_config,
                remote_dir=remote_base,
                run_uv_sync=True,
                run_check=False,
            )

        if config is not None and hf_repo:
            relative_datasets = _relative_datasets(config)
            if relative_datasets:
                ensure_hf_datasets(
                    ssh_config,
                    hf_repo=hf_repo,
                    datasets=relative_datasets,
                    remote_dataset_dir=config.runtime.dataset_dir,
                    remote_dir=remote_base,
                    revision=hf_revision or None,
                )

        if config is not None:
            subprocess.run(
                ssh_command(ssh_config, f"cat > {shlex.quote(remote_config_path)}"),
                input=config.to_toml(),
                text=True,
                check=True,
            )

        pretrain_args = [f"{remote_base}/.venv/bin/python", "-m", "pavullmo.pretrain"]
        if config is not None:
            pretrain_args.extend(["--config", remote_config_path])
        else:
            pretrain_args.extend(["--resume", str(resume)])
            for group in override_groups:
                pretrain_args.extend(["--override", group])

        remote_cmd = (
            f"cd {shlex.quote(remote_base)} && "
            f"export TORCHINDUCTOR_CACHE_DIR={shlex.quote(f'{REMOTE_CACHE_DIR}/torchinductor')} "
            f"TRITON_CACHE_DIR={shlex.quote(f'{REMOTE_CACHE_DIR}/triton')} "
            f"UV_CACHE_DIR={shlex.quote(f'{REMOTE_CACHE_DIR}/uv')} "
            "PYTHONUNBUFFERED=1 && "
            f"echo $$ > {shlex.quote(pid_file)} && "
            f"exec {shlex.join(pretrain_args)}"
        )

        logger.info("Running pretraining on Colab session '{}'", session)
        process = subprocess.Popen(ssh_command(ssh_config, remote_cmd))
        signal_count = 0
        previous_handlers: dict[int, Any] = {}

        def on_signal(signum: int, _frame: Any) -> None:
            nonlocal signal_count
            signal_count += 1
            if signal_count == 1:
                logger.warning(
                    "Received signal {}; forwarding SIGINT to remote training PID for clean checkpoint save",
                    signum,
                )
                _interrupt_remote_training(ssh_config, pid_file)
            else:
                logger.warning("Received second interrupt; terminating local SSH client")
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
            cleanup_targets = [pid_file]
            if remote_config_path:
                cleanup_targets.append(remote_config_path)
            cleanup_cmd = "rm -f " + " ".join(shlex.quote(path) for path in cleanup_targets)
            try:
                subprocess.run(
                    ssh_command(ssh_config, cleanup_cmd),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError):
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="TOML file or bundled preset name (supports extends)")
    source.add_argument("--resume", type=Path, help="remote checkpoint path to resume using its embedded config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FIELD=VALUE;FIELD=VALUE",
        help="semicolon-separated overrides; repeatable; later assignments win",
    )
    parser.add_argument(
        "--extends",
        action="append",
        default=[],
        metavar="CONFIG;CONFIG",
        help="append semicolon-separated presets to the config's extends; repeatable",
    )
    parser.add_argument("-s", "--session", default=None, help=f"Colab session name (default: {ENV_SESSION})")
    parser.add_argument("--hf-repo", default=None, help=f"Optional HF dataset repo for lazy staging (default: {ENV_HF_REPO})")
    parser.add_argument("--hf-revision", default=None, help=f"Optional HF revision (default: {ENV_HF_REVISION})")
    parser.add_argument(
        "--sync-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sync code and dependencies before running (skipped automatically inside colab_session.py)",
    )
    parser.add_argument("--remote-dir", default=None, help=f"Remote repo directory (default: {ENV_REMOTE_DIR} or {REMOTE_WORKSPACE})")
    parser.add_argument("--colab-auth", default=None, help=f"Optional colab --auth strategy (default: {ENV_COLAB_AUTH})")
    parser.add_argument("-i", "--identity", default=None, help=f"SSH key path (default: {ENV_SSH_IDENTITY})")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    try:
        overrides = parse_overrides(args.override)
        config = prepare_config(
            args.config, args.resume, overrides, extends=parse_extends(args.extends),
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))

    session = (args.session if args.session is not None else os.environ.get(ENV_SESSION, "")).strip()
    if not session:
        parser.error(
            f"provide --session / {ENV_SESSION}, or run via src/pavullmo/colab_session.py"
        )

    env_session = os.environ.get(ENV_SESSION, "").strip()
    managed_same_session = (
        os.environ.get(ENV_MANAGED_SESSION) == "1" and env_session == session
    )
    sync_code = (not managed_same_session) if args.sync_code is None else args.sync_code
    existing_ssh_config = (
        os.environ.get(ENV_SSH_CONFIG, "") if env_session == session else ""
    )

    hf_repo = (args.hf_repo if args.hf_repo is not None else os.environ.get(ENV_HF_REPO, "")).strip()
    hf_revision = (args.hf_revision if args.hf_revision is not None else os.environ.get(ENV_HF_REVISION, "")).strip()
    remote_dir = (args.remote_dir if args.remote_dir is not None else os.environ.get(ENV_REMOTE_DIR, REMOTE_WORKSPACE)).strip()
    colab_auth = (args.colab_auth if args.colab_auth is not None else os.environ.get(ENV_COLAB_AUTH, "")).strip()
    identity = (args.identity if args.identity is not None else os.environ.get(ENV_SSH_IDENTITY, "")).strip()

    returncode = run_remote_pretrain(
        config,
        resume=args.resume,
        override_groups=args.override,
        session=session,
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        sync_code=sync_code,
        remote_dir=remote_dir,
        colab_auth=colab_auth,
        identity=identity,
        existing_ssh_config=existing_ssh_config,
    )
    if returncode != 0:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
