"""Local TOML-configured pretraining; CLI parsing stays outside config classes."""

from __future__ import annotations

import argparse
import tomllib
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from loguru import logger

from config import PretrainingConfig, configure_logging

PRESET_DIR = Path(__file__).resolve().parent / "configs"


def parse_extends(groups: Sequence[str]) -> list[str]:
    """Parse semicolon-separated preset paths without involving config classes."""
    return [item.strip() for group in groups for item in group.split(";") if item.strip()]


def _config_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    """Find a file, optionally adding .toml; bundled presets also accept names."""
    path = Path(value).expanduser()
    names = [path] if path.suffix else [path, path.with_suffix(".toml")]
    directories = ([relative_to] if relative_to is not None else []) + [Path.cwd(), PRESET_DIR]
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"config not found: {value}")


def _override_items(text: str) -> list[str]:
    """Split on semicolons outside TOML strings, arrays and inline tables."""
    items = []
    start = depth = 0
    quote = ""
    escaped = False
    for index, character in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\" and quote == '"':
                escaped = True
            elif character == quote:
                quote = ""
        elif character in "\"'":
            quote = character
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced brackets in --override")
        elif character == ";" and depth == 0:
            items.append(text[start:index])
            start = index + 1
    if quote or depth:
        raise ValueError("unterminated string or brackets in --override")
    items.append(text[start:])
    return [item.strip() for item in items if item.strip()]


def parse_overrides(groups: Sequence[str]) -> dict[str, Any]:
    """Parse CLI values once, returning the typed mapping expected by configs.

    Values use TOML syntax; bare words/paths are accepted as strings. Later
    assignments win. Quote the whole argument so the shell retains semicolons.
    """
    overrides = {}
    for group in groups:
        for item in _override_items(group):
            path, separator, raw = item.partition("=")
            path, raw = path.strip(), raw.strip()
            if not separator or not path or not raw:
                raise ValueError(f"expected field=value in --override, got {item!r}")
            try:
                value = tomllib.loads(f"value = {raw}")["value"]
            except tomllib.TOMLDecodeError:
                value = raw
            overrides[path] = value
    return overrides


def prepare_config(
    config: str | Path | None, resume: str | Path | None, overrides: Mapping[str, Any],
    *, extends: Sequence[str | Path] = (),
) -> PretrainingConfig | None:
    """Resolve presets locally for fresh runs; resume uses the checkpoint's TOML."""
    if bool(config) == bool(resume):
        raise ValueError("provide exactly one of --config or --resume")
    if resume:
        if extends:
            raise ValueError("--extends requires --config; use --override to edit a resumed run")
        return None
    path = _config_path(config)
    includes = [_config_path(item, relative_to=path.parent) for item in extends]
    return PretrainingConfig.load(path, overrides=overrides, extends=includes)


def run_training(
    config: PretrainingConfig | None, *, resume: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    checkpoint_saved: Callable[[], None] | None = None,
    training_context: Callable[[Path], AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    """Common execution path for local training and cloud wrappers."""
    from components import PretrainingTrainer

    if bool(config) == bool(resume):
        raise ValueError("provide a run config or a resume checkpoint, exclusively")
    trainer = (PretrainingTrainer.from_checkpoint(resume, overrides=overrides) if resume
               else PretrainingTrainer(config.with_overrides(overrides) if overrides else config))
    with trainer:
        trainer.checkpoints.after_save = checkpoint_saved
        context = training_context(trainer.output_dir) if training_context else nullcontext()
        with context:
            trainer.train()
        return {
            "name": trainer.config.name,
            "completed": trainer.global_step == trainer.total_steps,
            "global_step": trainer.global_step,
            "total_steps": trainer.total_steps,
            "train_loss": trainer.train_loss,
            "validation_loss": trainer.validation_loss,
            "checkpoint": str(trainer.checkpoints.latest_checkpoint),
        }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="TOML file or bundled preset name (supports extends)")
    source.add_argument("--resume", type=Path, help="full checkpoint to resume using its embedded config")
    parser.add_argument("--override", action="append", default=[], metavar="FIELD=VALUE;FIELD=VALUE",
                        help="semicolon-separated overrides; repeatable; later assignments win")
    parser.add_argument("--extends", action="append", default=[], metavar="CONFIG;CONFIG",
                        help="append semicolon-separated presets to the config's extends; repeatable")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        overrides = parse_overrides(args.override)
        config = prepare_config(args.config, args.resume, overrides, extends=parse_extends(args.extends))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    result = run_training(config, resume=args.resume, overrides=overrides if args.resume else None)
    logger.info("Run '{}' reached {}/{}; checkpoint: {}", result["name"],
                result["global_step"], result["total_steps"], result["checkpoint"])


if __name__ == "__main__":
    main()
