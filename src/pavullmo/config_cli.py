"""Shared command-line config lookup and TOML override parsing."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PRESET_DIR = Path(__file__).resolve().parent / "configs"


def parse_extends(groups: Sequence[str]) -> list[str]:
    """Parse semicolon-separated preset paths without involving config classes."""
    return [item.strip() for group in groups for item in group.split(";") if item.strip()]


def resolve_config_path(
    value: str | Path, *, relative_to: Path | None = None, preset_dir: Path = PRESET_DIR,
) -> Path:
    """Find a file, optionally adding .toml; bundled presets also accept names."""
    path = Path(value).expanduser()
    names = [path] if path.suffix else [path, path.with_suffix(".toml")]
    directories = ([relative_to] if relative_to is not None else []) + [Path.cwd(), preset_dir]
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

