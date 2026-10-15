"""TOML composition, explicit overrides, and atomic standalone snapshots."""

from __future__ import annotations

import copy
import os
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, get_args

import tomlkit
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from .base import Config

ConfigT = TypeVar("ConfigT", bound="Config")


def merge_dicts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge tables recursively; scalars and lists replace earlier values."""
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(merged.get(key), Mapping) and isinstance(value, Mapping):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_toml(text: str) -> dict[str, Any]:
    data = tomllib.loads(text)
    if "extends" in data:
        raise ValueError("extends requires a file location; use Config.load(path)")
    return data


def parse_config_string(text: str) -> dict[str, Any]:
    """Read compact inline-table configs or former multiline TOML CSV values."""
    data = parse_toml(text)
    if set(data) == {"config"}:
        if not isinstance(data["config"], dict):
            raise ValueError("compact config must contain a TOML inline table")
        return data["config"]
    return data


def _read_file(
    path: Path, stack: tuple[Path, ...] = (), *, extends: Sequence[str | Path] = (),
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"circular config includes: {chain}")
    logger.debug("Reading config {}", path)
    with path.open("rb") as file:
        data = tomllib.load(file)
    includes = data.pop("extends", [])
    if not isinstance(includes, list) or any(
        not isinstance(item, str) or not item for item in includes
    ):
        raise ValueError(f"extends in {path} must be a list of nonempty file paths")
    merged: dict[str, Any] = {}
    for include in (*includes, *extends):
        merged = merge_dicts(merged, _read_file(path.parent / include, (*stack, path)))
    return merge_dicts(merged, data)


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    # An optional config section still exposes its fields for overrides.
    args = get_args(annotation)
    if type(None) in args:
        for arg in args:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    return None


def _validate_override_path(config_type: type[BaseModel], path: str) -> list[str]:
    parts = path.split(".")
    current = config_type
    for index, part in enumerate(parts):
        field = current.model_fields.get(part)
        if field is None:
            raise ValueError(f"unknown override path: {path!r}")
        if index != len(parts) - 1:
            nested = _nested_model(field.annotation)
            if nested is None:
                raise ValueError(f"override path is not a config section: {path!r}")
            current = nested
    return parts


def apply_overrides(
    config_type: type[BaseModel],
    data: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply typed values at known dotted paths without changing the input."""
    result = copy.deepcopy(dict(data))
    for path, value in overrides.items():
        if not isinstance(path, str):
            raise ValueError("override paths must be strings")
        parts = _validate_override_path(config_type, path)
        _validate_override_contents(config_type, parts, value)
        section = result
        for part in parts[:-1]:
            existing = section.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"cannot override {path!r}: {part!r} is not a table")
            section = existing
        field = parts[-1]
        if isinstance(section.get(field), Mapping) and isinstance(value, Mapping):
            section[field] = merge_dicts(section[field], value)
        else:
            section[field] = copy.deepcopy(value)
        logger.debug("Applying config override {}", path)
    return result


def _validate_override_contents(
    config_type: type[BaseModel], parts: list[str], value: Any,
) -> None:
    """A section-valued override must also reject typos in its child fields."""
    current = config_type
    for part in parts[:-1]:
        current = _nested_model(current.model_fields[part].annotation)
    nested = _nested_model(current.model_fields[parts[-1]].annotation)
    if nested is not None and isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise ValueError("override field names must be strings")
            path = ".".join((*parts, name))
            child_parts = _validate_override_path(config_type, path)
            _validate_override_contents(config_type, child_parts, item)


def load_config(
    config_type: type[ConfigT],
    paths: str | Path | Sequence[str | Path],
    *,
    overrides: Mapping[str, Any] | None = None,
    extends: Sequence[str | Path] | None = None,
) -> ConfigT:
    files = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not files:
        raise ValueError("at least one config file is required")
    try:
        if isinstance(extends, (str, Path)) or (extends is not None and any(
            not isinstance(item, (str, Path)) or not str(item) for item in extends
        )):
            raise ValueError("extends must be a sequence of nonempty file paths")
        data: dict[str, Any] = {}
        for path in files:
            data = merge_dicts(data, _read_file(Path(path), extends=extends or ()))
        if overrides:
            data = apply_overrides(config_type, data, overrides)
    except (OSError, ValueError) as error:
        logger.error("Could not load config: {}", error)
        raise
    config = config_type.from_dict(data)
    logger.info("Loaded {} from {}", config_type.__name__, ", ".join(map(str, files)))
    return config


def _check_toml_values(value: Any, path: str = "") -> None:
    if value is None:
        raise ValueError(
            f"TOML cannot represent None at {path!r}; use an explicit value "
            "such as enabled=false instead of null"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_toml_values(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_toml_values(item, f"{path}[{index}]")


def dump_toml(data: Mapping[str, Any], *, compact: bool = False) -> str:
    if "extends" in data:
        raise ValueError("extends is reserved for file composition, not resolved configs")
    _check_toml_values(data)
    if compact:
        table = tomlkit.inline_table()
        table.update(data)
        return tomlkit.dumps({"config": table}).rstrip("\r\n")
    return tomlkit.dumps(dict(data))


def save_toml(data: Mapping[str, Any], path: str | Path) -> None:
    """Atomically save a standalone TOML dictionary, including partial configs."""
    path = Path(path).expanduser()
    temporary_path: Path | None = None
    try:
        text = dump_toml(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        temporary_path.replace(path)
    except (OSError, ValueError, TypeError) as error:
        logger.error("Could not save config to {}: {}", path, error)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    logger.info("Saved TOML config to {}", path)


def save_config(config: Config, path: str | Path) -> None:
    # Revalidate in case a caller edited model attributes or mutable lists.
    validated = type(config).model_validate(
        config.to_dict(), context={"warn_unknown_config": False},
    )
    save_toml(validated.to_dict(), path)
