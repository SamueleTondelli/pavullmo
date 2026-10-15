"""Typed configuration boundaries shared by every config class."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError


class Config(BaseModel):
    """Preserve unknown settings while strictly validating supported settings.

    File composition happens before validation, so partial presets never inject
    defaults into other presets. Importing this module does not configure sinks.
    """

    model_config = ConfigDict(
        extra="allow", strict=True, validate_default=True, allow_inf_nan=False,
    )

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Self:
        try:
            config = super().model_validate(obj, **kwargs)
        except ValidationError as error:
            logger.error("Invalid {} configuration: {}", cls.__name__, error)
            raise
        context = kwargs.get("context")
        if not isinstance(context, Mapping) or context.get("warn_unknown_config", True):
            _warn_unknown_settings(config)
        return config

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Load already composed data; unknown settings are warned and retained."""
        return cls.model_validate(dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Include all defaults and unknown settings in a portable dictionary."""
        return self.model_dump(mode="python", round_trip=True)

    @classmethod
    def from_toml(cls, text: str) -> Self:
        """Decode standalone TOML. Use load() to resolve file includes."""
        from .serialization import parse_toml

        return cls.from_dict(parse_toml(text))

    def to_toml(self) -> str:
        from .serialization import dump_toml

        return dump_toml(self.to_dict())

    def to_string(self) -> str:
        """Single-line TOML with a config inline table, suitable for CSV fields."""
        from .serialization import dump_toml

        return dump_toml(self.to_dict(), compact=True)

    @classmethod
    def from_string(cls, text: str) -> Self:
        """Load to_string() output or a former multiline TOML config string."""
        from .serialization import parse_config_string

        return cls.from_dict(parse_config_string(text))

    @classmethod
    def load(
        cls,
        paths: str | Path | Sequence[str | Path],
        *,
        overrides: Mapping[str, Any] | None = None,
        extends: Sequence[str | Path] | None = None,
    ) -> Self:
        """Compose files and apply already parsed values at dotted field paths.

        Supplied extends paths append to each top-level file's authored includes.
        Argument parsing belongs to the calling entry point, not this layer.
        """
        from .serialization import load_config

        return load_config(cls, paths, overrides=overrides, extends=extends)

    def save(self, path: str | Path) -> None:
        """Atomically save a complete standalone TOML snapshot."""
        from .serialization import save_config

        save_config(self, path)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Self:
        """Return a validated copy, leaving this config and its extras intact."""
        from .serialization import apply_overrides

        return type(self).from_dict(apply_overrides(type(self), self.to_dict(), overrides))


def _warn_unknown_settings(value: Any, path: str = "") -> None:
    if isinstance(value, BaseModel):
        for name, extra in (value.model_extra or {}).items():
            location = f"{path}.{name}" if path else name
            kind = "section" if isinstance(extra, Mapping) else "field"
            logger.warning(
                "Unknown config {} '{}': preserved, not applied.", kind, location,
            )
        for name in type(value).model_fields:
            location = f"{path}.{name}" if path else name
            _warn_unknown_settings(getattr(value, name), location)
    elif isinstance(value, Mapping):
        for name, item in value.items():
            _warn_unknown_settings(item, f"{path}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _warn_unknown_settings(item, f"{path}[{index}]")
