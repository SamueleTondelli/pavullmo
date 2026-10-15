"""Explicit Loguru setup for local and cloud entry points."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from .schema import LoggingConfig

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | {name}:{line} | <level>{message}</level>"
)


def configure_logging(settings: LoggingConfig | None = None) -> list[int]:
    """Replace sinks explicitly; importing configs leaves existing sinks intact.

    The optional file sink uses Loguru's rotation/retention. Diagnostic locals
    are disabled so validation failures do not dump unrelated process state.
    """
    settings = settings or LoggingConfig()
    logger.remove()
    handlers = []
    options = dict(
        level=settings.level, format=LOG_FORMAT, enqueue=settings.enqueue,
        backtrace=False, diagnose=False,
    )
    if settings.console:
        handlers.append(logger.add(sys.stderr, **options))
    if settings.file:
        path = Path(settings.file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logger.add(
            path, rotation=settings.rotation, retention=settings.retention,
            encoding="utf-8", **options,
        ))
    return handlers
