"""Centralised logging configuration for FunnelRAG.

Provides structured (JSON) and human-readable formatters, a request-ID
injection filter, file-based rotation, and convenience decorators for
measuring execution time of synchronous and asynchronous callables.

Quick start::

    from python.utils.logging_config import setup_logging, get_logger

    setup_logging(level="INFO", log_file="app.log")
    log = get_logger(__name__)
    log.info("Server started")
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, overload

# ---------------------------------------------------------------------------
# Context variable for request-scoped correlation
# ---------------------------------------------------------------------------

_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set the request ID for the current async context.

    Args:
        request_id: Explicit ID to use.  When ``None`` a new UUID4 is
            generated.

    Returns:
        The request ID that was set.
    """
    rid = request_id or str(uuid.uuid4())
    _request_id_ctx.set(rid)
    return rid


def get_request_id() -> str:
    """Return the current request ID (empty string if unset)."""
    return _request_id_ctx.get()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class AddRequestIdFilter(logging.Filter):
    """Logging filter that injects ``request_id`` into every log record.

    The value is read from :data:`_request_id_ctx` so it works correctly
    across async tasks without thread-local gotchas.

    Usage::

        handler = logging.StreamHandler()
        handler.addFilter(AddRequestIdFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get()  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class _HumanFormatter(logging.Formatter):
    """Colour-free, human-readable single-line formatter."""

    _FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(request_id)s | "
        "%(name)s | %(message)s"
    )

    def __init__(self) -> None:
        super().__init__(fmt=self._FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter suitable for log aggregation systems.

    Every log record is serialised as a single JSON object on one line,
    containing at minimum: ``timestamp``, ``level``, ``logger``, ``message``,
    and ``request_id``.  Any ``extra`` keys are merged into the object.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Ensure the request_id attribute exists (filter may not be installed)
        record.request_id = getattr(record, "request_id", _request_id_ctx.get())  # type: ignore[attr-defined]

        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": record.request_id,  # type: ignore[attr-defined]
        }

        # Merge in any extra fields the caller attached
        if hasattr(record, "extra_payload") and isinstance(record.extra_payload, dict):
            payload.update(record.extra_payload)

        # Capture exception info when present
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)

        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps(
                {"message": record.getMessage(), "level": record.levelname},
                ensure_ascii=False,
            )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

_NOISY_LOGGERS: list[str] = [
    "urllib3",
    "httpcore",
    "httpx",
    "grpc",
    "pymilvus",
    "asyncio",
    "multipart",
]


def configure_third_party_logging(level: str = "WARNING") -> None:
    """Suppress noisy third-party loggers by raising their effective level.

    Args:
        level: Minimum level to set on all listed third-party loggers.
    """
    numeric = getattr(logging, level.upper(), logging.WARNING)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(numeric)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
    json_format: bool = False,
) -> None:
    """Configure the root logger with console (and optional file) output.

    Args:
        level: Root logging level (e.g. ``"DEBUG"``, ``"INFO"``).
        log_file: Path to a rotating log file.  ``None`` disables file logging.
        max_bytes: Maximum size of each log file before rotation (default 50 MB).
        backup_count: Number of rotated backup files to retain.
        json_format: Use :class:`JSONFormatter` instead of the human-readable
            format.  Recommended for production deployments shipping logs to
            an aggregation service.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if root.handlers:
        root.handlers.clear()

    # Shared filter
    request_id_filter = AddRequestIdFilter()

    # Choose formatter
    formatter: logging.Formatter = JSONFormatter() if json_format else _HumanFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(root.level)
    console.setFormatter(formatter)
    console.addFilter(request_id_filter)
    root.addHandler(console)

    # Optional file handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(root.level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_id_filter)
        root.addHandler(file_handler)

    # Quiet down third-party noise
    configure_third_party_logging()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a named logger instance.

    Args:
        name: Logger name (typically ``__name__``).  Falls back to the
            module name of the caller.
    """
    return logging.getLogger(name or __name__)


def get_request_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger that automatically includes request context.

    The returned logger wraps :func:`get_logger` and is intended for use
    inside request-scoped code where :data:`_request_id_ctx` is set.

    Args:
        name: Logger name.
    """
    return get_logger(name)


# ---------------------------------------------------------------------------
# Timing decorators
# ---------------------------------------------------------------------------

_F = TypeVar("_F", bound Callable[..., Any])


def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
) -> Callable[[_F], _F]:
    """Decorator that logs the wall-clock duration of a synchronous callable.

    Args:
        logger: Logger instance to use.  ``None`` → ``get_logger(__name__)``.
        level: Log level for the timing message.

    Example::

        @log_execution_time(level=logging.INFO)
        def expensive_computation():
            ...
    """

    def decorator(func: _F) -> _F:
        _logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _logger.log(
                    level,
                    "%s executed in %.2f ms",
                    func.__qualname__,
                    elapsed_ms,
                )

        return wrapper  # type: ignore[return-value]

    return decorator


def log_async_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
) -> Callable[[_F], _F]:
    """Decorator that logs the wall-clock duration of an async callable.

    Args:
        logger: Logger instance to use.  ``None`` → ``get_logger(__name__)``.
        level: Log level for the timing message.

    Example::

        @log_async_execution_time(level=logging.INFO)
        async def fetch_from_milvus():
            ...
    """

    def decorator(func: _F) -> _F:
        _logger = logger or get_logger(func.__module__)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                _logger.log(
                    level,
                    "%s executed in %.2f ms",
                    func.__qualname__,
                    elapsed_ms,
                )

        return wrapper  # type: ignore[return-value]

    return decorator
