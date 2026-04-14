"""Shared logging configuration for all backend services."""

import logging
import sys
from datetime import datetime
from typing import Iterable


DEFAULT_NOISY_LOGGERS = (
    "pymongo",
    "pymongo.topology",
    "pymongo.pool",
    "motor",
    "motor.motor_asyncio",
    "urllib3",
    "asyncio",
    "qdrant_client",
)


class CompactFormatter(logging.Formatter):
    """Compact service log format used across all services."""

    LEVEL_MAP = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    def formatTime(self, record, datefmt=None):
        return datetime.utcfromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S")

    def format(self, record):
        level = self.LEVEL_MAP.get(record.levelname, record.levelname[:1])
        short_name = record.name.split(".")[-1]
        message = record.getMessage()

        if record.exc_info:
            try:
                exc_text = self.formatException(record.exc_info).splitlines()[-1]
                message = f"{message} | {exc_text}"
            except Exception:
                pass

        return f"{self.formatTime(record)} {level} {short_name}: {message}"


def _resolve_level(log_level: str | int) -> int:
    if isinstance(log_level, int):
        return log_level

    normalized = str(log_level).upper()
    return logging.getLevelNamesMapping().get(normalized, logging.INFO)


def configure_logging(
    service_name: str,
    log_level: str | int = "INFO",
    noisy_loggers: Iterable[str] | None = None,
) -> logging.Logger:
    """Configure root logging with a consistent compact format."""
    level = _resolve_level(log_level)
    formatter = CompactFormatter()
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in list(root_logger.handlers):
            try:
                handler.setFormatter(formatter)
            except Exception:
                pass

    root_logger.setLevel(level)

    # Force uvicorn loggers to use root handlers/formatter so access logs match.
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(level)

    for logger_name in DEFAULT_NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if noisy_loggers:
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger = logging.getLogger(service_name)
    logger.setLevel(level)
    return logger
