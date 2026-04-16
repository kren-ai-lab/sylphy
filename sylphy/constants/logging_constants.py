"""Declare logging defaults and environment parsing helpers."""

import logging
import os

# ----------------------------
# Defaults & Env Overrides
# ----------------------------

LOG_ENV_PREFIX = "SYLPHY_LOG_"

LOG_DEFAULT_NAME = "sylphy"
LOG_DEFAULT_LEVEL = logging.INFO
LOG_DEFAULT_JSON = False
LOG_DEFAULT_STDERR = False
LOG_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
LOG_DEFAULT_BACKUPS = 3

# Accept common textual levels
LOG_LEVEL_MAP = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def env_log_level() -> int:
    """Return the configured log level from environment variables."""
    lvl = os.getenv(f"{LOG_ENV_PREFIX}LEVEL", "").upper()
    return LOG_LEVEL_MAP.get(lvl, LOG_DEFAULT_LEVEL)


def env_log_json(*, default: bool = LOG_DEFAULT_JSON) -> bool:
    """Return whether JSON logging is enabled."""
    v = os.getenv(f"{LOG_ENV_PREFIX}JSON")
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def env_log_stderr(*, default: bool = LOG_DEFAULT_STDERR) -> bool:
    """Return whether console logging should target stderr."""
    v = os.getenv(f"{LOG_ENV_PREFIX}STDERR")
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}
