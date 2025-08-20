# protein_representation/logging_config.py
from __future__ import annotations

import json
import logging
import os
import sys
import time
from logging import Logger, LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from protein_representation.constants.logging_constants import (_ENV_PREFIX,_DEF_NAME,
                                                                _DEF_LEVEL,_DEF_JSON, 
                                                                _DEF_STDERR, _DEF_MAX_BYTES, 
                                                                _DEF_BACKUPS, _LEVEL_MAP)

from protein_representation.core import config as _cfg

# Ensure asctime is UTC everywhere (file + console)
time.gmtime  # noqa: F401  # keep linter happy
logging.Formatter.converter = time.gmtime  # type: ignore[attr-defined]


# ----------------------------
# Utilities
# ----------------------------

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(_ENV_PREFIX + key)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(_ENV_PREFIX + key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    return os.getenv(_ENV_PREFIX + key, default)


def _resolve_logs_dir() -> Path:
    """
    Resolve a logs directory using the library config if available,
    otherwise fall back to ~/.protein_representation/logs.
    """
    # Preferred: ToolConfig.cache_paths.logs() if present
    if _cfg is not None:
        try:
            cfg = _cfg.get_config()
            logs_dir = None
            # Try method 'logs' if implemented
            if hasattr(cfg.cache_paths, "logs") and callable(cfg.cache_paths.logs):
                logs_dir = cfg.cache_paths.logs()
            else:
                # Fallback: cache_root/logs
                logs_dir = Path(cfg.cache_paths.cache_root) / "logs"
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            return Path(logs_dir)
        except Exception:
            pass  # fall back below

    # Fallback without core config available
    root = Path("~/.protein_representation").expanduser().resolve()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter suitable for console or files."""

    def format(self, record: LogRecord) -> str:  # noqa: D401
        base: Dict[str, Any] = {
            "ts": self.formatTime(record),  # uses UTC (see converter above)
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        # Include extra fields if present
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
            }:
                continue
            base[key] = value
        return json.dumps(base, ensure_ascii=False)


class _ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)sZ | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# A singleton guard so we don't add handlers twice
_CONFIGURED_LOGGERS: set[str] = set()


def setup_logger(
    name: str = _DEF_NAME,
    level: int | str = _DEF_LEVEL,
    enable: bool = True,
    json_console: Optional[bool] = None,
    to_stderr: Optional[bool] = None,
    log_file: Optional[str | Path] = None,
    max_bytes: Optional[int] = None,
    backups: Optional[int] = None,
) -> Logger:
    """
    Configure and return a library logger with console + rotating file handlers.

    Parameters
    ----------
    name : str, default="protein_representation"
        Logger name (use a package/module name to benefit from hierarchy).
    level : int | str, default=logging.INFO
        Logging level. Accepts numeric levels or strings (e.g., "DEBUG").
        Can be overridden by env `PR_LOG_LEVEL`.
    enable : bool, default=True
        If False, the logger is returned disabled (no output).
    json_console : bool, optional
        If True, console logs are JSON. Env: `PR_LOG_JSON`. Defaults to False.
    to_stderr : bool, optional
        If True, console logs go to stderr instead of stdout. Env: `PR_LOG_STDERR`.
    log_file : str | Path, optional
        Custom log file path. Env: `PR_LOG_FILE`. Defaults to
        `<logs_dir>/<name>.log`.
    max_bytes : int, optional
        File rotation max bytes. Env: `PR_LOG_MAX_BYTES`. Default 10 MiB.
    backups : int, optional
        Number of backup files to keep. Env: `PR_LOG_BACKUPS`. Default 3.

    Notes
    -----
    - Safe to call multiple times for the same `name`; handlers won't be duplicated.
    - Timestamps are in UTC (suffix 'Z').
    - Honors env `PR_LOG_DISABLE=true` to disable logging globally for this logger.
    """
    # Global disable?
    if _env_bool("DISABLE", False) or not enable:
        logger = logging.getLogger(name)
        logger.disabled = True
        return logger

    # Resolve level (string or int)
    if isinstance(level, str):
        lvl = _LEVEL_MAP.get(level.upper(), _DEF_LEVEL)
    else:
        lvl = int(level)
    lvl = _LEVEL_MAP.get(_env_str("LEVEL", ""), lvl)

    json_console = _env_bool("JSON", _DEF_JSON) if json_console is None else json_console
    to_stderr = _env_bool("STDERR", _DEF_STDERR) if to_stderr is None else to_stderr
    max_bytes = _env_int("MAX_BYTES", _DEF_MAX_BYTES) if max_bytes is None else max_bytes
    backups = _env_int("BACKUPS", _DEF_BACKUPS) if backups is None else backups

    logs_dir = _resolve_logs_dir()
    default_file = logs_dir / f"{name}.log"
    file_path = Path(_env_str("FILE", str(default_file))) if log_file is None else Path(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.propagate = False  # keep output tidy if root is configured elsewhere

    # Avoid duplicate handlers if we've already configured this logger
    if name in _CONFIGURED_LOGGERS:
        return logger

    # --- Console handler ---
    stream = sys.stderr if to_stderr else sys.stdout
    ch = logging.StreamHandler(stream)
    ch.setLevel(lvl)
    ch.setFormatter(_JsonFormatter() if json_console else _ConsoleFormatter())
    logger.addHandler(ch)

    # --- Rotating file handler (always DEBUG to keep full traces) ---
    fh = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=max_bytes,
        backupCount=backups,
        encoding="utf-8",
        delay=True,  # create on first use
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_ConsoleFormatter() if not json_console else _JsonFormatter())
    logger.addHandler(fh)

    _CONFIGURED_LOGGERS.add(name)
    return logger


def get_logger(name: str = _DEF_NAME) -> Logger:
    """
    Return an existing logger (configured or not). If not configured yet,
    call `setup_logger` with defaults first.
    """
    logger = logging.getLogger(name)
    if name not in _CONFIGURED_LOGGERS and not logger.handlers:
        return setup_logger(name=name)
    return logger


# ----------------------------
# Convenience helpers
# ----------------------------

class ContextFilter(logging.Filter):
    """
    Logging filter that injects static context key/values.
    Use `add_context` to attach it to a logger.
    """
    def __init__(self, **ctx: Any) -> None:
        super().__init__()
        self._ctx = dict(ctx)

    def filter(self, record: LogRecord) -> bool:
        for k, v in self._ctx.items():
            setattr(record, k, v)
        return True


def add_context(logger: Logger, **ctx: Any) -> None:
    """
    Add static context (e.g., run_id="abc", module="encoder") to all records of a logger.
    """
    logger.addFilter(ContextFilter(**ctx))


def set_global_level(level: int | str) -> None:
    """
    Update the root logger level (useful in notebooks/tests).
    """
    if isinstance(level, str):
        level = _LEVEL_MAP.get(level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)


def silence_external(names: list[str], level: int = logging.WARNING) -> None:
    """
    Raise levels of noisy external libraries (e.g., 'urllib3', 'huggingface_hub').
    """
    for n in names:
        logging.getLogger(n).setLevel(level)


def reset_logging(name: str = _DEF_NAME) -> None:
    """
    Remove handlers and reset the configuration guard for a given logger.
    Intended for tests.
    """
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    logger.disabled = False
    _CONFIGURED_LOGGERS.discard(name)


__all__ = [
    "setup_logger",
    "get_logger",
    "add_context",
    "set_global_level",
    "silence_external",
    "reset_logging",
]
