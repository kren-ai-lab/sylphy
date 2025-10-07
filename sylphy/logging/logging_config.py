from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Optional, Any

try:
    from appdirs import user_log_dir
except Exception:  # pragma: no cover
    user_log_dir = None  # type: ignore

# Track configured roots to avoid handler duplication
_CONFIGURED_ROOTS: set[str] = set()

# Import only lightweight constants/helpers (no cycles)
from sylphy.constants.logging_constants import (
    LOG_ENV_PREFIX,                     # "SYLPHY_LOG_"
    LOG_DEFAULT_NAME,                   # "sylphy"
    LOG_DEFAULT_LEVEL,                  # logging.INFO
    LOG_DEFAULT_JSON,                   # False
    LOG_DEFAULT_STDERR,                 # False
    LOG_DEFAULT_MAX_BYTES,              # 10 MB
    LOG_DEFAULT_BACKUPS,                # 3
    LOG_LEVEL_MAP,                      # str->level
    env_log_level, env_log_json, env_log_stderr,
)


def _resolve_log_file(default_name: str = "sylphy.log") -> Optional[Path]:
    """
    Decide log file path without importing heavy modules at import time.

    Priority:
      1) env SYLPHY_LOG_FILE
      2) sylphy.constants.tool_configs.get_config().cache_paths.logs()  (lazy import)
      3) appdirs user_log_dir()
      4) None (no file handler)
    """
    # 1) explicit env
    env_path = os.getenv(f"{LOG_ENV_PREFIX}FILE")
    if env_path:
        p = Path(env_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) Lazy import to avoid cycles
    try:
        from sylphy.constants.tool_configs import get_config  # local import
        root = Path(get_config().cache_paths.logs())
        root.mkdir(parents=True, exist_ok=True)
        return root / default_name
    except Exception:
        pass

    # 3) Fallback to appdirs
    if user_log_dir is not None:
        base = Path(user_log_dir("sylphy", "Sylphy"))
        base.mkdir(parents=True, exist_ok=True)
        return base / default_name

    # 4) give up: no file
    return None


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Attach extra context fields (added via filters)
        for k, v in record.__dict__.items():
            if k in ("args", "asctime", "created", "exc_info", "exc_text", "filename",
                     "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                     "msg", "name", "pathname", "process", "processName", "relativeCreated",
                     "stack_info", "thread", "threadName"):
                continue
            # only simple JSON-safe values
            try:
                json.dumps({k: v})
                payload[k] = v
            except Exception:
                payload[k] = str(v)
        return json.dumps(payload, ensure_ascii=False)


def setup_logger(
    name: str = LOG_DEFAULT_NAME,
    level: int | str | None = None,
    *,
    with_console: bool | None = None,
    with_file: bool = True,
    fmt_console: str = "[%(levelname)s] %(message)s",
    fmt_file: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> logging.Logger:
    """
    Configure the package root logger exactly once.
    Idempotent for the same `name`; subsequent calls update the level.
    """
    # Resolve defaults from env if not provided
    if level is None:
        level = env_log_level()
    elif isinstance(level, str):
        level = LOG_LEVEL_MAP.get(level.upper(), LOG_DEFAULT_LEVEL)

    if with_console is None:
        with_console = env_log_stderr(LOG_DEFAULT_STDERR)

    use_json = env_log_json(LOG_DEFAULT_JSON)

    logger = logging.getLogger(name)
    if name in _CONFIGURED_ROOTS:
        logger.setLevel(level)
        # also update handler levels
        for h in logger.handlers:
            h.setLevel(level if isinstance(level, int) else LOG_DEFAULT_LEVEL)
        return logger

    logger.setLevel(level)

    # Console handler (stderr)
    if with_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(_JsonFormatter() if use_json else logging.Formatter(fmt_console))
        logger.addHandler(ch)

    # File handler (rotating, verbose)
    if with_file:
        log_path = _resolve_log_file()
        if log_path is not None:
            try:
                from logging.handlers import RotatingFileHandler
                fh = RotatingFileHandler(
                    log_path,
                    maxBytes=LOG_DEFAULT_MAX_BYTES,
                    backupCount=LOG_DEFAULT_BACKUPS,
                    encoding="utf-8",
                )
            except Exception:
                fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)  # keep file verbose
            fh.setFormatter(_JsonFormatter() if use_json else logging.Formatter(fmt_file))
            logger.addHandler(fh)

    _CONFIGURED_ROOTS.add(name)
    return logger


def get_logger(name: str = LOG_DEFAULT_NAME) -> logging.Logger:
    """Return the configured root logger. If missing, configure with sane defaults."""
    if name not in _CONFIGURED_ROOTS:
        setup_logger(name=name)
    return logging.getLogger(name)


class _ContextFilter(logging.Filter):
    def __init__(self, **static_context: Any) -> None:
        super().__init__()
        self._ctx = static_context

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        for k, v in self._ctx.items():
            setattr(record, k, v)
        return True


def add_context(logger: logging.Logger, **context: Any) -> None:
    """Attach static context (component=..., encoder=..., model=...) to a logger."""
    if not context:
        return
    logger.addFilter(_ContextFilter(**context))


def set_global_level(level: int | str, name: str = LOG_DEFAULT_NAME) -> None:
    """Change the level of the root logger and its handlers."""
    if isinstance(level, str):
        level = LOG_LEVEL_MAP.get(level.upper(), LOG_DEFAULT_LEVEL)
    logger = get_logger(name)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)


def silence_external() -> None:
    """Lower noisy external libraries."""
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def reset_logging(name: str = LOG_DEFAULT_NAME) -> None:
    """Remove handlers from the root logger (for test teardown)."""
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _CONFIGURED_ROOTS.discard(name)
