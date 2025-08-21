# protein_representation/logging_config.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from appdirs import user_log_dir
except Exception:  # pragma: no cover
    user_log_dir = None  # type: ignore

# Keep track of configured root names to avoid handler duplication
_CONFIGURED_ROOTS: set[str] = set()

_LEVEL_MAP: Dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _resolve_log_file(default_name: str = "protein_representation.log") -> Optional[Path]:
    """
    Decide the log file path without importing core at module import time.

    Priority:
    1) env PR_LOG_FILE
    2) core.config.get_config().cache_paths.logs()  (lazy imported, best-effort)
    3) appdirs user_log_dir()
    4) None (no file handler)
    """
    # 1) explicit env
    env_path = os.getenv("PR_LOG_FILE")
    if env_path:
        p = Path(env_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) Try core.config lazily (break cycles: no top-level import!)
    try:
        from protein_representation.core import config as _cfg  # local import
        root = Path(_cfg.get_config().cache_paths.logs())
        root.mkdir(parents=True, exist_ok=True)
        return root / default_name
    except Exception:
        pass

    # 3) Fallback to appdirs
    if user_log_dir is not None:
        base = Path(user_log_dir("protein_representation", "Kren AI Lab"))
        base.mkdir(parents=True, exist_ok=True)
        return base / default_name

    # 4) give up: no file
    return None


def setup_logger(
    name: str = "protein_representation",
    level: int | str = "INFO",
    *,
    with_console: bool = True,
    with_file: bool = True,
    fmt_console: str = "[%(levelname)s] %(message)s",
    fmt_file: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
) -> logging.Logger:
    """
    Configure the **root** logger for the package exactly once.

    Subsequent calls with the same `name` are idempotent (no extra handlers).
    Child loggers created via `logging.getLogger(f"{name}.something")`
    will inherit handlers and levels unless explicitly overridden.
    """
    if isinstance(level, str):
        level = _LEVEL_MAP.get(level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    if name in _CONFIGURED_ROOTS:
        # Already configured; allow level bump/downgrade
        logger.setLevel(level)
        return logger

    logger.setLevel(level)

    if with_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(fmt_console))
        logger.addHandler(ch)

    if with_file:
        log_path = _resolve_log_file()
        if log_path is not None:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)  # keep file verbose
            fh.setFormatter(logging.Formatter(fmt_file))
            logger.addHandler(fh)

    _CONFIGURED_ROOTS.add(name)
    return logger


def get_logger(name: str = "protein_representation") -> logging.Logger:
    """
    Return the configured root logger. If not configured yet, set a sane default.
    """
    if name not in _CONFIGURED_ROOTS:
        setup_logger(name=name, level="INFO")
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
    """
    Attach static context (component=..., encoder=..., model=...) to a logger.
    """
    if not context:
        return
    logger.addFilter(_ContextFilter(**context))


def set_global_level(level: int | str, name: str = "protein_representation") -> None:
    """
    Change the level of the root logger and its handlers.
    """
    if isinstance(level, str):
        level = _LEVEL_MAP.get(level.upper(), logging.INFO)
    logger = get_logger(name)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)


def silence_external() -> None:
    """
    Lower noisy external libraries.
    """
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)


def reset_logging(name: str = "protein_representation") -> None:
    """
    Remove handlers from the root logger (for test teardown).
    """
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _CONFIGURED_ROOTS.discard(name)
