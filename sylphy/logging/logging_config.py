"""Configure structured and plain-text logging for Sylphy."""

from __future__ import annotations

import json
import logging
import os
import time
from importlib import import_module
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from appdirs import user_log_dir
except ImportError:  # pragma: no cover
    def user_log_dir(
        _appname: str | None = None,
        _appauthor: str | None = None,
        _version: str | None = None,
        *,
        _opinion: bool = True,
    ) -> str:
        """Return an empty fallback log directory when appdirs is unavailable."""
        return ""

# Import lightweight constants/helpers (avoid cycles)
from sylphy.constants.logging_constants import (
    LOG_DEFAULT_BACKUPS,
    LOG_DEFAULT_JSON,
    LOG_DEFAULT_LEVEL,
    LOG_DEFAULT_MAX_BYTES,
    LOG_DEFAULT_NAME,
    LOG_DEFAULT_STDERR,
    LOG_ENV_PREFIX,
    LOG_LEVEL_MAP,
    env_log_json,
    env_log_level,
    env_log_stderr,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Track configured roots to avoid handler duplication across repeated calls.
_CONFIGURED_ROOTS: set[str] = set()

# ---------- helpers ----------


def _env_int(name: str, default: int) -> int:
    """Parse an integer from environment. Returns default on failure."""
    raw = os.getenv(f"{LOG_ENV_PREFIX}{name}")
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, *, default: bool) -> bool:
    """Parse a boolean from environment. Accepts: 1, true, yes, on."""
    raw = os.getenv(f"{LOG_ENV_PREFIX}{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_log_file(default_name: str = "sylphy.log", explicit_path: Path | None = None) -> Path | None:
    """Decide log file path without importing heavy modules at import time."""
    if explicit_path is not None:
        p = Path(explicit_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    env_path = os.getenv(f"{LOG_ENV_PREFIX}FILE")
    if env_path:
        p = Path(env_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    try:
        module = import_module("sylphy.constants.tool_configs")
        get_config = module.get_config
        root = Path(get_config().cache_paths.logs())
        root.mkdir(parents=True, exist_ok=True)
        return root / default_name
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
        logging.getLogger(__name__).debug("Could not resolve log file from tool config.", exc_info=True)

    if user_log_dir is not None:
        base = Path(user_log_dir("sylphy", "Sylphy"))
        base.mkdir(parents=True, exist_ok=True)
        return base / default_name

    return None


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter with safe extras serialization."""

    def __init__(self, *, use_utc: bool = False) -> None:
        super().__init__()
        self._use_utc = use_utc

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        if self._use_utc:
            ct = time.gmtime(record.created)
            return time.strftime("%Y-%m-%dT%H:%M:%S", ct)
        return super().formatTime(record, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
            }:
                continue
            try:
                json.dumps({k: v})
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = str(v)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)


def _make_stream_handler(
    level: int, fmt: str, *, use_json: bool, use_utc: bool, datefmt: str | None,
) -> logging.Handler:
    h = logging.StreamHandler()
    h.setLevel(level)
    if use_json:
        h.setFormatter(_JsonFormatter(use_utc=use_utc))
    else:
        h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    return h


def _make_file_handler(
    path: Path,
    level: int,
    fmt: str,
    *,
    use_json: bool,
    use_utc: bool,
    datefmt: str | None,
    max_bytes: int,
    backups: int,
) -> logging.Handler:
    try:
        fh = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backups,
            encoding="utf-8",
        )
    except OSError:
        fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(level)
    if use_json:
        fh.setFormatter(_JsonFormatter(use_utc=use_utc))
    else:
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    return fh


# ---------- setup_logger decomposition ----------


def _resolve_setup_params(
    level: int | str | None,
    *,
    with_console: bool | None,
    use_json: bool | None,
    use_utc: bool | None,
    max_bytes: int | None,
    backups: int | None,
) -> tuple[int, bool, bool, bool, int, int]:
    if level is None:
        lvl = env_log_level()
    elif isinstance(level, str):
        lvl = LOG_LEVEL_MAP.get(level.upper(), LOG_DEFAULT_LEVEL)
    else:
        lvl = int(level)

    wc = env_log_stderr(default=LOG_DEFAULT_STDERR) if with_console is None else with_console
    uj = env_log_json(default=LOG_DEFAULT_JSON) if use_json is None else use_json
    uu = _env_bool("UTC", default=False) if use_utc is None else use_utc
    mb = _env_int("MAX_BYTES", LOG_DEFAULT_MAX_BYTES) if max_bytes is None else max_bytes
    bk = _env_int("BACKUPS", LOG_DEFAULT_BACKUPS) if backups is None else backups

    return lvl, wc, uj, uu, mb, bk


def _update_existing_handlers(logger: logging.Logger, lvl: int) -> None:
    logger.setLevel(lvl)
    for h in logger.handlers:
        h.setLevel(lvl if isinstance(lvl, int) else LOG_DEFAULT_LEVEL)


def _clear_existing_handlers(logger: logging.Logger, name: str) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _CONFIGURED_ROOTS.discard(name)


def _add_console_handler(
    logger: logging.Logger,
    lvl: int,
    fmt: str,
    *,
    json: bool,
    utc: bool,
    datefmt: str | None,
) -> None:
    ch = _make_stream_handler(lvl, fmt, use_json=json, use_utc=utc, datefmt=datefmt)
    logger.addHandler(ch)


def _add_file_handler(
    logger: logging.Logger,
    file_path: Path | None,
    fmt: str,
    *,
    json: bool,
    utc: bool,
    datefmt: str | None,
    max_bytes: int,
    backups: int,
) -> None:
    path = _resolve_log_file(explicit_path=file_path)
    if path is not None:
        fh = _make_file_handler(
            path,
            level=logging.DEBUG,
            fmt=fmt,
            use_json=json,
            use_utc=utc,
            datefmt=datefmt,
            max_bytes=max_bytes,
            backups=backups,
        )
        logger.addHandler(fh)


# ---------- public API ----------


def setup_logger(
    name: str = LOG_DEFAULT_NAME,
    level: int | str | None = None,
    *,
    with_console: bool | None = None,
    with_file: bool = True,
    file_path: Path | None = None,
    fmt_console: str = "[%(levelname)s] %(message)s",
    fmt_file: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt_console: str | None = None,
    datefmt_file: str | None = "%Y-%m-%d %H:%M:%S",
    use_json: bool | None = None,
    use_utc: bool | None = None,
    max_bytes: int | None = None,
    backups: int | None = None,
    propagate: bool = False,
    force_reconfigure: bool = False,
    extra_filters: Iterable[logging.Filter] | None = None,
) -> logging.Logger:
    """Configure and return the package root logger."""
    lvl, wc, uj, uu, mb, bk = _resolve_setup_params(
        level,
        with_console=with_console,
        use_json=use_json,
        use_utc=use_utc,
        max_bytes=max_bytes,
        backups=backups,
    )

    logger = logging.getLogger(name)
    logger.propagate = propagate

    already_configured = name in _CONFIGURED_ROOTS
    if already_configured and not force_reconfigure:
        _update_existing_handlers(logger, lvl)
        return logger

    if already_configured and force_reconfigure:
        _clear_existing_handlers(logger, name)

    logger.setLevel(lvl)

    if wc:
        _add_console_handler(logger, lvl, fmt_console, json=uj, utc=uu, datefmt=datefmt_console)

    if with_file:
        _add_file_handler(
            logger, file_path, fmt_file, json=uj, utc=uu, datefmt=datefmt_file, max_bytes=mb, backups=bk,
        )

    if extra_filters:
        for flt in extra_filters:
            logger.addFilter(flt)

    _CONFIGURED_ROOTS.add(name)
    return logger


def get_logger(name: str = LOG_DEFAULT_NAME) -> logging.Logger:
    """Return the configured root logger. If not configured, configure with sane defaults."""
    if name not in _CONFIGURED_ROOTS:
        setup_logger(name=name)
    return logging.getLogger(name)


class _ContextFilter(logging.Filter):
    """Static key-value context injector."""

    def __init__(self, **static_context: object) -> None:
        super().__init__()
        self._ctx = static_context

    def filter(self, record: logging.LogRecord) -> bool:
        for k, v in self._ctx.items():
            setattr(record, k, v)
        return True


def add_context(logger: logging.Logger, **context: object) -> None:
    """Attach static context to a logger."""
    if not context:
        return
    logger.addFilter(_ContextFilter(**context))


def set_global_level(level: int | str, name: str = LOG_DEFAULT_NAME) -> None:
    """Change the level of the root logger and its handlers."""
    lvl = LOG_LEVEL_MAP.get(level.upper(), LOG_DEFAULT_LEVEL) if isinstance(level, str) else int(level)
    logger = get_logger(name)
    logger.setLevel(lvl)
    for h in logger.handlers:
        h.setLevel(lvl)


def silence_external() -> None:
    """Lower verbosity of common external libraries."""
    for lib in ("urllib3", "transformers", "accelerate", "httpx"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def reset_logging(name: str = LOG_DEFAULT_NAME) -> None:
    """Remove all handlers for the given logger name and mark it as unconfigured."""
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _CONFIGURED_ROOTS.discard(name)
