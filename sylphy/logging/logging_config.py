from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Optional, Any, Iterable

try:
    from appdirs import user_log_dir
except Exception:  # pragma: no cover
    user_log_dir = None  # type: ignore

# Track configured roots to avoid handler duplication across repeated calls.
_CONFIGURED_ROOTS: set[str] = set()

# Import lightweight constants/helpers (avoid cycles)
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


# ---------- helpers ----------

def _env_int(name: str, default: int) -> int:
    """
    Parse an integer from environment. Returns default on failure.
    """
    raw = os.getenv(f"{LOG_ENV_PREFIX}{name}")
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """
    Parse a boolean from environment. Accepts: 1, true, yes, on.
    """
    raw = os.getenv(f"{LOG_ENV_PREFIX}{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_log_file(default_name: str = "sylphy.log",
                      explicit_path: Optional[Path] = None) -> Optional[Path]:
    """
    Decide log file path without importing heavy modules at import time.

    Priority:
      1) explicit argument `explicit_path`
      2) env SYLPHY_LOG_FILE
      3) sylphy.constants.tool_configs.get_config().cache_paths.logs()  (lazy import)
      4) appdirs user_log_dir()
      5) None (no file handler)
    """
    # 1) explicit path argument
    if explicit_path is not None:
        p = Path(explicit_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 2) env override
    env_path = os.getenv(f"{LOG_ENV_PREFIX}FILE")
    if env_path:
        p = Path(env_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # 3) Lazy import to avoid cycles
    try:
        from sylphy.constants.tool_configs import get_config  # local import
        root = Path(get_config().cache_paths.logs())
        root.mkdir(parents=True, exist_ok=True)
        return root / default_name
    except Exception:
        pass

    # 4) Fallback to appdirs
    if user_log_dir is not None:
        base = Path(user_log_dir("sylphy", "Sylphy"))
        base.mkdir(parents=True, exist_ok=True)
        return base / default_name

    # 5) give up: no file
    return None


class _JsonFormatter(logging.Formatter):
    """
    Minimal JSON formatter with safe extras serialization and optional UTC timestamps.
    """

    def __init__(self, *, use_utc: bool = False) -> None:
        super().__init__()
        self._use_utc = use_utc

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:  # type: ignore[override]
        # ISO-ish time; respect UTC flag
        if self._use_utc:
            import time
            ct = time.gmtime(record.created)
            return time.strftime("%Y-%m-%dT%H:%M:%S", ct)
        return super().formatTime(record, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Attach extra context fields (added via filters or loggerAdapter)
        for k, v in record.__dict__.items():
            if k in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "msg", "name", "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName",
            }:
                continue
            # Keep only JSON-safe values; fallback to str.
            try:
                json.dumps({k: v})
                payload[k] = v
            except Exception:
                payload[k] = str(v)

        # Exception info, if any
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)  # type: ignore[arg-type]
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False)


def _make_stream_handler(level: int, fmt: str, *,
                         use_json: bool, use_utc: bool, datefmt: Optional[str]) -> logging.Handler:
    h = logging.StreamHandler()
    h.setLevel(level)
    if use_json:
        h.setFormatter(_JsonFormatter(use_utc=use_utc))
    else:
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        h.setFormatter(formatter)
    return h


def _make_file_handler(path: Path, level: int, fmt: str, *,
                       use_json: bool, use_utc: bool, datefmt: Optional[str],
                       max_bytes: int, backups: int) -> logging.Handler:
    # Prefer rotating handler; gracefully fallback to simple file handler.
    try:
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backups,
            encoding="utf-8",
        )
    except Exception:
        fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(level)
    if use_json:
        fh.setFormatter(_JsonFormatter(use_utc=use_utc))
    else:
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    return fh


# ---------- public API ----------

def setup_logger(
    name: str = LOG_DEFAULT_NAME,
    level: int | str | None = None,
    *,
    with_console: bool | None = None,
    with_file: bool = True,
    file_path: Optional[Path] = None,
    fmt_console: str = "[%(levelname)s] %(message)s",
    fmt_file: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt_console: Optional[str] = None,
    datefmt_file: Optional[str] = "%Y-%m-%d %H:%M:%S",
    use_json: Optional[bool] = None,
    use_utc: Optional[bool] = None,
    max_bytes: Optional[int] = None,
    backups: Optional[int] = None,
    propagate: bool = False,
    force_reconfigure: bool = False,
    extra_filters: Optional[Iterable[logging.Filter]] = None,
) -> logging.Logger:
    """
    Configure and return the package root logger.

    Idempotent per `name`: subsequent calls update the level and (optionally) reconfigure
    handlers if `force_reconfigure=True`.

    Parameters
    ----------
    name : str
        Logger name (package root).
    level : int | str | None
        Logging level (int or textual). If None, read from env (SYLPHY_LOG_LEVEL).
    with_console : bool | None
        If None, read from env (SYLPHY_LOG_STDERR). If True, add STDERR handler.
    with_file : bool
        Whether to add a file handler (rotating if possible).
    file_path : Optional[Path]
        Force a specific file path. Otherwise resolved by _resolve_log_file.
    fmt_console : str
        Format string for console logs (ignored if JSON).
    fmt_file : str
        Format string for file logs (ignored if JSON).
    datefmt_console : Optional[str]
        Date format for console logs (ignored if JSON).
    datefmt_file : Optional[str]
        Date format for file logs (ignored if JSON).
    use_json : Optional[bool]
        If None, read from env (SYLPHY_LOG_JSON). If True, use JSON formatter.
    use_utc : Optional[bool]
        If None, read from env (SYLPHY_LOG_UTC). If True, timestamps in UTC.
    max_bytes : Optional[int]
        Max bytes per log file; if None, read from env (SYLPHY_LOG_MAX_BYTES).
    backups : Optional[int]
        Number of rotated backups; if None, read from env (SYLPHY_LOG_BACKUPS).
    propagate : bool
        Whether the logger should propagate to parent loggers.
    force_reconfigure : bool
        If True, remove existing handlers and re-add with current parameters.
    extra_filters : Optional[Iterable[logging.Filter]]
        Additional filters to attach to the logger (e.g., context filters).

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    # Resolve defaults from env if not provided
    if level is None:
        lvl = env_log_level()
    elif isinstance(level, str):
        lvl = LOG_LEVEL_MAP.get(level.upper(), LOG_DEFAULT_LEVEL)
    else:
        lvl = int(level)

    if with_console is None:
        with_console = env_log_stderr(LOG_DEFAULT_STDERR)

    if use_json is None:
        use_json = env_log_json(LOG_DEFAULT_JSON)

    if use_utc is None:
        use_utc = _env_bool("UTC", False)

    if max_bytes is None:
        max_bytes = _env_int("MAX_BYTES", LOG_DEFAULT_MAX_BYTES)

    if backups is None:
        backups = _env_int("BACKUPS", LOG_DEFAULT_BACKUPS)

    logger = logging.getLogger(name)
    logger.propagate = propagate

    already_configured = name in _CONFIGURED_ROOTS

    if already_configured and not force_reconfigure:
        # Just update levels across existing handlers.
        logger.setLevel(lvl)
        for h in logger.handlers:
            h.setLevel(lvl if isinstance(lvl, int) else LOG_DEFAULT_LEVEL)
        return logger

    # (Re)configure from scratch
    if already_configured and force_reconfigure:
        for h in list(logger.handlers):
            logger.removeHandler(h)
        _CONFIGURED_ROOTS.discard(name)

    logger.setLevel(lvl)

    # Console handler (stderr)
    if with_console:
        ch = _make_stream_handler(
            lvl, fmt_console, use_json=bool(use_json), use_utc=bool(use_utc),
            datefmt=datefmt_console,
        )
        logger.addHandler(ch)

    # File handler
    if with_file:
        path = _resolve_log_file(explicit_path=file_path)
        if path is not None:
            fh = _make_file_handler(
                path, level=logging.DEBUG,  # keep file verbose
                fmt=fmt_file, use_json=bool(use_json), use_utc=bool(use_utc),
                datefmt=datefmt_file, max_bytes=int(max_bytes), backups=int(backups),
            )
            logger.addHandler(fh)

    # Attach any extra filters
    if extra_filters:
        for flt in extra_filters:
            logger.addFilter(flt)

    _CONFIGURED_ROOTS.add(name)
    return logger


def get_logger(name: str = LOG_DEFAULT_NAME) -> logging.Logger:
    """
    Return the configured root logger. If not configured, configure with sane defaults.
    """
    if name not in _CONFIGURED_ROOTS:
        setup_logger(name=name)
    return logging.getLogger(name)


class _ContextFilter(logging.Filter):
    """
    Static key-value context injector. Useful for adding component/model identifiers.
    """
    def __init__(self, **static_context: Any) -> None:
        super().__init__()
        self._ctx = static_context

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        for k, v in self._ctx.items():
            setattr(record, k, v)
        return True


def add_context(logger: logging.Logger, **context: Any) -> None:
    """
    Attach static context (e.g., component='encoder', model='esm2') to a logger.
    """
    if not context:
        return
    logger.addFilter(_ContextFilter(**context))


def set_global_level(level: int | str, name: str = LOG_DEFAULT_NAME) -> None:
    """
    Change the level of the root logger and its handlers.
    """
    if isinstance(level, str):
        lvl = LOG_LEVEL_MAP.get(level.upper(), LOG_DEFAULT_LEVEL)
    else:
        lvl = int(level)
    logger = get_logger(name)
    logger.setLevel(lvl)
    for h in logger.handlers:
        h.setLevel(lvl)


def silence_external() -> None:
    """
    Lower verbosity of common external libraries.
    """
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def reset_logging(name: str = LOG_DEFAULT_NAME) -> None:
    """
    Remove all handlers for the given logger name and mark it as unconfigured.
    Useful for test teardown or dynamic reconfiguration.
    """
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    _CONFIGURED_ROOTS.discard(name)
