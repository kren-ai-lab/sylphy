# core/config.py
from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from sylphy.constants.config_constants import CachePaths
from sylphy.constants.tool_configs import ToolConfig
from sylphy.constants.tool_configs import get_config as _get_config
from sylphy.constants.tool_configs import set_config as _set_config
from sylphy.logging import get_logger

_LOG = get_logger(__name__)
_LOCK = RLock()


def get_config() -> ToolConfig:
    """
    Return the global ToolConfig managed by sylphy.constants.tool_configs.

    This is a thin wrapper to avoid duplicating global state in the 'core' package.
    """
    with _LOCK:
        cfg = _get_config()
        # tool_configs.get_config() already ensures cache directories exist.
        return cfg


def set_cache_root(new_root: Path | str) -> None:
    """
    Programmatically override the cache root directory while preserving
    the rest of the current configuration (debug, device, log_level, seed).

    Parameters
    ----------
    new_root : Path or str
        New root path. Will be expanded and resolved.
    """
    root = Path(new_root).expanduser().resolve()
    with _LOCK:
        old = _get_config()
        new_cfg = ToolConfig(
            cache_paths=CachePaths(root),
            debug=old.debug,
            default_device=old.default_device,
            log_level=old.log_level,
            seed=old.seed,
        )
        _set_config(new_cfg)
        _LOG.info("Cache root set to: %s", root)


@contextmanager
def temporary_cache_root(temp_root: Path | str) -> Generator[None, None, None]:
    """
    Temporarily override the cache root (useful for tests or isolated runs).

    Example
    -------
    >>> from pathlib import Path
    >>> with temporary_cache_root(Path('./.tmp_cache')):
    ...     # operations here use the temporary cache
    ...     pass
    """
    prev_root = get_config().cache_paths.cache_root
    set_cache_root(temp_root)
    try:
        yield
    finally:
        set_cache_root(prev_root)


__all__ = ["get_config", "set_cache_root", "temporary_cache_root", "ToolConfig", "CachePaths"]
