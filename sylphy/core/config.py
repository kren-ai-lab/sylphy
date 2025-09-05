# core/config.py
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Generator

from sylphy.constants.config_constants import _LOCK as _EXTERNAL_LOCK, CachePaths
from sylphy.constants.tool_configs import ToolConfig, _GLOBAL_CONFIG

# Local lock to guard lazy initialization in this module.
_LOCK = _EXTERNAL_LOCK or RLock()


def get_config() -> ToolConfig:
    """
    Return the global ToolConfig (lazy-initialized).

    The first call initializes a default ``ToolConfig`` and ensures the cache
    directories exist.
    """
    global _GLOBAL_CONFIG
    with _LOCK:
        if _GLOBAL_CONFIG is None:
            _GLOBAL_CONFIG = ToolConfig()
            _GLOBAL_CONFIG.cache_paths.ensure_all()
        return _GLOBAL_CONFIG


def set_cache_root(new_root: Path | str) -> None:
    """
    Programmatically override the cache root directory.

    Parameters
    ----------
    new_root : Path or str
        New root path. Will be expanded and resolved.

    Notes
    -----
    - Updates the global configuration and ensures directories exist.
    """
    global _GLOBAL_CONFIG
    root = Path(new_root).expanduser().resolve()
    with _LOCK:
        _GLOBAL_CONFIG = ToolConfig(cache_paths=CachePaths(root))
        _GLOBAL_CONFIG.cache_paths.ensure_all()


@contextmanager
def temporary_cache_root(temp_root: Path | str) -> Generator[None, None, None]:
    """
    Context manager to temporarily override the cache root (useful for tests).

    Example
    -------
    >>> from pathlib import Path
    >>> with temporary_cache_root(Path("./.tmp_cache")):
    ...     # resolve models into ./.tmp_cache
    ...     pass
    """
    old = get_config().cache_paths.cache_root
    set_cache_root(temp_root)
    try:
        yield
    finally:
        set_cache_root(old)


__all__ = ["get_config", "set_cache_root", "temporary_cache_root", "ToolConfig", "CachePaths"]
