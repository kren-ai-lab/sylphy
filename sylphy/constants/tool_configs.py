# tool_configs.py
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

from .config_constants import CachePaths
from .logging_constants import env_log_level

DEFAULT_CACHE_APP = "sylphy"
DEFAULT_CACHE_ROOT_ENV = "SYLPHY_CACHE_ROOT"
DEFAULT_CACHE_DIR_ENV = "SYLPHY_CACHE_DIR"

def _default_cache_parent() -> Path:
    """Return the platform-specific parent directory used for application caches."""
    system = platform.system().lower()
    if system == "darwin":
        return Path.home() / "Library" / "Caches"
    if system == "windows":
        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))


def resolve_cache_dir() -> Path:
    """Resolve the final Sylphy cache directory from environment or platform defaults."""
    env_dir = os.getenv(DEFAULT_CACHE_DIR_ENV)
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    env_root = os.getenv(DEFAULT_CACHE_ROOT_ENV)
    if env_root:
        return (Path(env_root).expanduser().resolve() / DEFAULT_CACHE_APP).resolve()

    return (_default_cache_parent() / DEFAULT_CACHE_APP).expanduser().resolve()


def default_cache_paths() -> CachePaths:
    """Build cache paths from the resolved final cache directory."""
    cache_dir = resolve_cache_dir()
    return CachePaths(cache_dir.parent, tool_name=cache_dir.name)


def _detect_cuda_available() -> bool:
    """Best-effort CUDA availability check without hard dependency on torch.
    Returns False if torch cannot be imported.
    """
    try:
        import torch  # noqa: PLC0415

        return bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
    except Exception:
        return False


def _default_device() -> str:
    """Default device to use for model-backed features.
    Priority:
      1) Respect SYLPHY_DEVICE if set (e.g., 'cpu', 'cuda').
      2) Use 'cuda' only if available.
      3) Fallback to 'cpu'.
    """
    env_device = os.getenv("SYLPHY_DEVICE")
    if env_device:
        return env_device

    if _detect_cuda_available():
        return "cuda"
    return "cpu"


@dataclass
class ToolConfig:
    """Global configuration container for sylphy runtime."""

    cache_paths: CachePaths = field(default_factory=default_cache_paths)
    debug: bool = False
    default_device: str = field(default_factory=_default_device)
    log_level: int = field(default_factory=env_log_level)
    seed: int = int(os.getenv("SYLPHY_SEED", "42"))


# Global singleton for convenience (simple and testable)
_GLOBAL: ToolConfig | None = None


def get_config() -> ToolConfig:
    """Return the global ToolConfig, creating it on first use.
    Ensures cache directories exist.
    """
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = ToolConfig()
        _GLOBAL.cache_paths.ensure_all()
    return _GLOBAL


def set_config(cfg: ToolConfig) -> None:
    """Replace the global ToolConfig with a custom instance.
    Ensures cache directories exist.
    """
    global _GLOBAL
    _GLOBAL = cfg
    _GLOBAL.cache_paths.ensure_all()
