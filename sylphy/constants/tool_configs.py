# tool_configs.py
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

from .config_constants import CachePaths
from .logging_constants import env_log_level


def _default_cache_root() -> Path:
    """
    Determine the default cache root honoring SYLPHY_CACHE_ROOT if set
    and following OS-specific conventions otherwise.
    """
    system = platform.system().lower()
    if system == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif system == "windows":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))

    env = os.getenv("SYLPHY_CACHE_ROOT")
    return Path(env) if env else base


def _detect_cuda_available() -> bool:
    """
    Best-effort CUDA availability check without hard dependency on torch.
    Returns False if torch cannot be imported.
    """
    try:
        import torch  # type: ignore

        return bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()
    except Exception:
        return False


def _default_device() -> str:
    """
    Default device to use for model-backed features.
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
    """
    Global configuration container for sylphy runtime.
    """

    cache_paths: CachePaths = field(default_factory=lambda: CachePaths(_default_cache_root()))
    debug: bool = False
    default_device: str = field(default_factory=_default_device)
    log_level: int = field(default_factory=env_log_level)
    seed: int = int(os.getenv("SYLPHY_SEED", "42"))


# Global singleton for convenience (simple and testable)
_GLOBAL: ToolConfig | None = None


def get_config() -> ToolConfig:
    """
    Return the global ToolConfig, creating it on first use.
    Ensures cache directories exist.
    """
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = ToolConfig()
        _GLOBAL.cache_paths.ensure_all()
    return _GLOBAL


def set_config(cfg: ToolConfig) -> None:
    """
    Replace the global ToolConfig with a custom instance.
    Ensures cache directories exist.
    """
    global _GLOBAL
    _GLOBAL = cfg
    _GLOBAL.cache_paths.ensure_all()
