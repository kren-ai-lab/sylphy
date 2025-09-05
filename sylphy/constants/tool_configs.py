from dataclasses import dataclass, field
from pathlib import Path
import os, platform
from typing import Optional
from .config_constants import CachePaths
from .logging_constants import env_log_level

def _default_cache_root() -> Path:
    system = platform.system().lower()
    if system == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif system == "windows":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    env = os.getenv("SYLPHY_CACHE_ROOT")
    return Path(env) if env else base

def _default_device() -> str:
    return os.getenv("SYLPHY_DEVICE", "cuda")


@dataclass
class ToolConfig:
    cache_paths: CachePaths = field(default_factory=lambda: CachePaths(_default_cache_root()))
    debug: bool = False
    default_device: str = field(default_factory=_default_device)
    log_level: int = field(default_factory=env_log_level)
    seed: int = int(os.getenv("SYLPHY_SEED", "42"))


# --- Global config singleton (simple & testable) ---
_GLOBAL: Optional[ToolConfig] = None

def get_config() -> ToolConfig:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = ToolConfig()
        _GLOBAL.cache_paths.ensure_all()
    return _GLOBAL

def set_config(cfg: ToolConfig) -> None:
    global _GLOBAL
    _GLOBAL = cfg
    _GLOBAL.cache_paths.ensure_all()
