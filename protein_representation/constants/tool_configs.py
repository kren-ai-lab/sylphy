from dataclasses import dataclass, field
from .config_constants import CachePaths
from pathlib import Path
import platform
import os
import logging
from typing import Optional

def _default_cache_root() -> Path:

    # OS-specific defaults
    system = platform.system().lower()
    if system == "darwin":      # macOS
        base = Path.home() / "Library" / "Caches"
    elif system == "windows":
        base = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:                       # linux/unix
        base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))

    return base

@dataclass
class ToolConfig:
    cache_paths: CachePaths = field(default_factory=lambda: CachePaths(_default_cache_root()))
    debug : bool = False
    default_device : str = "cuda"
    log_level : int = logging.INFO
    seed : int = 42

# --- Global config singleton (simple & testable) ---
_GLOBAL_CONFIG: Optional[ToolConfig] = None