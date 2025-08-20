# core/config.py
from __future__ import annotations
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import threading
import logging

_LOCK = threading.RLock()

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
class CachePaths:
    cache_root: Path
    tool_name: str = "bioclust"

    def base(self) -> Path:
        return self.cache_root / self.tool_name

    # Subtrees
    def models(self) -> Path:
        return self.base() / "models"

    def hf_model_dir(self, org: str, model: str) -> Path:
        return self.models() / "huggingface" / org / model

    def other_model_dir(self, provider: str, name: str) -> Path:
        return self.models() / "other" / provider / name

    def structures(self) -> Path:
        return self.base() / "structures"

    def structures_pdb(self) -> Path:
        return self.structures() / "pdb"

    def structures_afdb(self) -> Path:
        return self.structures() / "alphafold"

    def structures_predictions(self, predictor: str = "esmfold") -> Path:
        return self.structures() / "predictions" / predictor

    def data(self) -> Path:
        return self.base() / "data"

    def datasets(self) -> Path:
        return self.base() / "datasets"
    
    def tmp(self) -> Path:
        return self.base() / "tmp"

    def logs(self) -> Path:
        return self.base() / "logs"

    def ensure_all(self) -> None:
        with _LOCK:
            for p in [
                self.base(), self.models(), self.structures(),
                self.structures_pdb(), self.structures_afdb(),
                self.structures_predictions(), self.data(), self.tmp(), self.logs()
            ]:
                p.mkdir(parents=True, exist_ok=True)

@dataclass
class ToolConfig:
    cache_paths: CachePaths = field(default_factory=lambda: CachePaths(_default_cache_root()))
    debug : bool = False
    default_device : str = "cuda"
    log_level : int = logging.INFO
    seed : int = 42

# --- Global config singleton (simple & testable) ---
_GLOBAL_CONFIG: Optional[ToolConfig] = None

def get_config() -> ToolConfig:
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = ToolConfig()
        _GLOBAL_CONFIG.cache_paths.ensure_all()
    return _GLOBAL_CONFIG

def set_cache_root(new_root: Path | str) -> None:
    """Programmatic override of the cache root."""
    global _GLOBAL_CONFIG
    with _LOCK:
        root = Path(new_root).expanduser().resolve()
        _GLOBAL_CONFIG = ToolConfig(cache_paths=CachePaths(root))
        _GLOBAL_CONFIG.cache_paths.ensure_all()

@contextmanager
def temporary_cache_root(temp_root: Path | str):
    """Context manager for tests: use a temporary cache and revert back."""
    old = get_config().cache_paths.cache_root
    set_cache_root(temp_root)
    try:
        yield
    finally:
        set_cache_root(old)
