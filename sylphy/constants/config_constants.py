"""Define cache path helpers shared across configuration code."""

import threading
from dataclasses import dataclass
from pathlib import Path

# Thread-safe creation of cache directories
_LOCK = threading.RLock()


@dataclass
class CachePaths:
    """Helper to manage sylphy cache layout and ensure directories exist."""

    cache_root: Path
    tool_name: str = "sylphy"

    def base(self) -> Path:
        """Return the root directory for this tool's cache tree."""
        return self.cache_root / self.tool_name

    # Subtrees
    def models(self) -> Path:
        """Return the directory that stores model artifacts."""
        return self.base() / "models"

    def other_model_dir(self, provider: str, name: str) -> Path:
        """Return a path for a non-HF model cache directory."""
        return self.models() / "other" / provider / name

    def data(self) -> Path:
        """Return the directory that stores cached data outputs."""
        return self.base() / "data"

    def tmp(self) -> Path:
        """Return the directory used for temporary cache files."""
        return self.base() / "tmp"

    def logs(self) -> Path:
        """Return the directory that stores log files."""
        return self.base() / "logs"

    def ensure_all(self) -> None:
        """Create the common cache directories if they do not exist."""
        with _LOCK:
            for p in [self.base(), self.models(), self.data(), self.tmp(), self.logs()]:
                p.mkdir(parents=True, exist_ok=True)
