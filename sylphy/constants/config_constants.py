import threading
from pathlib import Path
from dataclasses import dataclass

_LOCK = threading.RLock()

@dataclass
class CachePaths:
    cache_root: Path
    tool_name: str = "sylphy"

    def base(self) -> Path:
        return self.cache_root / self.tool_name

    # Subtrees
    def models(self) -> Path:
        return self.base() / "models"

    def hf_model_dir(self, org: str, model: str, revision: str | None = None) -> Path:
        p = self.models() / "huggingface" / org / model
        return p if not revision else p / revision

    def other_model_dir(self, provider: str, name: str) -> Path:
        return self.models() / "other" / provider / name
    
    def data(self) -> Path:
        return self.base() / "data"
    
    def tmp(self) -> Path:
        return self.base() / "tmp"

    def logs(self) -> Path:
        return self.base() / "logs"

    def ensure_all(self) -> None:
        with _LOCK:
            for p in [
                self.base(), self.models(), self.data(), self.tmp(), self.logs()
            ]:
                p.mkdir(parents=True, exist_ok=True)