"""Provide global runtime configuration and cache resolution helpers."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir

from .config_constants import CachePaths
from .logging_constants import env_log_level

DEFAULT_CACHE_APP = "sylphy"
DEFAULT_CACHE_ROOT_ENV = "SYLPHY_CACHE_ROOT"
DEFAULT_CACHE_DIR_ENV = "SYLPHY_CACHE_DIR"

_logger = logging.getLogger(__name__)


def _wire_cache_envs(root: Path) -> None:
    """Set HF/Torch env vars to subdirs of the Sylphy cache root (only if not already set)."""
    root = Path(root).expanduser()
    os.environ.setdefault("SYLPHY_CACHE_DIR", str(root))
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "hf" / "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf" / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_cache_dir() -> Path:
    """Resolve the final Sylphy cache directory from environment or platform defaults."""
    env_dir = os.getenv(DEFAULT_CACHE_DIR_ENV)
    if env_dir:
        return Path(env_dir).expanduser().resolve()

    env_root = os.getenv(DEFAULT_CACHE_ROOT_ENV)
    if env_root:
        return (Path(env_root).expanduser().resolve() / DEFAULT_CACHE_APP).resolve()

    return Path(user_cache_dir(DEFAULT_CACHE_APP))


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
    except ImportError:
        return False
    except Exception as e:  # noqa: BLE001
        _logger.debug("Unexpected error checking CUDA availability: %s", e)
        return False
    else:
        return bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()


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


class _ConfigStore:
    """Internal container for the global ToolConfig singleton."""

    _instance: ToolConfig | None = None

    @classmethod
    def get(cls) -> ToolConfig:
        if cls._instance is None:
            cls._instance = ToolConfig()
            cls._instance.cache_paths.ensure_all()
            _wire_cache_envs(cls._instance.cache_paths.base())
        return cls._instance

    @classmethod
    def set(cls, cfg: ToolConfig) -> None:
        cls._instance = cfg
        cls._instance.cache_paths.ensure_all()


def get_config() -> ToolConfig:
    """Return the global ToolConfig, creating it on first use.

    Ensures cache directories exist.
    """
    return _ConfigStore.get()


def set_config(cfg: ToolConfig) -> None:
    """Replace the global ToolConfig with a custom instance.

    Ensures cache directories exist.
    """
    _ConfigStore.set(cfg)
