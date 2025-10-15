# core/__init__.py
from __future__ import annotations

from importlib import metadata as _metadata

from .config import get_config, set_cache_root, temporary_cache_root
from .model_registry import (
    ModelDownloadError,
    ModelNotFoundError,
    ModelRegistryError,
    clear_registry,
    get_model_spec,
    list_registered_models,
    register_alias,
    register_model,
    resolve_model,
    unregister,
)
from .model_spec import ModelSpec

__all__ = [
    # version
    "__version__",
    # config
    "get_config",
    "set_cache_root",
    "temporary_cache_root",
    # specs
    "ModelSpec",
    # registry API
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "register_model",
    "register_alias",
    "unregister",
    "clear_registry",
    "list_registered_models",
    "get_model_spec",
    "resolve_model",
]

try:  # prefer package metadata; fallback during local dev without installed dist
    __version__ = _metadata.version("sylphy")
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0-dev"
