"""Expose Sylphy's public top-level API."""

from __future__ import annotations

__version__ = "2.0.0"

# Public core API (re-export)
from .core import (
    ModelSpec,
    clear_registry,
    get_config,
    get_model_spec,
    list_registered_models,
    register_alias,
    register_model,
    resolve_model,
    set_cache_root,
    temporary_cache_root,
    unregister,
)

__all__ = [
    # core API
    "ModelSpec",
    "__version__",
    "clear_registry",
    "get_config",
    "get_model_spec",
    "list_registered_models",
    "register_alias",
    "register_model",
    "resolve_model",
    "set_cache_root",
    "temporary_cache_root",
    "unregister",
]
