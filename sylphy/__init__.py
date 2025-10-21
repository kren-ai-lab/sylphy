# sylphy/__init__.py
from __future__ import annotations

from importlib import metadata as _metadata

"""
Sylphy public package surface.

Re-exports core registry/config symbols at the top-level so that users can:
    import sylphy as pr
    pr.ModelSpec
    pr.register_model(...)
    pr.resolve_model(...)
    pr.get_config()
"""

# Version
try:
    __version__ = _metadata.version("sylphy")
except _metadata.PackageNotFoundError:  # local dev / not installed
    __version__ = "0.0.1-dev"

# Public core API (re-export)
from .core import (  # noqa: E402
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
    "__version__",
    # core API
    "ModelSpec",
    "register_model",
    "register_alias",
    "unregister",
    "clear_registry",
    "list_registered_models",
    "get_model_spec",
    "resolve_model",
    "get_config",
    "set_cache_root",
    "temporary_cache_root",
]
