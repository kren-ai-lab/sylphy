# sylphy/__init__.py
from __future__ import annotations

"""
Sylphy public package surface.

Re-exports core registry/config symbols at the top-level so that users can:
    import sylphy as pr
    pr.ModelSpec
    pr.register_model(...)
    pr.resolve_model(...)
    pr.get_config()
"""

__version__ = "0.1.3"

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
