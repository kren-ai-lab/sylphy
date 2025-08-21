# protein_representation/__init__.py
"""
protein_representation
======================

A toolkit to turn protein sequences into numerical representations:
- Classical **sequence encoders** (one-hot, ordinal, k-mers, physicochemical, FFT)
- **Embedding extraction** from pretrained PLMs (ESM2, ProtT5, ProtBERT, Mistral-Prot, Ankh2, ESM-C)
- **Reductions** (linear and non-linear) for downstream analysis
- A small, consistent **logging** facade and a **model registry/cache** layer

Design goals
------------
- Thin, well-typed public API at the top level
- Lazy loading to keep import time low and avoid optional-deps failures
- No side-effects on import (logging configured only when you call it)
"""

from __future__ import annotations

from importlib import import_module, metadata as importlib_metadata
from typing import Any, Dict, Tuple, TYPE_CHECKING

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def _resolve_version() -> str:
    """
    Return installed package version, falling back to a dev tag when running from source.
    """
    for dist_name in ("protein-representation", "protein_representation"):
        try:
            return importlib_metadata.version(dist_name)
        except Exception:
            pass
    return "0.0.dev0"

__version__ = _resolve_version()

# ---------------------------------------------------------------------------
# Public API (lazy exports)
# ---------------------------------------------------------------------------

__all__ = [
    "__version__",

    # logging
    "setup_logger", "get_logger", "add_context", "reset_logging",
    "silence_external", "set_global_level",

    # core: config
    "get_config", "set_cache_root", "temporary_cache_root",

    # core: registry/spec
    "ModelSpec", "register_model", "register_alias", "unregister", "clear_registry",
    "list_registered_models", "get_model_spec", "resolve_model",
    "ModelRegistryError", "ModelNotFoundError", "ModelDownloadError",

    # sequence encoders
    "Encoders", "OrdinalEncoder", "OneHotEncoder", "FrequencyEncoder",
    "KMersEncoders", "PhysicochemicalEncoder", "FFTEncoder", "create_encoder",

    # embeddings
    "EmbeddingBased", "ESMBasedEmbedding", "Prot5Based", "BertBasedEmbedding",
    "MistralBasedEmbedding", "ESMCBasedEmbedding", "Ankh2BasedEmbedding",
    "EmbeddingFactory", "create_embedding", "SUPPORTED_FAMILIES",

    # reductions
    "Reductions", "LinearReduction", "NonLinearReductions",
    "reduce_dimensionality", "get_available_methods",
    "is_linear_method", "is_nonlinear_method",
]

# Map exported names -> (module, attribute). Modules are relative to this package.
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # logging
    "setup_logger":         (".logging", "setup_logger"),
    "get_logger":           (".logging", "get_logger"),
    "add_context":          (".logging", "add_context"),
    "reset_logging":        (".logging", "reset_logging"),
    "silence_external":     (".logging", "silence_external"),
    "set_global_level":     (".logging", "set_global_level"),

    # core: config
    "get_config":           (".core.config", "get_config"),
    "set_cache_root":       (".core.config", "set_cache_root"),
    "temporary_cache_root": (".core.config", "temporary_cache_root"),

    # core: registry/spec
    "ModelSpec":            (".core.model_spec", "ModelSpec"),
    "register_model":       (".core.model_registry", "register_model"),
    "register_alias":       (".core.model_registry", "register_alias"),
    "unregister":           (".core.model_registry", "unregister"),
    "clear_registry":       (".core.model_registry", "clear_registry"),
    "list_registered_models": (".core.model_registry", "list_registered_models"),
    "get_model_spec":       (".core.model_registry", "get_model_spec"),
    "resolve_model":        (".core.model_registry", "resolve_model"),
    "ModelRegistryError":   (".core.model_registry", "ModelRegistryError"),
    "ModelNotFoundError":   (".core.model_registry", "ModelNotFoundError"),
    "ModelDownloadError":   (".core.model_registry", "ModelDownloadError"),

    # sequence encoders (proxied through subpackage __init__)
    "Encoders":             (".sequence_encoder", "Encoders"),
    "OrdinalEncoder":       (".sequence_encoder", "OrdinalEncoder"),
    "OneHotEncoder":        (".sequence_encoder", "OneHotEncoder"),
    "FrequencyEncoder":     (".sequence_encoder", "FrequencyEncoder"),
    "KMersEncoders":        (".sequence_encoder", "KMersEncoders"),
    "PhysicochemicalEncoder": (".sequence_encoder", "PhysicochemicalEncoder"),
    "FFTEncoder":           (".sequence_encoder", "FFTEncoder"),
    "create_encoder":       (".sequence_encoder", "create_encoder"),

    # embeddings (proxied through subpackage __init__ which is lazy itself)
    "EmbeddingBased":       (".embedding_extraction", "EmbeddingBased"),
    "ESMBasedEmbedding":    (".embedding_extraction", "ESMBasedEmbedding"),
    "Prot5Based":           (".embedding_extraction", "Prot5Based"),
    "BertBasedEmbedding":   (".embedding_extraction", "BertBasedEmbedding"),
    "MistralBasedEmbedding": (".embedding_extraction", "MistralBasedEmbedding"),
    "ESMCBasedEmbedding":   (".embedding_extraction", "ESMCBasedEmbedding"),
    "Ankh2BasedEmbedding":  (".embedding_extraction", "Ankh2BasedEmbedding"),
    "EmbeddingFactory":     (".embedding_extraction", "EmbeddingFactory"),
    # create_embedding is handled specially below
    "SUPPORTED_FAMILIES":   (".embedding_extraction", "SUPPORTED_FAMILIES"),

    # reductions (proxied through subpackage __init__)
    "Reductions":           (".reductions", "Reductions"),
    "LinearReduction":      (".reductions", "LinearReduction"),
    "NonLinearReductions":  (".reductions", "NonLinearReductions"),
    "reduce_dimensionality": (".reductions", "reduce_dimensionality"),
    "get_available_methods": (".reductions", "get_available_methods"),
    "is_linear_method":     (".reductions", "is_linear_method"),
    "is_nonlinear_method":  (".reductions", "is_nonlinear_method"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import attributes from submodules on first access.
    Keeps the root import light and avoids importing optional heavy deps.
    """
    # Special case: alias without double-indirection
    if name == "create_embedding":
        module = import_module(".embedding_extraction", package=__name__)
        value = getattr(module, "EmbeddingFactory")
        globals()[name] = value
        return value

    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    mod_name, attr = spec
    module = import_module(mod_name, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value  # cache for subsequent access
    return value


def __dir__() -> list[str]:
    return sorted(list(__all__))
    

# Typing-only imports so static analyzers see the symbols, while runtime stays lazy.
if TYPE_CHECKING:  # pragma: no cover
    from .logging import setup_logger, get_logger, add_context, reset_logging, silence_external, set_global_level
    from .core.config import get_config, set_cache_root, temporary_cache_root
    from .core.model_spec import ModelSpec
    from .core.model_registry import (
        register_model, register_alias, unregister, clear_registry, list_registered_models,
        get_model_spec, resolve_model, ModelRegistryError, ModelNotFoundError, ModelDownloadError,
    )
    from .sequence_encoder import (
        Encoders, OrdinalEncoder, OneHotEncoder, FrequencyEncoder,
        KMersEncoders, PhysicochemicalEncoder, FFTEncoder, create_encoder,
    )
    from .embedding_extractor import (
        EmbeddingBased, ESMBasedEmbedding, Prot5Based, BertBasedEmbedding,
        MistralBasedEmbedding, ESMCBasedEmbedding, Ankh2BasedEmbedding,
        EmbeddingFactory, create_embedding, SUPPORTED_FAMILIES,
    )
    from .reductions import (
        Reductions, LinearReduction, NonLinearReductions,
        reduce_dimensionality, get_available_methods, is_linear_method, is_nonlinear_method,
    )
