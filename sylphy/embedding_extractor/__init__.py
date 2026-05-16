"""Expose embedding backends with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from sylphy.core.optional_dependencies import wrap_optional_dependency_error

__all__ = [
    "Ankh2Embedding",
    "ESMCEmbedding",
    "ESMEmbedding",
    "EmbeddingBase",
    "MistralEmbedding",
    "ProtBertEmbedding",
    "ProtT5Embedding",
    "create_embedding",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EmbeddingBase": (".embedding_base", "EmbeddingBase"),
    "ESMEmbedding": (".esm_embedding", "ESMEmbedding"),
    "ProtT5Embedding": (".prot_t5_embedding", "ProtT5Embedding"),
    "ProtBertEmbedding": (".prot_bert_embedding", "ProtBertEmbedding"),
    "MistralEmbedding": (".mistral_embedding", "MistralEmbedding"),
    "ESMCEmbedding": (".esmc_embedding", "ESMCEmbedding"),
    "Ankh2Embedding": (".ankh2_embedding", "Ankh2Embedding"),
    "create_embedding": (".embedding_factory", "create_embedding"),
}


def __getattr__(name: str) -> object:
    """Resolve lazy exports and cache the loaded symbol."""
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        msg = f"module '{__name__}' has no attribute '{name}'"
        raise AttributeError(msg)
    mod_name, attr = spec
    try:
        module = import_module(mod_name, package=__name__)
    except (ImportError, ModuleNotFoundError) as exc:
        wrapped = wrap_optional_dependency_error(
            exc,
            feature="Embedding features",
            extra="embeddings",
            packages=("torch", "transformers", "sentencepiece", "esm"),
        )
        if wrapped is not None:
            raise wrapped from exc
        raise
    value = getattr(module, attr)
    globals()[name] = value  # cache
    return value


def __dir__() -> list[str]:
    """Return available public exports for this package."""
    return sorted(__all__)


# Optional typing-only exposure (keeps runtime lazy)
if TYPE_CHECKING:  # pragma: no cover
    from .ankh2_embedding import Ankh2Embedding
    from .embedding_base import EmbeddingBase
    from .embedding_factory import create_embedding
    from .esm_embedding import ESMEmbedding
    from .esmc_embedding import ESMCEmbedding
    from .mistral_embedding import MistralEmbedding
    from .prot_bert_embedding import ProtBertEmbedding
    from .prot_t5_embedding import ProtT5Embedding
