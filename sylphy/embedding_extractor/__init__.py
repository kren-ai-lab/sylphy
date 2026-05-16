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
    "EmbeddingFactory",
    "MistralEmbedding",
    "ProtBertEmbedding",
    "ProtT5Embedding",
    "create_embedding",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EmbeddingBase": (".embedding_based", "EmbeddingBase"),
    "ESMEmbedding": (".esm_based", "ESMEmbedding"),
    "ProtT5Embedding": (".prot5_based", "ProtT5Embedding"),
    "ProtBertEmbedding": (".bert_based", "ProtBertEmbedding"),
    "MistralEmbedding": (".mistral_based", "MistralEmbedding"),
    "ESMCEmbedding": (".esmc_based", "ESMCEmbedding"),
    "Ankh2Embedding": (".ankh2_based", "Ankh2Embedding"),
    "EmbeddingFactory": (".embedding_factory", "EmbeddingFactory"),
}

def __getattr__(name: str) -> object:
    """Resolve lazy exports and cache the loaded symbol."""
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        if name == "create_embedding":
            module = import_module(".embedding_factory", package=__name__)
            return module.EmbeddingFactory
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
    from .ankh2_based import Ankh2Embedding
    from .bert_based import ProtBertEmbedding
    from .embedding_based import EmbeddingBase
    from .embedding_factory import EmbeddingFactory
    from .esm_based import ESMEmbedding
    from .esmc_based import ESMCEmbedding
    from .mistral_based import MistralEmbedding
    from .prot5_based import ProtT5Embedding

    create_embedding = EmbeddingFactory
