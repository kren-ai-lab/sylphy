# protein_representation/embedding_extraction/__init__.py
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

"""
Embedding Extraction
====================

Unified interface to extract embeddings from protein sequence models.

- Lazy-loaded backends to keep imports fast.
- The factory logs which backend was selected.

Public API
----------
Base:
    EmbeddingBased
Backends:
    ESMBasedEmbedding, Prot5Based, BertBasedEmbedding,
    MistralBasedEmbedding, ESMCBasedEmbedding, Ankh2BasedEmbedding
Factory:
    EmbeddingFactory, create_embedding
Meta:
    SUPPORTED_FAMILIES
"""

__all__ = [
    "EmbeddingBased",
    "ESMBasedEmbedding",
    "Prot5Based",
    "BertBasedEmbedding",
    "MistralBasedEmbedding",
    "ESMCBasedEmbedding",
    "Ankh2BasedEmbedding",
    "EmbeddingFactory",
    "create_embedding",
    "SUPPORTED_FAMILIES",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "EmbeddingBased": (".embedding_based", "EmbeddingBased"),
    "ESMBasedEmbedding": (".esm_based", "ESMBasedEmbedding"),
    "Prot5Based": (".prot5_based", "Prot5Based"),
    "BertBasedEmbedding": (".bert_based", "BertBasedEmbedding"),
    "MistralBasedEmbedding": (".mistral_based", "MistralBasedEmbedding"),
    "ESMCBasedEmbedding": (".esmc_based", "ESMCBasedEmbedding"),
    "Ankh2BasedEmbedding": (".ankh2_based", "Ankh2BasedEmbedding"),
    "EmbeddingFactory": (".embedding_factory", "EmbeddingFactory"),
}

SUPPORTED_FAMILIES = ("esm2", "ankh2", "prot_t5", "prot_bert", "mistral_prot", "esmc")


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        if name == "create_embedding":
            from .embedding_factory import EmbeddingFactory  # lazy import

            return EmbeddingFactory
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    mod_name, attr = spec
    module = import_module(mod_name, package=__name__)
    value = getattr(module, attr)
    globals()[name] = value  # cache
    return value


def __dir__():
    return sorted(list(__all__))


# Optional typing-only exposure (keeps runtime lazy)
if TYPE_CHECKING:  # pragma: no cover
    from .embedding_factory import EmbeddingFactory

    create_embedding = EmbeddingFactory
