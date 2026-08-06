"""Select and instantiate embedding backends from model names."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING, Any

from sylphy.core.model_registry import normalize_name
from sylphy.core.optional_dependencies import wrap_optional_dependency_error
from sylphy.logging import get_child_logger

if TYPE_CHECKING:
    import polars as pl

    from sylphy.types import PrecisionType

    from .embedding_base import EmbeddingBase  # for type hints only


logger = get_child_logger(
    "embedding_extraction.factory",
    component="embedding_extraction",
    facility="factory",
)

_BACKENDS: list[tuple[tuple[str, ...], str, str, str, dict[str, Any]]] = [
    (
        ("esm2", "facebook/esm2"),
        ".esm_embedding",
        "ESMEmbedding",
        "ESM2",
        {},
    ),
    (
        ("ankh2", "elnaggarlab/ankh2", "ankh3", "elnaggarlab/ankh3"),
        ".ankh2_embedding",
        "Ankh2Embedding",
        "Ankh2/3",
        {"use_encoder_only": True},
    ),
    (
        ("t5", "prot_t5", "rostlab/prot_t5"),
        ".prot_t5_embedding",
        "ProtT5Embedding",
        "ProtT5",
        {},
    ),
    (
        ("bert", "prot_bert", "rostlab/prot_bert"),
        ".prot_bert_embedding",
        "ProtBertEmbedding",
        "ProtBERT",
        {},
    ),
    (
        ("mistral", "mistral-prot"),
        ".mistral_embedding",
        "MistralEmbedding",
        "Mistral-Prot",
        {},
    ),
]


def create_embedding(
    model_name: str,
    dataset: pl.DataFrame,
    column_seq: str,
    name_device: str = "cuda",
    precision: PrecisionType = "fp32",
    debug_mode: int = logging.INFO,
    *,
    oom_backoff: bool = True,
    debug: bool = False,
) -> EmbeddingBase:
    """Instantiate an embedding backend from a model name.

    Returns:
        Concrete backend instance.

    """
    name = normalize_name(model_name)

    common_kwargs: dict[str, Any] = {
        "name_device": name_device,
        "name_model": model_name,
        "name_tokenizer": model_name,
        "dataset": dataset,
        "column_seq": column_seq,
        "debug_mode": debug_mode,
        "precision": precision,
        "debug": debug,
        "oom_backoff": oom_backoff,
    }

    for patterns, module_suffix, class_name, label, extra in _BACKENDS:
        if any(p in name for p in patterns):
            logger.info("Selecting %s backend", label, extra={"model": model_name})
            module = import_module(module_suffix, package=__package__)
            cls = getattr(module, class_name)
            return cls(**common_kwargs, **extra)  # type: ignore[no-any-return]

    if "esmc" in name:
        logger.info("Selecting ESM-C backend", extra={"model": model_name})
        try:
            module = import_module(".esmc_embedding", package=__package__)
            cls = module.ESMCEmbedding
        except (ImportError, ModuleNotFoundError) as exc:
            wrapped = wrap_optional_dependency_error(
                exc,
                feature="ESM-C embeddings",
                extra="embeddings",
                packages=("esm", "torch"),
            )
            if wrapped is not None:
                raise wrapped from exc
            raise
        esmc_kwargs = {k: v for k, v in common_kwargs.items() if k != "name_tokenizer"}
        return cls(**esmc_kwargs)  # type: ignore[no-any-return]

    msg = (
        f"Unknown model name '{model_name}'. "
        f"Supported: ESM2 ('facebook/esm2_*'), Ankh2/3 ('ElnaggarLab/ankh2-*', 'ElnaggarLab/ankh3-*'), "
        f"ProtT5 ('Rostlab/prot_t5_*'), ProtBERT ('Rostlab/prot_bert'), "
        f"Mistral-Prot ('RaphaelMourad/Mistral-Prot-*'), ESM-C ('esmc_*')."
    )
    raise ValueError(msg)
