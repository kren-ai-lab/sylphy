# sylphy/embedding_extraction/embedding_factory.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sylphy.logging import add_context, get_logger

from .ankh2_based import Ankh2BasedEmbedding
from .bert_based import BertBasedEmbedding
from .esm_based import ESMBasedEmbedding
from .esmc_based import ESMCBasedEmbedding
from .mistral_based import MistralBasedEmbedding
from .prot5_based import Prot5Based

if TYPE_CHECKING:
    from .embedding_based import EmbeddingBased  # for type hints only


def _norm(s: str) -> str:
    return (s or "").strip().lower()


# logging: ensure parent once, then a child for the factory
_ = get_logger("sylphy")
logger = logging.getLogger("sylphy.embedding_extraction.factory")
add_context(logger, component="embedding_extraction", backend="factory")


def EmbeddingFactory(  # noqa: N802
    model_name: str,
    dataset,
    column_seq: str,
    name_device: str = "cuda",
    precision: str = "fp32",
    oom_backoff: bool = True,
    debug: bool = False,
    debug_mode: int = logging.INFO,
) -> "EmbeddingBased":
    """
    Instantiate an embedding backend based on `model_name`.

    Returns
    -------
    EmbeddingBased
        Concrete backend (ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, or ESM-C).
    """
    name = _norm(model_name)

    if "esm2" in name or name.startswith("facebook/esm2"):
        logger.info("Selecting ESM2 backend", extra={"model": model_name})
        return ESMBasedEmbedding(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug=debug,
            debug_mode=debug_mode,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    if "ankh2" in name or name.startswith("elnaggarlab/ankh2"):
        logger.info("Selecting Ankh2 backend", extra={"model": model_name})
        return Ankh2BasedEmbedding(
            dataset=dataset,
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            column_seq=column_seq,
            use_encoder_only=True,
            debug=debug,
            debug_mode=debug_mode,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    if "t5" in name or "prot_t5" in name or name.startswith("rostlab/prot_t5"):
        logger.info("Selecting ProtT5 backend", extra={"model": model_name})
        return Prot5Based(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug=debug,
            debug_mode=debug_mode,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    if "bert" in name or "prot_bert" in name or name.startswith("rostlab/prot_bert"):
        logger.info("Selecting ProtBERT backend", extra={"model": model_name})
        return BertBasedEmbedding(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug=debug,
            debug_mode=debug_mode,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    if "mistral" in name or "mistral-prot" in name:
        logger.info("Selecting Mistral-Prot backend", extra={"model": model_name})
        return MistralBasedEmbedding(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug=debug,
            debug_mode=debug_mode,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    if "esmc" in name:
        logger.info("Selecting ESM-C backend", extra={"model": model_name})
        return ESMCBasedEmbedding(
            name_device=name_device,
            name_model=model_name,  # e.g., "esmc_300m" or registry key
            dataset=dataset,
            column_seq=column_seq,
            debug=debug,
            debug_mode=debug_mode,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    raise ValueError(
        f"Unknown model name '{model_name}'. "
        f"Supported: ESM2 ('facebook/esm2_*'), Ankh2 ('ElnaggarLab/ankh2-*'), "
        f"ProtT5 ('Rostlab/prot_t5_*'), ProtBERT ('Rostlab/prot_bert'), "
        f"Mistral-Prot ('RaphaelMourad/Mistral-Prot-*'), and ESM-C ('esmc_*')."
    )
