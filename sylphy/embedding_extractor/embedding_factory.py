# protein_representation/embedding_extraction/embedding_factory.py
"""
embedding_factory.py

Factory to instantiate the appropriate embedding class based on model name.
Uses unified package logging and emits informative selection messages.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sylphy.constants.tool_configs import ToolConfig
from sylphy.logging import get_logger, add_context

from .esm_based import ESMBasedEmbedding
from .ankh2_based import Ankh2BasedEmbedding
from .prot5_based import Prot5Based
from .bert_based import BertBasedEmbedding
from .mistral_based import MistralBasedEmbedding
from .esmc_based import ESMCBasedEmbedding

if TYPE_CHECKING:
    from .embedding_based import EmbeddingBased  # for type hints only


def _norm(s: str) -> str:
    return (s or "").strip().lower()


# --- logging: ensure parent once, use a child logger for the factory ----
_ = get_logger("protein_representation")  # idempotent, config done once
logger = logging.getLogger("protein_representation.embedding_extraction.factory")
add_context(logger, component="embedding_extraction", backend="factory")


def EmbeddingFactory(  # noqa: N802 (factory name kept for public API)
    model_name: str,
    dataset,
    column_seq: str,
    name_device: str = "cuda",
    precision: str = "fp32",
    oom_backoff: bool = True,
    debug: bool = True,
    debug_mode: int = ToolConfig.log_level,
) -> "EmbeddingBased":
    """
    Factory function to return the appropriate embedding class instance.

    Parameters
    ----------
    model_name : str
        Model key or HF ref (e.g., "facebook/esm2_t6_8M_UR50D", "Rostlab/prot_bert", "esmc_300m").
    dataset : pd.DataFrame
        Input dataset containing sequences.
    column_seq : str
        Column name with sequences.
    name_device : str
        "cuda" or "cpu".
    precision : {"fp32","fp16","bf16"}
        AMP precision for forward passes (if CUDA & model supports it).
    oom_backoff : bool
        If True, reduce batch size automatically on CUDA OOM.
    debug : bool
        Enable logger for the embedding class.
    debug_mode : logging level
        Logging level (e.g., logging.INFO).

    Returns
    -------
    EmbeddingBased
        Instance of a subclass of EmbeddingBased.

    Raises
    ------
    ValueError
        If no matching backend is found.
    """
    name = _norm(model_name)

    # Order matters (more specific first)
    if "esm2" in name or name.startswith("facebook/esm2"):
        logger.info("Selecting ESM2 backend", extra={"model": model_name, "family": "esm2"})
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
        logger.info("Selecting Ankh2 backend", extra={"model": model_name, "family": "ankh2"})
        return Ankh2BasedEmbedding(
            dataset=dataset,
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            column_seq=column_seq,
            use_encoder_only=True,
            debug=debug,
            debug_mode=debug_mode,
        )

    if "t5" in name or "prot_t5" in name or name.startswith("rostlab/prot_t5"):
        logger.info("Selecting ProtT5 backend", extra={"model": model_name, "family": "prot_t5"})
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
        logger.info("Selecting ProtBERT backend", extra={"model": model_name, "family": "prot_bert"})
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
        logger.info("Selecting Mistral-Prot backend", extra={"model": model_name, "family": "mistral_prot"})
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
        logger.info("Selecting ESM-C backend", extra={"model": model_name, "family": "esmc"})
        return ESMCBasedEmbedding(
            name_device=name_device,
            name_model=model_name,  # e.g., "esmc_300m" or a registry key for local path/URL
            dataset=dataset,
            column_seq=column_seq,
            debug=debug,
            debug_mode=debug_mode,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    # If we reach here, we didn't recognize the backend
    raise ValueError(
        f"Unknown model name '{model_name}'. "
        f"Supported families include: ESM2 ('facebook/esm2_*'), Ankh2 ('ElnaggarLab/ankh2-*'), "
        f"ProtT5 ('Rostlab/prot_t5_*'), ProtBERT ('Rostlab/prot_bert'), "
        f"Mistral-Prot ('RaphaelMourad/Mistral-Prot-*'), and ESM-C ('esmc_*')."
    )
