"""
embedding_factory.py

Factory to instantiate the appropriate embedding class based on model name.
"""

from typing import Optional, List, Callable, Type
import logging

from .esm_based import ESMBasedEmbedding
from .ankh2_based import Ankh2BasedEmbedding
from .prot5_based import Prot5Based
from .bert_based import BertBasedEmbedding
from .mistral_based import MistralBasedEmbedding
from .esmc_based import ESMCBasedEmbedding


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def EmbeddingFactory(
    model_name: str,
    dataset,
    column_seq: str,
    name_device: str = "cuda",
    precision: str = "fp32",
    oom_backoff: bool = True,
    debug: bool = True,
    debug_mode: int = logging.INFO,
):
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
    object
        Instance of a subclass of EmbeddingBased.

    Raises
    ------
    ValueError
        If no matching backend is found.
    """

    name = _norm(model_name)

    # Order matters (more specific first)
    if "esm2" in name or name.startswith("facebook/esm2"):
        # ESM (HF)
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
        # Ankh2 (HF, trust_remote_code=True in class)
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
        # ProtT5 (HF)
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
        # ProtBERT (HF)
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
        # Mistral-Prot (HF)
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
        # ESM-C (Meta SDK; optional registry via provider="other")
        return ESMCBasedEmbedding(
            name_device=name_device,
            name_model=model_name,   # e.g., "esmc_300m" or a registry key for local path/URL
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
