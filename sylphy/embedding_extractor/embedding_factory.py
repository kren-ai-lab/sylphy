# sylphy/embedding_extraction/embedding_factory.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sylphy.core.optional_dependencies import wrap_optional_dependency_error
from sylphy.logging import add_context, get_logger

if TYPE_CHECKING:
    import pandas as pd

    from sylphy.types import PrecisionType

    from .embedding_based import EmbeddingBased  # for type hints only


def _norm(s: str) -> str:
    return (s or "").strip().lower()


# logging: ensure parent once, then a child for the factory
_ = get_logger("sylphy")
logger = logging.getLogger("sylphy.embedding_extraction.factory")
add_context(logger, component="embedding_extraction", backend="factory")


def EmbeddingFactory(
    model_name: str,
    dataset: pd.DataFrame,
    column_seq: str,
    name_device: str = "cuda",
    precision: PrecisionType = "fp32",
    debug_mode: int = logging.INFO,
    *,
    oom_backoff: bool = True,
    debug: bool = False,
) -> EmbeddingBased:
    """Instantiate an embedding backend based on `model_name`.

    Returns
    -------
    EmbeddingBased
        Concrete backend (ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, or ESM-C).

    """
    name = _norm(model_name)

    if "esm2" in name or name.startswith("facebook/esm2"):
        logger.info("Selecting ESM2 backend", extra={"model": model_name})
        from .esm_based import ESMBasedEmbedding  # noqa: PLC0415

        return ESMBasedEmbedding(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            oom_backoff=oom_backoff,
        )

    if "ankh2" in name or name.startswith("elnaggarlab/ankh2"):
        logger.info("Selecting Ankh2 backend", extra={"model": model_name})
        from .ankh2_based import Ankh2BasedEmbedding  # noqa: PLC0415

        return Ankh2BasedEmbedding(
            dataset=dataset,
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            use_encoder_only=True,
            debug=debug,
            oom_backoff=oom_backoff,
        )

    if "ankh3" in name or name.startswith("elnaggarlab/ankh3"):
        logger.info("Selecting Ankh2 backend", extra={"model": model_name})
        from .ankh2_based import Ankh2BasedEmbedding  # noqa: PLC0415

        return Ankh2BasedEmbedding(
            dataset=dataset,
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            use_encoder_only=True,
            debug=debug,
            oom_backoff=oom_backoff,
        )

    if "t5" in name or "prot_t5" in name or name.startswith("rostlab/prot_t5"):
        logger.info("Selecting ProtT5 backend", extra={"model": model_name})
        from .prot5_based import Prot5Based  # noqa: PLC0415

        return Prot5Based(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            oom_backoff=oom_backoff,
        )

    if "bert" in name or "prot_bert" in name or name.startswith("rostlab/prot_bert"):
        logger.info("Selecting ProtBERT backend", extra={"model": model_name})
        from .bert_based import BertBasedEmbedding  # noqa: PLC0415

        return BertBasedEmbedding(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            oom_backoff=oom_backoff,
        )

    if "mistral" in name or "mistral-prot" in name:
        logger.info("Selecting Mistral-Prot backend", extra={"model": model_name})
        from .mistral_based import MistralBasedEmbedding  # noqa: PLC0415

        return MistralBasedEmbedding(
            name_device=name_device,
            name_model=model_name,
            name_tokenizer=model_name,
            dataset=dataset,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            oom_backoff=oom_backoff,
        )

    if "esmc" in name:
        logger.info("Selecting ESM-C backend", extra={"model": model_name})
        try:
            from .esmc_based import ESMCBasedEmbedding  # noqa: PLC0415
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
        return ESMCBasedEmbedding(
            name_device=name_device,
            name_model=model_name,  # e.g., "esmc_300m" or registry key
            dataset=dataset,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            oom_backoff=oom_backoff,
        )

    msg = (
        f"Unknown model name '{model_name}'. "
        f"Supported: ESM2 ('facebook/esm2_*'), Ankh2 ('ElnaggarLab/ankh2-*'), "
        f"ProtT5 ('Rostlab/prot_t5_*'), ProtBERT ('Rostlab/prot_bert'), "
        f"Mistral-Prot ('RaphaelMourad/Mistral-Prot-*'), and ESM-C ('esmc_*')."
    )
    raise ValueError(
        msg,
    )
