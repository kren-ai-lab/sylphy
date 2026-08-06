"""Implement the ProtT5 embedding backend."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn
from transformers import AutoConfig, T5EncoderModel, T5Tokenizer

from sylphy.core.optional_dependencies import wrap_optional_dependency_error

from .embedding_base import DEFAULT_DEBUG_MODE, DEFAULT_DEVICE, DEFAULT_PRECISION, EmbeddingBase

if TYPE_CHECKING:
    import polars as pl

    from sylphy.types import PrecisionType


class ProtT5Embedding(EmbeddingBase):
    """Extract embeddings using ProtT5 encoder models."""

    def __init__(
        self,
        name_device: str = DEFAULT_DEVICE,
        name_model: str = "Rostlab/prot_t5_xl_uniref50",
        name_tokenizer: str = "Rostlab/prot_t5_xl_uniref50",
        dataset: pl.DataFrame | None = None,
        column_seq: str | None = "sequence",
        debug_mode: int = DEFAULT_DEBUG_MODE,
        precision: PrecisionType = DEFAULT_PRECISION,
        *,
        debug: bool = False,
        oom_backoff: bool = True,
    ) -> None:
        """Initialize the ProtT5 backend."""
        if dataset is None:
            msg = "dataset must be provided"
            raise ValueError(msg)

        super().__init__(
            dataset=dataset,
            name_device=name_device,
            name_model=name_model,
            name_tokenizer=name_tokenizer,
            provider="huggingface",
            revision=None,
            column_seq=column_seq or "sequence",
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            trust_remote_code=False,
            oom_backoff=oom_backoff,
        )

    def load_model_tokenizer(self) -> None:
        """Load ProtT5 tokenizer and encoder model."""
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)

            self.__logger__.info("Loading ProtT5 tokenizer from: %s", local_dir)
            tokenizer = T5Tokenizer.from_pretrained(local_dir, do_lower_case=False, use_fast=False)  # type: ignore[possibly-missing-attribute]
            tokenizer_any = cast("Any", tokenizer)
            pad_token_id: int | None = getattr(tokenizer_any, "pad_token_id", None)
            if pad_token_id is None:
                tokenizer_any.add_special_tokens({"pad_token": "<pad>"})
                pad_token_id = getattr(tokenizer_any, "pad_token_id", None)
            if pad_token_id is not None:
                self.__logger__.debug("pad_token_id set to: %s", pad_token_id)
            self.tokenizer = tokenizer_any

            self.__logger__.info("Loading ProtT5 encoder from: %s on device=%s", local_dir, self.device)
            model = T5EncoderModel.from_pretrained(local_dir)  # type: ignore[possibly-missing-attribute]
            cast("nn.Module", model).to(self.device)
            self.model = model
            model.eval()
        except (ImportError, ModuleNotFoundError) as e:
            wrapped = wrap_optional_dependency_error(
                e,
                feature="ProtT5 embeddings",
                extra="embeddings",
                packages=("sentencepiece",),
            )
            if wrapped is not None:
                self.__logger__.error("%s", wrapped)
                raise wrapped from e
            raise
        except Exception as e:
            self.__logger__.error("Failed to load ProtT5 tokenizer/model: %s", e)
            raise

    def _pre_tokenize(self, batch: list[str]) -> list[str]:
        # Replace uncommon amino acids and space-separate
        formatted = []
        for s in batch:
            seq = (s or "").strip()
            clean = re.sub(r"[UZOB]", "X", seq)
            formatted.append(" ".join(clean))
        return formatted

    @torch.no_grad()
    def embedding_batch(
        self,
        batch: list[str],
        max_length: int = 1024,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Embed a batch and return hidden states with an attention mask."""
        if not batch:
            msg = "Input batch is empty."
            raise ValueError(msg)
        self.ensure_loaded()
        if self.tokenizer is None:
            msg = "Tokenizer not loaded."
            raise RuntimeError(msg)
        return self._forward_hidden_states(batch, max_length=max_length)
