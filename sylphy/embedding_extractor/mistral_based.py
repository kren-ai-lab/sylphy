"""Implement the Mistral-Prot embedding backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .embedding_based import EmbeddingBase

if TYPE_CHECKING:
    import pandas as pd

    from sylphy.types import PrecisionType


class MistralEmbedding(EmbeddingBase):
    """Extract embeddings using Mistral-Prot models."""

    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "RaphaelMourad/Mistral-Prot-v1-134M",
        name_tokenizer: str = "RaphaelMourad/Mistral-Prot-v1-134M",
        dataset: pd.DataFrame | None = None,
        column_seq: str | None = "sequence",
        debug_mode: int = logging.INFO,
        precision: PrecisionType = "fp32",
        *,
        debug: bool = False,
        oom_backoff: bool = True,
    ) -> None:
        """Initialize the Mistral-Prot backend."""
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
        """Load Mistral tokenizer and model from the resolved model directory."""
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)

            self.__logger__.info("Loading Mistral tokenizer from: %s", local_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                local_dir, do_lower_case=False, use_fast=True, trust_remote_code=False,
            )
            if getattr(tokenizer, "pad_token_id", None) is None:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.__logger__.debug("pad_token_id set to: %s", tokenizer.pad_token_id)
            self.tokenizer = tokenizer

            self.__logger__.info("Loading Mistral model from: %s on device=%s", local_dir, self.device)
            model = AutoModel.from_pretrained(local_dir, trust_remote_code=False)  # type: ignore[possibly-missing-attribute]
            model.to(self.device)
            self.model = model
            model.eval()
        except Exception as e:
            self.__logger__.error("Failed to load Mistral tokenizer/model: %s", e)
            raise

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
        return self._forward_hidden_states(batch, max_length=max_length)
