"""Implement the ESM2 embedding backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .embedding_base import DEFAULT_DEBUG_MODE, DEFAULT_DEVICE, DEFAULT_PRECISION, EmbeddingBase

if TYPE_CHECKING:
    import polars as pl

    from sylphy.types import PrecisionType


class ESMEmbedding(EmbeddingBase):
    """Extract embeddings using ESM2 Hugging Face models."""

    def __init__(
        self,
        name_device: str = DEFAULT_DEVICE,
        name_model: str = "facebook/esm2_t6_8M_UR50D",
        name_tokenizer: str = "facebook/esm2_t6_8M_UR50D",
        dataset: pl.DataFrame | None = None,
        column_seq: str | None = "sequence",
        debug_mode: int = DEFAULT_DEBUG_MODE,
        precision: PrecisionType = DEFAULT_PRECISION,
        *,
        debug: bool = False,
        oom_backoff: bool = True,
    ) -> None:
        """Initialize the ESM backend."""
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
        """Load ESM tokenizer and model from the resolved local directory."""
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)

            self.__logger__.info("Loading ESM tokenizer from: %s", local_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                do_lower_case=False,
                use_fast=True,
                trust_remote_code=False,
            )
            if getattr(tokenizer, "pad_token_id", None) is None:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.__logger__.debug("pad_token_id set to: %s", tokenizer.pad_token_id)
            self.tokenizer = tokenizer

            self.__logger__.info("Loading ESM model from: %s on device=%s", local_dir, self.device)
            model = AutoModel.from_pretrained(local_dir, trust_remote_code=False)  # type: ignore[possibly-missing-attribute]
            model.to(self.device)
            self.model = model
            model.eval()
        except Exception as e:
            self.__logger__.error("Failed to load ESM tokenizer/model: %s", e)
            raise

    def _pre_tokenize(self, batch: list[str]) -> list[str]:
        vocab = getattr(self.tokenizer, "get_vocab", dict)()
        has_cls = "<cls>" in vocab
        if has_cls:
            return [f"<cls> {' '.join(seq.strip())}" for seq in batch]
        return [seq.strip() for seq in batch]

    @torch.no_grad()
    def embedding_batch(
        self,
        batch: list[str],
        max_length: int = 1024,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Return hidden states and attention mask for a sequence batch.

        Returns:
            A tuple of hidden states by layer and the attention mask.

        """
        if not batch:
            msg = "Input batch is empty."
            raise ValueError(msg)
        self.ensure_loaded()
        return self._forward_hidden_states(batch, max_length=max_length)
