# sylphy/embedding_extraction/prot5_based.py
from __future__ import annotations

import logging
import re
from typing import Any, cast

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoConfig, T5EncoderModel, T5Tokenizer

from sylphy.types import PrecisionType

from .embedding_based import EmbeddingBased


class Prot5Based(EmbeddingBased):
    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "Rostlab/prot_t5_xl_uniref50",
        name_tokenizer: str = "Rostlab/prot_t5_xl_uniref50",
        dataset: pd.DataFrame | None = None,
        column_seq: str | None = "sequence",
        debug: bool = False,
        debug_mode: int = logging.INFO,
        precision: PrecisionType = "fp32",
        oom_backoff: bool = True,
    ) -> None:
        if dataset is None:
            raise ValueError("dataset must be provided")

        super().__init__(
            dataset=dataset,
            name_device=name_device,
            name_model=name_model,
            name_tokenizer=name_tokenizer,
            provider="huggingface",
            revision=None,
            column_seq=column_seq or "sequence",
            debug=debug,
            debug_mode=debug_mode,
            name_logging=Prot5Based.__name__,
            trust_remote_code=False,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    def load_model_tokenizer(self) -> None:
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)

            self.__logger__.info("Loading ProtT5 tokenizer from: %s", local_dir)
            tokenizer = T5Tokenizer.from_pretrained(local_dir, do_lower_case=False, use_fast=False)  # type: ignore[possibly-missing-attribute]
            tokenizer_any = cast(Any, tokenizer)
            pad_token_id: int | None = getattr(tokenizer_any, "pad_token_id", None)
            if pad_token_id is None:
                tokenizer_any.add_special_tokens({"pad_token": "<pad>"})
                pad_token_id = getattr(tokenizer_any, "pad_token_id", None)
            if pad_token_id is not None:
                self.__logger__.debug("pad_token_id set to: %s", pad_token_id)
            self.tokenizer = tokenizer_any

            self.__logger__.info("Loading ProtT5 encoder from: %s on device=%s", local_dir, self.device)
            model = T5EncoderModel.from_pretrained(local_dir)  # type: ignore[possibly-missing-attribute]
            cast(nn.Module, model).to(self.device)
            self.model = model
            model.eval()
        except Exception as e:
            self.status = False
            self.message = f"Failed to load ProtT5 tokenizer/model: {e}"
            self.__logger__.error(self.message)
            raise

    def _pre_tokenize(self, batch: list[str]) -> list[str]:
        # Replace uncommon amino acids and space-separate
        formatted = []
        for seq in batch:
            seq = (seq or "").strip()
            clean = re.sub(r"[UZOB]", "X", seq)
            formatted.append(" ".join(clean))
        return formatted

    @torch.no_grad()
    def embedding_batch(
        self,
        batch: list[str],
        max_length: int = 1024,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if not batch:
            raise ValueError("Input batch is empty.")
        self.ensure_loaded()
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded.")
        return self._forward_hidden_states(batch, max_length=max_length)
