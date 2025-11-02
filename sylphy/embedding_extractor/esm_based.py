# sylphy/embedding_extraction/esm_based.py
from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .embedding_based import EmbeddingBased


class ESMBasedEmbedding(EmbeddingBased):
    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "facebook/esm2_t6_8M_UR50D",
        name_tokenizer: str = "facebook/esm2_t6_8M_UR50D",
        dataset: object | None = None,
        column_seq: str | None = "sequence",
        debug: bool = False,
        debug_mode: int = logging.INFO,
        precision: str = "fp32",
        oom_backoff: bool = True,
    ) -> None:
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
            name_logging=ESMBasedEmbedding.__name__,
            trust_remote_code=False,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    def load_model_tokenizer(self) -> None:
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)

            self.__logger__.info("Loading ESM tokenizer from: %s", local_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir, do_lower_case=False, use_fast=True, trust_remote_code=False
            )
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.__logger__.debug("pad_token_id set to: %s", self.tokenizer.pad_token_id)

            self.__logger__.info("Loading ESM model from: %s on device=%s", local_dir, self.device)
            self.model = AutoModel.from_pretrained(local_dir, trust_remote_code=False).to(self.device)
            self.model.eval()
        except Exception as e:
            self.status = False
            self.message = f"Failed to load ESM tokenizer/model: {e}"
            self.__logger__.error(self.message)
            raise

    def _pre_tokenize(self, batch: list[str]) -> list[str]:
        vocab = getattr(self.tokenizer, "get_vocab", lambda: {})()
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
        """
        Return (hidden_states, attention_mask) with shapes:
          - hidden_states: tuple of length n_layers, each (B, L, H)
          - attention_mask: (B, L)
        """
        if not batch:
            raise ValueError("Input batch is empty.")
        self.ensure_loaded()
        return self._forward_hidden_states(batch, max_length=max_length)
