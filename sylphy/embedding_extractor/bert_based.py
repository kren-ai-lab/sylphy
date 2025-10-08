# sylphy/embedding_extraction/bert_based.py
from __future__ import annotations

import logging
from typing import Optional, List, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .embedding_based import EmbeddingBased


class BertBasedEmbedding(EmbeddingBased):
    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "Rostlab/prot_bert",
        name_tokenizer: str = "Rostlab/prot_bert",
        dataset: Optional[object] = None,
        column_seq: Optional[str] = "sequence",
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
            name_logging=BertBasedEmbedding.__name__,
            trust_remote_code=False,
            precision=precision,
            oom_backoff=oom_backoff,
        )

    def load_model_tokenizer(self) -> None:
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)

            self.__logger__.info("Loading ProtBERT tokenizer from: %s", local_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir, do_lower_case=False, use_fast=True, trust_remote_code=False
            )
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.__logger__.debug("pad_token_id set to: %s", self.tokenizer.pad_token_id)

            self.__logger__.info("Loading ProtBERT model from: %s on device=%s", local_dir, self.device)
            self.model = AutoModel.from_pretrained(local_dir, trust_remote_code=False).to(self.device)
            self.model.eval()
        except Exception as e:
            self.status = False
            self.message = f"Failed to load ProtBERT tokenizer/model: {e}"
            self.__logger__.error(self.message)
            raise

    def _pre_tokenize(self, sequences: List[str]) -> List[str]:
        return [" ".join((seq or "").strip()) for seq in sequences]

    @torch.no_grad()
    def embedding_batch(
        self,
        batch: List[str],
        max_length: int = 1024,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        if not batch:
            raise ValueError("Input batch is empty.")
        self.load_model_tokenizer()
        return self._forward_hidden_states(batch, max_length=max_length)
