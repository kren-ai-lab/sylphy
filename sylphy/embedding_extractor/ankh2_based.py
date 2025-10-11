# sylphy/embedding_extraction/ankh2_based.py
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5EncoderModel

from .embedding_based import EmbeddingBased


class Ankh2BasedEmbedding(EmbeddingBased):
    def __init__(
        self,
        dataset,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "ElnaggarLab/ankh2-ext1",
        name_tokenizer: str = "ElnaggarLab/ankh2-ext1",
        column_seq: str = "sequence",
        use_encoder_only: bool = True,
        debug: bool = False,
        debug_mode: int = logging.INFO,
        *,
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
            column_seq=column_seq,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=Ankh2BasedEmbedding.__name__,
            trust_remote_code=True,
            precision=precision,
            oom_backoff=oom_backoff,
        )
        self.use_encoder_only = use_encoder_only

    def load_model_tokenizer(self) -> None:
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)

            self.__logger__.info("Loading Ankh2 tokenizer from: %s", local_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir, do_lower_case=False, use_fast=True, trust_remote_code=True
            )
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                if getattr(self.tokenizer, "pad_token", None) is None:
                    if getattr(self.tokenizer, "eos_token", None) is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.__logger__.debug("pad_token_id set to: %s", self.tokenizer.pad_token_id)

            self.__logger__.info("Loading Ankh2 encoder from: %s on device=%s", local_dir, self.device)
            self.model = T5EncoderModel.from_pretrained(local_dir, trust_remote_code=True).to(self.device)

            self.model.eval()
        except Exception as e:
            self.status = False
            self.message = f"Failed to load Ankh2 tokenizer/model: {e}"
            self.__logger__.error(self.message)
            raise

    @torch.no_grad()
    def embedding_batch(
        self,
        batch: List[str],
        max_length: int = 1024,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        if not batch:
            raise ValueError("Input batch is empty.")
        self.load_model_tokenizer()

        enc = self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        amp_dtype = self._amp_dtype()
        use_amp = (self.device.type == "cuda") and (amp_dtype is not None)

        if self.use_encoder_only and hasattr(self.model, "encoder"):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.model.encoder(input_ids=enc["input_ids"], output_hidden_states=True)
            else:
                out = self.model.encoder(input_ids=enc["input_ids"], output_hidden_states=True)
        else:
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.model(**enc, output_hidden_states=True)
            else:
                out = self.model(**enc, output_hidden_states=True)

        hs = out.hidden_states if getattr(out, "hidden_states", None) is not None else (out.last_hidden_state,)
        attn = enc.get("attention_mask", None)
        if attn is None:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            attn = enc["input_ids"].ne(pad_id).to(hs[0].dtype)
        return hs, attn
