"""Implement the Ankh2 embedding backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from sylphy.core.optional_dependencies import wrap_optional_dependency_error

from .embedding_based import DEFAULT_DEBUG_MODE, DEFAULT_DEVICE, DEFAULT_PRECISION, EmbeddingBase

if TYPE_CHECKING:
    import pandas as pd

    from sylphy.types import PrecisionType


class Ankh2Embedding(EmbeddingBase):
    """Extract embeddings using Ankh2 models."""

    def __init__(
        self,
        dataset: pd.DataFrame,
        name_device: str = DEFAULT_DEVICE,
        name_model: str = "ElnaggarLab/ankh2-ext1",
        name_tokenizer: str = "ElnaggarLab/ankh2-ext1",
        column_seq: str = "sequence",
        debug_mode: int = DEFAULT_DEBUG_MODE,
        *,
        use_encoder_only: bool = True,
        debug: bool = False,
        precision: PrecisionType = DEFAULT_PRECISION,
        oom_backoff: bool = True,
    ) -> None:
        """Initialize the Ankh2 backend."""
        super().__init__(
            dataset=dataset,
            name_device=name_device,
            name_model=name_model,
            name_tokenizer=name_tokenizer,
            provider="huggingface",
            revision=None,
            column_seq=column_seq,
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            trust_remote_code=True,
            oom_backoff=oom_backoff,
        )
        self.use_encoder_only = use_encoder_only

    def load_model_tokenizer(self) -> None:
        """Load the Ankh2 tokenizer and encoder model."""
        self.release_resources()
        try:
            local_dir = self._register_and_resolve()
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)

            self.__logger__.info("Loading Ankh2 tokenizer from: %s", local_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                do_lower_case=False,
                use_fast=True,
                trust_remote_code=True,
            )
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                if getattr(self.tokenizer, "pad_token", None) is None:
                    if getattr(self.tokenizer, "eos_token", None) is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.__logger__.debug("pad_token_id set to: %s", self.tokenizer.pad_token_id)

            self.__logger__.info("Loading Ankh2 encoder from: %s on device=%s", local_dir, self.device)
            model = T5EncoderModel.from_pretrained(local_dir, trust_remote_code=True)  # type: ignore[possibly-missing-attribute]
            cast("nn.Module", model).to(self.device)
            self.model = model

            model.eval()
        except (ImportError, ModuleNotFoundError) as e:
            wrapped = wrap_optional_dependency_error(
                e,
                feature="Ankh2 embeddings",
                extra="embeddings",
                packages=("sentencepiece",),
            )
            if wrapped is not None:
                self.__logger__.error("%s", wrapped)
                raise wrapped from e
            raise
        except Exception as e:
            self.__logger__.error("Failed to load Ankh2 tokenizer/model: %s", e)
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

        tokenizer = self.tokenizer
        model = self.model
        if tokenizer is None or model is None:
            msg = "Tokenizer/model not loaded."
            raise RuntimeError(msg)

        enc = tokenizer(
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

        if self.use_encoder_only and hasattr(model, "encoder"):
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model.encoder(input_ids=enc["input_ids"], output_hidden_states=True)
            else:
                out = model.encoder(input_ids=enc["input_ids"], output_hidden_states=True)
        elif use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**enc, output_hidden_states=True)
        else:
            out = model(**enc, output_hidden_states=True)

        hs = (
            out.hidden_states if getattr(out, "hidden_states", None) is not None else (out.last_hidden_state,)
        )
        attn = enc.get("attention_mask")
        if attn is None:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            attn = enc["input_ids"].ne(pad_id).to(hs[0].dtype)
        return hs, attn
