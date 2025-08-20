from typing import Optional, List
import logging
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .embedding_based import EmbeddingBased

class ESMBasedEmbedding(EmbeddingBased):

    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "facebook/esm2_t6_8M_UR50D",
        name_tokenizer: str = "facebook/esm2_t6_8M_UR50D",
        dataset: Optional[object] = None,
        column_seq: Optional[str] = "sequence",
        debug: bool = True,
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

    # ----------------------------- #
    #   Loading (registry -> local)   #
    # ----------------------------- #
    def load_model_tokenizer(self) -> None:
        """
        Resolve (and auto-register if needed) → load tokenizer/model from local dir.
        """
        try:
            local_dir = self._register_and_resolve()

            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=False)

            self.__logger__.info(f"Loading ESM tokenizer from: {local_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                do_lower_case=False,
                use_fast=True,
                trust_remote_code=False,
            )

            if getattr(self.tokenizer, "pad_token_id", None) is None:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.__logger__.debug(f"pad_token_id set to: {self.tokenizer.pad_token_id}")

            self.__logger__.info(f"Loading ESM model from: {local_dir} on device={self.device}")
            self.model = AutoModel.from_pretrained(
                local_dir,
                trust_remote_code=False,
            ).to(self.device)
            self.model.eval()

        except Exception as e:
            self.status = False
            self.message = f"Failed to load ESM tokenizer/model: {e}"
            self.__logger__.error(self.message)
            raise

    def _format_batch_for_esm(self, batch: List[str]) -> List[str]:

        vocab = getattr(self.tokenizer, "get_vocab", lambda: {})()
        has_cls = "<cls>" in vocab

        if has_cls:
            return [f"<cls> {' '.join(seq.strip())}" for seq in batch]
        else:
            return [seq.strip() for seq in batch]

    @torch.no_grad()
    def embedding_batch(
        self,
        batch: List[str],
        max_length: int = 1024
    ):
        """
        Returns:
          last_hidden_state (B, L, H), attention_mask (B, L)
        """
        if not batch:
            msg = "Input batch is empty. Cannot perform embedding."
            self.__logger__.error(msg)
            raise ValueError(msg)

        try:
            formatted = self._format_batch_for_esm(batch)
            vocab = getattr(self.tokenizer, "get_vocab", lambda: {})()
            has_cls = "<cls>" in vocab

            enc = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=not has_cls,   
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            amp_dtype = self._amp_dtype()
            use_amp = (self.device.type == "cuda") and (amp_dtype is not None)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.model(**enc, output_hidden_states=False)
            else:
                out = self.model(**enc, output_hidden_states=False)

            last_hidden = out.last_hidden_state
            attn = enc.get("attention_mask", None)
            return last_hidden, attn

        except Exception as e:
            msg = f"Error during ESM embedding batch: {e}"
            self.__logger__.error(msg)
            raise RuntimeError(msg)
