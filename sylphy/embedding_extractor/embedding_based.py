# protein_representation/embedding_extraction/embedding_based.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from abc import ABC
from typing import List, Optional, Literal
from transformers import AutoTokenizer, AutoModel, AutoConfig

from sylphy.logging import get_logger, add_context
from sylphy.core.model_registry import ModelSpec, resolve_model, register_model
from sylphy.core.config import ToolConfig
from sylphy.misc.utils_lib import UtilsLib


class EmbeddingBased(ABC):
    """
    Base class for embedding extraction from protein sequences using HF models.

    This class standardizes device handling, tokenizer/model loading from the local
    registry/cache, mixed precision (optional), and a consistent logging strategy.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataframe with a column of sequences (`column_seq`).
    name_device : {"cuda","cpu"}, default=auto
        Preferred device; falls back to CPU if CUDA not available.
    name_model : str
        Model identifier (HF ref or registry key).
    name_tokenizer : str
        Tokenizer identifier; defaults to `name_model`.
    provider : {"huggingface","other"}, default="huggingface"
        Where to resolve `name_model` from (registry/provider).
    revision : str, optional
        HF revision (branch/tag/SHA).
    column_seq : str, default="sequence"
        Column name holding raw sequences.
    debug : bool, default=ToolConfig.debug
        If True, set this backend logger to DEBUG (file handler logs everything).
    debug_mode : int, default=ToolConfig.log_level
        Logging level when `debug=True`; ignored otherwise.
    name_logging : str, deprecated
        Kept for backwards compatibility; ignored in favor of hierarchical loggers.
    trust_remote_code : bool, default=False
        Forwarded to HF `from_pretrained`.
    precision : {"fp32","fp16","bf16"}, default="fp32"
        AMP autocast dtype for CUDA.
    oom_backoff : bool, default=True
        If True, halves batch size on CUDA OOM and retries once.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "",
        name_tokenizer: str = "",
        provider: str = "huggingface",
        revision: Optional[str] = None,
        column_seq: str = "sequence",
        debug: bool = ToolConfig.debug,
        debug_mode: int = ToolConfig.log_level,
        name_logging: str = "protein_representation.embedding",  # deprecated (ignored)
        trust_remote_code: bool = False,
        precision: Literal["fp32", "fp16", "bf16"] = "fp32",
        oom_backoff: bool = True,
    ) -> None:

        # --- dataset & columns
        self.dataset = dataset
        self.column_seq = column_seq

        # --- device
        self.name_device = name_device
        self.device = torch.device(self.name_device if torch.cuda.is_available() or name_device == "cpu" else "cpu")

        # --- identifiers
        self.name_model = name_model.strip()
        self.name_tokenizer = (name_tokenizer.strip() or self.name_model)
        self.provider = provider
        self.revision = revision
        self.trust_remote_code = bool(trust_remote_code)

        # --- execution
        self.precision = precision
        self.oom_backoff = bool(oom_backoff)

        # --- unified logging ----------------------------------------------
        # Ensure package-level logger is configured once
        _ = get_logger("protein_representation")
        import logging
        self.__logger__ = logging.getLogger(
            f"protein_representation.embedding_extraction.{EmbeddingBased.__name__}"
        )
        # Inherit parent level unless debug=True for this encoder
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(
            self.__logger__,
            component="embedding_extraction",
            backend=EmbeddingBased.__name__,
            model=self.name_model or "<unset>",
        )

        self.tokenizer = None
        self.model = None

        self.status: bool = True
        self.message: str = ""

    # ---------------------------------------------------------------------
    # Model resolution & loading
    # ---------------------------------------------------------------------

    def _register_and_resolve(self) -> str:
        """
        Try to resolve the model via the registry. If missing and provider is HF
        and `name_model` looks like 'org/model', register it, then resolve.
        Returns a local directory path to the model.
        """
        try:
            local_dir = resolve_model(self.name_model)
            return str(local_dir)
        except Exception:
            # If not registered and it's a HF ref, self-register
            if self.provider == "huggingface" and ("/" in self.name_model):
                try:
                    register_model(ModelSpec(name=self.name_model, provider="huggingface", ref=self.name_model))
                    local_dir = resolve_model(self.name_model)
                    return str(local_dir)
                except Exception as e:
                    self.__logger__.error("Failed to self-register model '%s': %s", self.name_model, e)
                    raise
            # else: let the original error surface
            raise

    # ---------------------------------------------------------------------
    # Tokenization / batching utils
    # ---------------------------------------------------------------------

    def _amp_dtype(self) -> Optional[torch.dtype]:
        if self.device.type != "cuda":
            return None
        return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, None)

    def _make_batches(self, seqs: List[str], batch_size: int) -> List[List[str]]:
        return [seqs[i : i + batch_size] for i in range(0, len(seqs), batch_size)]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def load_hf_tokenizer_and_model(self) -> None:
        """
        Load tokenizer and model from local cache/registry into `self.device`.
        """
        try:
            local_dir = self._register_and_resolve()

            # Optional read of config (e.g., hidden size)
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=self.trust_remote_code)

            self.__logger__.info("Loading tokenizer from: %s", local_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                do_lower_case=False,
                use_fast=True,
                trust_remote_code=self.trust_remote_code,
            )

            # Ensure a pad token exists
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                # fallbacks: eos_token or add [PAD]
                if getattr(self.tokenizer, "pad_token", None) is None:
                    if getattr(self.tokenizer, "eos_token", None) is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                self.__logger__.debug("pad_token_id set to: %s", self.tokenizer.pad_token_id)

            self.__logger__.info("Loading model from: %s (device=%s)", local_dir, self.device)
            self.model = AutoModel.from_pretrained(local_dir, trust_remote_code=self.trust_remote_code)
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Failed to load tokenizer/model: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)

    def encode_batch_last_hidden(
        self,
        sequences: List[str],
        max_length: int = 1024,
        batch_size: int = 8,
        pool: Literal["mean", "cls", "eos"] = "mean",
    ) -> np.ndarray:
        """
        Tokenize and run a batch through the model, returning pooled last hidden states.

        Parameters
        ----------
        sequences : list of str
        max_length : int
        batch_size : int
        pool : {"mean","cls","eos"}
            Pooling strategy.

        Returns
        -------
        np.ndarray
            (N, H) matrix of pooled embeddings.
        """
        assert self.model is not None and self.tokenizer is not None, "Call load_hf_tokenizer_and_model() first."

        try:
            enc = self.tokenizer(
                sequences,
                return_tensors="pt",
                truncation=True,
                padding=True,
                add_special_tokens=False,
                max_length=max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            amp_dtype = self._amp_dtype()
            use_amp = (self.device.type == "cuda") and (amp_dtype is not None)

            if use_amp:
                self.__logger__.debug("Using autocast dtype=%s for batch forward.", amp_dtype)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.model(**enc)
            else:
                out = self.model(**enc)

            last_hidden = out.last_hidden_state  # (B, L, H)

            if pool == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (last_hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            elif pool == "cls":
                pooled = last_hidden[:, 0, :]
            elif pool == "eos":
                # find last non-pad token per row
                idx = enc["attention_mask"].sum(dim=1) - 1
                pooled = last_hidden[torch.arange(last_hidden.size(0)), idx, :]
            else:
                raise ValueError(f"Unknown pool strategy: {pool}")

            return pooled.detach().cpu().numpy()

        except torch.cuda.OutOfMemoryError as oom:
            if self.oom_backoff and batch_size > 1:
                self.__logger__.warning("CUDA OOM at bs=%d. Retrying with bs=%d.", batch_size, batch_size // 2)
                return self.encode_batch_last_hidden(sequences, max_length, max(batch_size // 2, 1), pool)
            self.__logger__.exception("CUDA OOM without backoff.")
            raise
        except Exception as e:
            self.__logger__.exception("Encoding batch failed: %s", e)
            raise

    # ---------------------------------------------------------------------
    # Convenience: end-to-end on the dataset
    # ---------------------------------------------------------------------

    def run_process(
        self,
        max_length: int = 1024,
        batch_size: int = 8,
        pool: Literal["mean", "cls", "eos"] = "mean",
    ) -> None:
        """
        Encode all sequences in `self.dataset` and store the matrix in `coded_dataset`.
        """
        try:
            seqs = self.dataset[self.column_seq].astype(str).tolist()
            chunks = self._make_batches(seqs, batch_size)
            mats = [self.encode_batch_last_hidden(c, max_length=max_length, batch_size=len(c), pool=pool) for c in chunks]
            X = np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)

            header = [f"p_{i}" for i in range(X.shape[1])]
            self.coded_dataset = pd.DataFrame(X, columns=header, index=self.dataset.index)
            self.coded_dataset[self.column_seq] = self.dataset[self.column_seq].values
            self.status = True
            self.message = "OK"
            self.__logger__.info("Embedding extraction complete. Shape: %s", X.shape)
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] run_process failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)

    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------

    def export_encoder(self, path: str, file_format: Literal["csv", "npy"] = "csv") -> None:
        """
        Persist the encoded matrix to disk.
        """
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            __logger__=self.__logger__,
            base_message="Embeddings generated",
            file_format=file_format,
        )
