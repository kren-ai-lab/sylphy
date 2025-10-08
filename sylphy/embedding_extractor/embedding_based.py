# sylphy/embedding_extraction/embedding_based.py
from __future__ import annotations

import logging
from abc import ABC
from typing import List, Optional, Literal, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from sylphy.logging import get_logger, add_context
from sylphy.core.model_registry import ModelSpec, resolve_model, register_model
from sylphy.misc.utils_lib import UtilsLib


LayerSpec = Union[str, int, Sequence[int]]
LayerAgg = Literal["mean", "sum", "concat"]
Pool = Literal["mean", "cls", "eos"]


class EmbeddingBased(ABC):
    """
    Base class for embedding extraction from protein sequences using HF-like models.

    - Unified logging under "sylphy.embedding_extraction.*".
    - Device & AMP handling (fp16/bf16) with safe CUDA fallback.
    - Batch utilities and robust OOM backoff (halve batch on OOM and retry).

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
    debug : bool, default=False
        If True, set this backend logger to `debug_mode`.
    debug_mode : int, default=logging.INFO
        Logging level when `debug=True`; ignored otherwise.
    trust_remote_code : bool, default=False
        Forwarded to HF `from_pretrained`.
    precision : {"fp32","fp16","bf16"}, default="fp32"
        AMP autocast dtype for CUDA.
    oom_backoff : bool, default=True
        If True, halves batch size on CUDA OOM and retries.
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
        debug: bool = False,
        debug_mode: int = logging.INFO,
        name_logging: str = "EmbeddingBased",  # deprecated; kept for context info
        trust_remote_code: bool = False,
        precision: Literal["fp32", "fp16", "bf16"] = "fp32",
        oom_backoff: bool = True,
    ) -> None:

        # --- dataset & columns
        self.dataset = dataset
        self.column_seq = column_seq

        # --- device
        self.name_device = name_device
        self.device = torch.device(
            self.name_device if (torch.cuda.is_available() or self.name_device == "cpu") else "cpu"
        )

        # --- identifiers
        self.name_model = name_model.strip()
        self.name_tokenizer = (name_tokenizer.strip() or self.name_model)
        self.provider = provider
        self.revision = revision
        self.trust_remote_code = bool(trust_remote_code)

        # --- execution
        self.precision = precision
        self.oom_backoff = bool(oom_backoff)

        # --- logging
        _ = get_logger("sylphy")  # ensure package root once
        self.__logger__ = logging.getLogger(f"sylphy.embedding_extraction.{name_logging}")
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(
            self.__logger__,
            component="embedding_extraction",
            backend=name_logging,
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
            if self.provider == "huggingface" and ("/" in self.name_model):
                register_model(ModelSpec(name=self.name_model, provider="huggingface", ref=self.name_model))
                local_dir = resolve_model(self.name_model)
                return str(local_dir)
            raise

    def load_hf_tokenizer_and_model(self) -> None:
        """
        Load tokenizer and model from local cache/registry into `self.device`.
        """
        try:
            local_dir = self._register_and_resolve()

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

    # ---------------------------------------------------------------------
    # Hooks & utilities
    # ---------------------------------------------------------------------

    def _pre_tokenize(self, batch: List[str]) -> List[str]:
        """
        Hook for subclasses to adapt raw sequences before tokenization.
        Default: identity.
        """
        return batch

    def _amp_dtype(self) -> Optional[torch.dtype]:
        if self.device.type != "cuda":
            return None
        return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, None)

    def _make_batches(self, seqs: List[str], batch_size: int) -> List[List[str]]:
        return [seqs[i : i + batch_size] for i in range(0, len(seqs), batch_size)]

    # ---------------------------------------------------------------------
    # Forward helpers (HuggingFace-style)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _forward_hidden_states(
        self,
        sequences: List[str],
        *,
        max_length: int,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Tokenize → forward pass with `output_hidden_states=True`.

        Returns
        -------
        hidden_states : tuple of Tensors
            Tuple of length n_layers: each (B, L, H).
        attention_mask : torch.Tensor
            (B, L) attention mask (1 for real tokens).
        """
        assert self.model is not None and self.tokenizer is not None, \
            "Call load_*_tokenizer_and_model() before forward."

        enc = self.tokenizer(
            self._pre_tokenize(sequences),
            return_tensors="pt",
            truncation=True,
            padding=True,
            add_special_tokens=True,
            max_length=max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        amp_dtype = self._amp_dtype()
        use_amp = (self.device.type == "cuda") and (amp_dtype is not None)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = self.model(**enc, output_hidden_states=True)
        else:
            out = self.model(**enc, output_hidden_states=True)

        hidden_states = out.hidden_states  # type: ignore[attr-defined]
        if hidden_states is None:
            # Fallback to just last_hidden_state if the backend doesn't return hidden states.
            hidden_states = (out.last_hidden_state,)  # type: ignore[attr-defined]

        attn = enc.get("attention_mask", None)
        if attn is None:
            # Some tokenizers might not return it; synthesize a full-ones mask.
            attn = torch.ones(out.last_hidden_state.shape[:2], device=self.device)  # type: ignore[attr-defined]
        return hidden_states, attn

    # ---------------------------------------------------------------------
    # Layer & token aggregation
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_layers(layers: LayerSpec, n_layers: int) -> List[int]:
        """
        Normalize a layer spec into a sorted list of 0-based indices.
        Supports: "last", "last4", "all", int, [ints] (negative allowed).
        """
        if isinstance(layers, str):
            key = layers.strip().lower()
            if key == "last":
                idx = [n_layers - 1]
            elif key == "last4":
                idx = list(range(max(0, n_layers - 4), n_layers))
            elif key == "all":
                idx = list(range(n_layers))
            else:
                raise ValueError(
                    f"Unknown layer spec '{layers}'. Use 'last'|'last4'|'all' or an int/list of ints."
                )
        elif isinstance(layers, int):
            idx = [layers]
        else:
            idx = list(layers)

        # Convert negatives and validate range
        fixed: List[int] = []
        for j in idx:
            jj = j if j >= 0 else n_layers + j
            if not (0 <= jj < n_layers):
                raise ValueError(f"Layer index {j} out of bounds (n_layers={n_layers}).")
            fixed.append(jj)
        return sorted(set(fixed))

    @staticmethod
    def _pool_tokens(
        reps: torch.Tensor,  # (B, L, H)
        attn: torch.Tensor,  # (B, L)
        pool: Pool,
    ) -> torch.Tensor:  # (B, H) or (B, H*)
        if pool == "mean":
            mask = attn.unsqueeze(-1).float()
            summed = (reps * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom
        if pool == "cls":
            return reps[:, 0, :]
        if pool == "eos":
            idx = attn.sum(dim=1) - 1
            return reps[torch.arange(reps.size(0), device=reps.device), idx, :]
        raise ValueError(f"Unknown token pool strategy '{pool}'")

    @staticmethod
    def _aggregate_layers(
        hs: Tuple[torch.Tensor, ...],
        select: List[int],
        agg: LayerAgg,
    ) -> torch.Tensor:
        """
        Aggregate selected layers into a single (B, L, H*) representation.
        """
        chosen = [hs[i] for i in select]
        if agg == "concat":
            return torch.cat(chosen, dim=-1)
        if agg == "mean":
            return torch.stack(chosen, dim=0).mean(dim=0)
        if agg == "sum":
            return torch.stack(chosen, dim=0).sum(dim=0)
        raise ValueError(f"Unknown layer aggregation '{agg}'")

    # ---------------------------------------------------------------------
    # Public HF-style API (end-to-end on batches / dataset)
    # ---------------------------------------------------------------------

    def encode_batch_layers(
        self,
        sequences: List[str],
        *,
        max_length: int = 1024,
        layers: LayerSpec = "last",
        layer_agg: LayerAgg = "mean",
        pool: Pool = "mean",
    ) -> np.ndarray:
        """
        Encode a batch returning pooled, layer-aggregated embeddings.

        Returns
        -------
        np.ndarray
            (B, H') with H' depending on `layer_agg` and the number of selected layers.
        """
        hidden_states, attn = self._forward_hidden_states(sequences, max_length=max_length)
        n_layers = len(hidden_states)
        select = self._parse_layers(layers, n_layers)

        reps = self._aggregate_layers(hidden_states, select, layer_agg)  # (B, L, H' or H)
        pooled = self._pool_tokens(reps, attn, pool)                    # (B, H' or H)
        return pooled.detach().cpu().numpy()

    def run_process(
        self,
        *,
        max_length: int = 1024,
        batch_size: int = 8,
        layers: LayerSpec = "last",
        layer_agg: LayerAgg = "mean",
        pool: Pool = "mean",
    ) -> None:
        """
        Encode all sequences in the dataset using the selected layers and pooling.
        Stores the matrix in `self.coded_dataset`.
        """
        try:
            seqs = self.dataset[self.column_seq].astype(str).tolist()
            mats: List[np.ndarray] = []
            bs = max(1, int(batch_size))

            for chunk in self._make_batches(seqs, bs):
                try:
                    X = self.encode_batch_layers(
                        chunk,
                        max_length=max_length,
                        layers=layers,
                        layer_agg=layer_agg,
                        pool=pool,
                    )
                    mats.append(X)
                except torch.cuda.OutOfMemoryError:
                    if not (self.oom_backoff and bs > 1 and self.device.type == "cuda"):
                        raise
                    new_bs = max(1, bs // 2)
                    self.__logger__.warning("CUDA OOM at bs=%d. Retrying with bs=%d.", bs, new_bs)
                    bs = new_bs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        try:
                            torch.cuda.reset_peak_memory_stats()
                        except Exception:
                            pass
                    # retry this same chunk with smaller batch
                    for sub in self._make_batches(chunk, bs):
                        X = self.encode_batch_layers(
                            sub,
                            max_length=max_length,
                            layers=layers,
                            layer_agg=layer_agg,
                            pool=pool,
                        )
                        mats.append(X)

            Xall = np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)
            header = [f"p_{i}" for i in range(Xall.shape[1])]
            self.coded_dataset = pd.DataFrame(Xall, columns=header, index=self.dataset.index)
            self.coded_dataset[self.column_seq] = self.dataset[self.column_seq].values
            self.status = True
            self.message = "OK"
            self.__logger__.info(
                "Embedding extraction complete. Shape=%s | layers=%s | layer_agg=%s | pool=%s",
                Xall.shape, layers, layer_agg, pool
            )
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] run_process failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)

    def export_encoder(self, path: str, file_format: Literal["csv", "npy", "npz", "parquet"] = "csv") -> None:
        """
        Persist the encoded matrix to disk.
        """
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            base_message="Embeddings",
            file_format=file_format,
        )
