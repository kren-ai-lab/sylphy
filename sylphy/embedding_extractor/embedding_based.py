"""Provide the shared base implementation for embedding backends."""

from __future__ import annotations

import contextlib
import gc
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from sylphy.core.model_registry import ModelSpec, register_model, resolve_model
from sylphy.logging import add_context, get_logger
from sylphy.misc.utils_lib import UtilsLib
from sylphy.types import FileFormat, LayerAggType, PoolType, PrecisionType

Pool = PoolType
LayerAgg = LayerAggType
LayerSpec = str | int | Sequence[int]


class EmbeddingBased:
    """Base class for embedding extraction from protein sequences.

    Supports:
    - HuggingFace-style backends (model + tokenizer).
    - Non-tokenizer backends (e.g., ESM-C) via `embedding_process()`.

    Behavior:
    - If `requires_tokenizer=True` (default for provider="huggingface"), the HF path is used.
    - If `requires_tokenizer=False`, `run_process()` delegates to `embedding_process()`.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "",
        name_tokenizer: str = "",
        provider: str = "huggingface",  # {"huggingface","other"}
        revision: str | None = None,
        column_seq: str = "sequence",
        debug_mode: int = logging.INFO,
        precision: PrecisionType = "fp32",
        *,
        debug: bool = False,
        trust_remote_code: bool = False,
        oom_backoff: bool = True,
    ) -> None:
        """Initialize common state for embedding backends."""
        self.dataset: pd.DataFrame = dataset
        self.column_seq = column_seq

        self.name_model = name_model
        self.name_tokenizer = name_tokenizer or name_model
        self.provider = provider
        self.revision = revision

        self._device: Any = torch.device(
            name_device if torch.cuda.is_available() and name_device == "cuda" else "cpu",
        )
        self.trust_remote_code = trust_remote_code
        self.precision = precision
        self.oom_backoff = oom_backoff

        # logger
        backend_name = type(self).__name__
        self.__logger__ = get_logger(f"sylphy.embedding_extraction.{backend_name}")
        if debug:
            self.__logger__.setLevel(debug_mode)
        add_context(
            self.__logger__,
            component="embedding_extraction",
            backend=backend_name,
            model=self.name_model or "<unset>",
        )

        self.tokenizer: Any | None = None
        self.model: Any | None = None

        self.requires_tokenizer: bool = self.provider == "huggingface"

        self.status: bool = True
        self.message: str = ""

    @property
    def device(self) -> torch.device:
        """Return the torch device used for model execution."""
        return cast("torch.device", self._device)

    # ---------------------------------------------------------------------
    # Model resolution & loading
    # ---------------------------------------------------------------------

    def _register_and_resolve(self) -> str:
        """Resolve a model through the registry.

        If missing and provider is HF and ``name_model`` looks like ``org/model``,
        register it dynamically and resolve again.

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
        """Load tokenizer and model from local cache/registry into `self.device`."""
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
            self.model = AutoModel.from_pretrained(local_dir, trust_remote_code=self.trust_remote_code)  # type: ignore[possibly-missing-attribute]
            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Failed to load tokenizer/model: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message) from e

    # ---------------------------------------------------------------------
    # Readiness helpers
    # ---------------------------------------------------------------------

    def _is_ready(self) -> bool:
        """Check if model (and tokenizer, when required) are loaded."""
        has_model = getattr(self, "model", None) is not None
        if not has_model:
            return False
        if self.requires_tokenizer:
            return getattr(self, "tokenizer", None) is not None
        return True

    def ensure_loaded(self) -> None:
        """Load model resources when they are not already loaded.

        Preference order:
        - subclass ``load_model_tokenizer()`` when provided
        - fallback to ``load_hf_tokenizer_and_model()`` for HF backends
        """
        if self._is_ready():
            return
        loader: Callable[[], None] | None = getattr(self, "load_model_tokenizer", None)
        if callable(loader):
            try:
                loader()  # subclass-provided (e.g., ESM-C)
            except Exception as e:
                self.__logger__.error("load_model_tokenizer() failed: %s", e)
                raise
        elif self.provider == "huggingface":
            self.load_hf_tokenizer_and_model()
        else:
            msg = "No loader available to initialize model/tokenizer."
            raise RuntimeError(msg)

    # ---------------------------------------------------------------------
    # Hooks & utilities
    # ---------------------------------------------------------------------

    def _pre_tokenize(self, batch: list[str]) -> list[str]:
        """Adjust raw input sequences before tokenization."""
        return [s.strip() for s in batch]

    def _amp_dtype(self) -> torch.dtype | None:
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "bf16":
            return torch.bfloat16
        return None

    def embedding_process(
        self,
        batch_size: int = 32,
        *,
        seq_len: int | None = None,
        layers: LayerSpec = "last",
        layer_agg: LayerAgg = "mean",
        pool: Pool = "mean",
    ) -> pd.DataFrame:
        """Process embeddings for non-tokenizer backends.

        Subclasses that set ``requires_tokenizer = False`` must override this method.
        """
        msg = "Non-tokenizer backends must implement embedding_process()."
        raise NotImplementedError(msg)

    def _make_batches(self, seqs: list[str], batch_size: int) -> list[list[str]]:
        return [seqs[i : i + batch_size] for i in range(0, len(seqs), batch_size)]

    # ---------------------------------------------------------------------
    # Forward helpers (HuggingFace-style)
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def _forward_hidden_states(
        self,
        sequences: list[str],
        *,
        max_length: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Tokenize → forward pass with `output_hidden_states=True`.

        Returns:
            Tuple of ``(hidden_states, attention_mask)`` where hidden states are
            ``(B, L, H)`` per layer and attention mask is ``(B, L)``.

        """
        if (self.model is None) or (self.requires_tokenizer and self.tokenizer is None):
            msg = "Model/tokenizer not loaded. Call load_model_tokenizer() before forward."
            raise RuntimeError(msg)

        tokenizer = self.tokenizer
        model = self.model
        if tokenizer is None or model is None:
            msg = "Model/tokenizer not loaded."
            raise RuntimeError(msg)

        batch = self._pre_tokenize(sequences)
        enc = tokenizer(
            batch,
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
                out = model(**enc, output_hidden_states=True)
        else:
            out = model(**enc, output_hidden_states=True)

        hidden_states = out.hidden_states
        if hidden_states is None:
            hidden_states = (out.last_hidden_state,)

        attn = enc.get("attention_mask")
        if attn is None:
            attn = torch.ones(out.last_hidden_state.shape[:2], device=self.device)
        return hidden_states, attn

    # ---------------------------------------------------------------------
    # Layer & token aggregation
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_layers(spec: LayerSpec, n_layers: int) -> list[int]:
        """Normalize layer spec ('last'|'all'|int|[ints]|'last4') into a sorted list of indices."""
        if isinstance(spec, int):
            return [spec]
        if isinstance(spec, (list, tuple)):
            layer_items = cast("Sequence[Any]", spec)
            return sorted({int(i) for i in layer_items})
        if isinstance(spec, str):
            if spec == "last":
                return [n_layers - 1]
            if spec == "all":
                return list(range(n_layers))
            if spec == "last4":
                return list(range(max(0, n_layers - 4), n_layers))
            try:
                i = int(spec)
            except ValueError:
                pass
            else:
                return [i]
        msg = f"Invalid layer spec: {spec!r}"
        raise ValueError(msg)

    @staticmethod
    def _pool_tokens(
        reps: torch.Tensor,  # (B, L, H)
        attn: torch.Tensor,  # (B, L)
        pool: Pool,
    ) -> torch.Tensor:
        """Pool tokens into a single (B, H) representation."""
        if pool == "cls":
            return reps[:, 0, :]
        if pool == "eos":
            lengths = attn.sum(dim=1).long() - 1
            return reps[torch.arange(reps.shape[0]), lengths, :]
        if pool == "mean":
            mask = attn.unsqueeze(-1).to(reps.dtype)
            num = (reps * mask).sum(dim=1)
            den = mask.sum(dim=1).clamp_min(1e-6)
            return num / den
        msg = f"Unknown pool '{pool}'"
        raise ValueError(msg)

    @staticmethod
    def _aggregate_layers(
        hs: tuple[torch.Tensor, ...],  # tuple of (B, L, H)
        select: list[int],
        agg: LayerAgg,
    ) -> torch.Tensor:
        """Aggregate selected layers into a single (B, L, H*) representation."""
        chosen = [hs[i] for i in select]
        if agg == "concat":
            return torch.cat(chosen, dim=-1)
        if agg == "mean":
            return torch.stack(chosen, dim=0).mean(dim=0)
        if agg == "sum":
            return torch.stack(chosen, dim=0).sum(dim=0)
        msg = f"Unknown layer aggregation '{agg}'"
        raise ValueError(msg)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def encode_batch_layers(
        self,
        sequences: list[str],
        *,
        max_length: int = 1024,
        layers: LayerSpec = "last",
        layer_agg: LayerAgg = "mean",
        pool: Pool = "mean",
    ) -> np.ndarray:
        """Encode a batch returning pooled, layer-aggregated embeddings.

        Returns:
            Array of shape ``(B, H')`` where ``H'`` depends on layer selection
            and aggregation.

        """
        if not self.requires_tokenizer:
            msg = (
                "encode_batch_layers is only available for HuggingFace-style backends. "
                "Backends without tokenizer must implement `embedding_process()`."
            )
            raise NotImplementedError(
                msg,
            )

        self.ensure_loaded()
        if not self._is_ready():
            msg = "Model/tokenizer not loaded. Call load_model_tokenizer() before forward."
            raise RuntimeError(msg)

        hidden_states, attn = self._forward_hidden_states(sequences, max_length=max_length)
        n_layers = len(hidden_states)
        select = self._parse_layers(layers, n_layers)

        reps = self._aggregate_layers(hidden_states, select, layer_agg)  # (B, L, H' or H)
        pooled = self._pool_tokens(reps, attn, pool)  # (B, H' or H)
        return pooled.detach().cpu().numpy()

    def clean_memory(self) -> None:
        """Release Python and CUDA memory caches when available."""
        try:
            if torch.cuda.is_available():
                with contextlib.suppress(Exception):
                    torch.cuda.synchronize()
                torch.cuda.empty_cache()
                with contextlib.suppress(Exception):
                    torch.cuda.reset_peak_memory_stats()
            gc.collect()
        except Exception as e:  # noqa: BLE001
            self.__logger__.debug("clean_memory() warning: %s", e)

    def release_resources(self) -> None:
        """Move model off-device and clear model/tokenizer references."""
        try:
            model = getattr(self, "model", None)
            if model is not None:
                with contextlib.suppress(Exception):
                    model.to("cpu")
            with contextlib.suppress(Exception):
                del self.model
            self.model = None
            with contextlib.suppress(Exception):
                del self.tokenizer
            self.tokenizer = None
        finally:
            self.clean_memory()

    def run_process(
        self,
        *,
        max_length: int = 1024,
        batch_size: int = 8,
        layers: LayerSpec = "last",
        layer_agg: LayerAgg = "mean",
        pool: Pool = "mean",
    ) -> None:
        """Encode all sequences in the dataset and store `self.coded_dataset`.

        Routing:
        - If `requires_tokenizer=False` and subclass implements `embedding_process`,
          delegate to that pipeline (non-HF backends).
        - Else, use the HF path (tokenizer + model).
        """
        self.ensure_loaded()
        if not self._is_ready():
            msg = "Model/tokenizer not loaded. Call load_model_tokenizer() before forward."
            raise RuntimeError(msg)

        try:
            if not self.requires_tokenizer:
                df = self.embedding_process(
                    batch_size=batch_size,
                    seq_len=max_length,
                    layers=layers,
                    layer_agg=layer_agg,
                    pool=pool,
                )
                self.release_resources()
                self.coded_dataset = df
                self.status = True
                self.message = "OK"
                self.__logger__.info(
                    "Embedding extraction (non-HF) complete. Shape=%s | layers=%s | layer_agg=%s | pool=%s",
                    df.shape,
                    layers,
                    layer_agg,
                    pool,
                )
                return

            seqs = self.dataset[self.column_seq].astype(str).tolist()
            mats: list[np.ndarray] = []
            bs: int = int(batch_size)
            bs = max(bs, 1)

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
                    new_bs: int = bs // 2
                    new_bs = max(new_bs, 1)
                    self.__logger__.warning("CUDA OOM at bs=%d. Retrying with bs=%d.", bs, new_bs)
                    bs = new_bs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        with contextlib.suppress(Exception):
                            torch.cuda.reset_peak_memory_stats()
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

            self.release_resources()
            Xall: np.ndarray = np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)
            header: list[str] = [f"p_{i}" for i in range(Xall.shape[1])]
            columns_index = pd.Index(header)
            self.coded_dataset = pd.DataFrame(Xall, columns=columns_index, index=self.dataset.index)
            self.coded_dataset[self.column_seq] = self.dataset[self.column_seq].to_numpy()
            self.status = True
            self.message = "OK"
            self.__logger__.info(
                "Embedding extraction complete. Shape=%s | layers=%s | layer_agg=%s | pool=%s",
                Xall.shape,
                layers,
                layer_agg,
                pool,
            )
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] run_process failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message) from e

    def export_encoder(
        self,
        path: str | Path,
        file_format: FileFormat = "csv",
    ) -> None:
        """Persist the encoded matrix to disk."""
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            base_message="Embeddings",
            file_format=file_format,
        )
