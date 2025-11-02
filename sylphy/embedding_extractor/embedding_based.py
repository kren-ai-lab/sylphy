# sylphy/embedding_extraction/embedding_based.py
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from sylphy.core.model_registry import ModelSpec, register_model, resolve_model
from sylphy.logging import add_context, get_logger
from sylphy.misc.utils_lib import UtilsLib

Pool = Literal["mean", "cls", "eos"]
LayerAgg = Literal["mean", "sum", "concat"]
LayerSpec = Union[str, int, Sequence[int]]


class EmbeddingBased:
    """
    Base class for embedding extraction from protein sequences.

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
        revision: Optional[str] = None,
        column_seq: str = "sequence",
        debug: bool = False,
        debug_mode: int = logging.INFO,
        name_logging: str = "EmbeddingBased",  # deprecated; kept for context info
        trust_remote_code: bool = False,
        precision: Literal["fp32", "fp16", "bf16"] = "fp32",
        oom_backoff: bool = True,
    ) -> None:
        self._cache_root = UtilsLib.get_cache_dir()  # <- resuelve _siteconfig/env/platformdirs
        self._wire_cache_envs(self._cache_root)

        self.dataset = dataset
        self.column_seq = column_seq

        self.name_model = name_model
        self.name_tokenizer = name_tokenizer or name_model
        self.provider = provider
        self.revision = revision

        self.device = torch.device(
            name_device if torch.cuda.is_available() and name_device == "cuda" else "cpu"
        )
        self.trust_remote_code = trust_remote_code
        self.precision = precision
        self.oom_backoff = oom_backoff

        # logger
        self.__logger__ = get_logger("sylphy.embedding_extraction.base")
        if debug:
            self.__logger__.setLevel(debug_mode)
        add_context(
            self.__logger__,
            component="embedding_extraction",
            backend=name_logging,
            model=self.name_model or "<unset>",
        )

        self.tokenizer = None
        self.model = None

        self.requires_tokenizer: bool = self.provider == "huggingface"

        self.status: bool = True
        self.message: str = ""

    @staticmethod
    def _wire_cache_envs(root: Path) -> None:
        root = Path(root).expanduser()
        # Carpeta unificada de Sylphy
        os.environ.setdefault("SYLPHY_CACHE_DIR", str(root))

        # Hugging Face / Transformers / Datasets
        os.environ.setdefault("HF_HOME", str(root / "hf"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "hf" / "transformers"))
        os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf" / "datasets"))

        # Torch hub
        os.environ.setdefault("TORCH_HOME", str(root / "torch"))

        # (opcional) Tokenizers (a veces usan su propia caché)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _hf_cache_dir_models(self) -> str:
        # subcarpeta consistente para modelos HF bajo el cache de Sylphy
        return str(Path(self._cache_root) / "models" / "huggingface")

    # ---------------------------------------------------------------------
    # Model resolution & loading
    # ---------------------------------------------------------------------

    def _register_and_resolve(self) -> str:
        """
        Resolve the model via the registry. If missing and provider is HF
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
        """
        Idempotent loader. If not ready, tries:
        - subclass's `load_model_tokenizer()` if present (custom backends),
        - else falls back to `load_hf_tokenizer_and_model()` (HF backends).
        """
        if self._is_ready():
            return
        if hasattr(self, "load_model_tokenizer"):
            try:
                # type: ignore[attr-defined]
                self.load_model_tokenizer()  # subclass-provided (e.g., ESM-C)
            except Exception as e:
                self.__logger__.error("load_model_tokenizer() failed: %s", e)
                raise
        elif self.provider == "huggingface":
            self.load_hf_tokenizer_and_model()
        else:
            raise RuntimeError("No loader available to initialize model/tokenizer.")

    # ---------------------------------------------------------------------
    # Hooks & utilities
    # ---------------------------------------------------------------------

    def _pre_tokenize(self, batch: List[str]) -> List[str]:
        """Optional hook to adjust raw strings before tokenization (e.g., insert spaces)."""
        return [s.strip() for s in batch]

    def _amp_dtype(self) -> Optional[torch.dtype]:
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "bf16":
            return torch.bfloat16
        return None

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
        if (self.model is None) or (self.requires_tokenizer and self.tokenizer is None):
            raise RuntimeError("Model/tokenizer not loaded. Call load_model_tokenizer() before forward.")

        batch = self._pre_tokenize(sequences)
        enc = self.tokenizer(
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
                out = self.model(**enc, output_hidden_states=True)
        else:
            out = self.model(**enc, output_hidden_states=True)

        hidden_states = out.hidden_states  # type: ignore[attr-defined]
        if hidden_states is None:
            hidden_states = (out.last_hidden_state,)  # type: ignore[attr-defined]

        attn = enc.get("attention_mask", None)
        if attn is None:
            attn = torch.ones(out.last_hidden_state.shape[:2], device=self.device)  # type: ignore[attr-defined]
        return hidden_states, attn

    # ---------------------------------------------------------------------
    # Layer & token aggregation
    # ---------------------------------------------------------------------

    @staticmethod
    def _parse_layers(spec: LayerSpec, n_layers: int) -> List[int]:
        """Normalize layer spec ('last'|'all'|int|[ints]|'last4') into a sorted list of indices."""
        if isinstance(spec, int):
            return [spec]
        if isinstance(spec, (list, tuple)):
            return sorted(list({int(i) for i in spec}))
        if isinstance(spec, str):
            if spec == "last":
                return [n_layers - 1]
            if spec == "all":
                return list(range(n_layers))
            if spec == "last4":
                return list(range(max(0, n_layers - 4), n_layers))
            try:
                i = int(spec)
                return [i]
            except ValueError:
                pass
        raise ValueError(f"Invalid layer spec: {spec!r}")

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
        raise ValueError(f"Unknown pool '{pool}'")

    @staticmethod
    def _aggregate_layers(
        hs: Tuple[torch.Tensor, ...],  # tuple of (B, L, H)
        select: List[int],
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
        raise ValueError(f"Unknown layer aggregation '{agg}'")

    # ---------------------------------------------------------------------
    # Public API
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
        if not self.requires_tokenizer:
            raise NotImplementedError(
                "encode_batch_layers is only available for HuggingFace-style backends. "
                "Backends without tokenizer must implement `embedding_process()`."
            )

        self.ensure_loaded()
        if not self._is_ready():
            raise RuntimeError("Model/tokenizer not loaded. Call load_model_tokenizer() before forward.")

        hidden_states, attn = self._forward_hidden_states(sequences, max_length=max_length)
        n_layers = len(hidden_states)
        select = self._parse_layers(layers, n_layers)

        reps = self._aggregate_layers(hidden_states, select, layer_agg)  # (B, L, H' or H)
        pooled = self._pool_tokens(reps, attn, pool)  # (B, H' or H)
        return pooled.detach().cpu().numpy()

    def clean_memory(self):
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            import gc

            gc.collect()
        except Exception as e:
            self.__logger__.debug("clean_memory() warning: %s", e)

    def release_resources(self):
        try:
            if getattr(self, "model", None) is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            try:
                del self.tokenizer
            except Exception:
                pass
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
        """
        Encode all sequences in the dataset and store `self.coded_dataset`.

        Routing:
        - If `requires_tokenizer=False` and subclass implements `embedding_process`,
          delegate to that pipeline (non-HF backends).
        - Else, use the HF path (tokenizer + model).
        """
        try:
            self.ensure_loaded()
            if not self._is_ready():
                raise RuntimeError("Model/tokenizer not loaded. Call load_model_tokenizer() before forward.")

            if not self.requires_tokenizer and hasattr(self, "embedding_process"):
                df = self.embedding_process(  # type: ignore[attr-defined]
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

            self.release_resources()
            Xall = np.vstack(mats) if mats else np.zeros((0, 0), dtype=np.float32)
            header = [f"p_{i}" for i in range(Xall.shape[1])]
            self.coded_dataset = pd.DataFrame(Xall, columns=header, index=self.dataset.index)
            self.coded_dataset[self.column_seq] = self.dataset[self.column_seq].values
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
        path: Union[str, Path],
        file_format: Literal["csv", "npy", "npz", "parquet"] = "csv",
    ) -> None:
        """Persist the encoded matrix to disk."""
        UtilsLib.export_data(
            df_encoded=self.coded_dataset,
            path=path,
            base_message="Embeddings",
            file_format=file_format,
        )
