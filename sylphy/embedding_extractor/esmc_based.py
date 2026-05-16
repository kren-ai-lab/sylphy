"""Implement the ESM-C backend using Meta's ESM SDK."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

from .embedding_based import EmbeddingBase, LayerAgg, LayerSpec, Pool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sylphy.types import PrecisionType


class ESMCEmbedding(EmbeddingBase):
    """ESM-C backend using Meta's ESM SDK."""

    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "esmc_300m",
        _name_tokenizer: str | None = None,
        dataset: pd.DataFrame | None = None,
        column_seq: str | None = "sequence",
        debug_mode: int = logging.INFO,
        precision: PrecisionType = "fp32",
        *,
        debug: bool = False,
        oom_backoff: bool = True,
    ) -> None:
        """Initialize the ESM-C backend."""
        if dataset is None:
            msg = "dataset must be provided"
            raise ValueError(msg)

        super().__init__(
            dataset=dataset,
            name_device=name_device,
            name_model=name_model,
            name_tokenizer="",
            provider="other",
            revision=None,
            column_seq=column_seq or "sequence",
            debug_mode=debug_mode,
            precision=precision,
            debug=debug,
            trust_remote_code=False,
            oom_backoff=oom_backoff,
        )
        self.requires_tokenizer = False
        self._embedding_dim: int | None = None
        self.model: ESMC | None = None

    def _has_hidden_states(self, hs: object) -> bool:
        if hs is None:
            return False
        if isinstance(hs, (list, tuple)):
            return len(hs) > 0
        if torch.is_tensor(hs):
            return hs.numel() > 0
        return False

    def ensure_loaded(self) -> None:
        """Idempotent loader."""
        if getattr(self, "model", None) is None:
            self.load_model_tokenizer()

    def load_model_tokenizer(self) -> None:
        """Load an ESM-C model from the registry path or model identifier."""
        try:
            local_dir: str | None = None
            try:
                local_dir = self._register_and_resolve()
            except Exception:  # noqa: BLE001
                local_dir = None

            load_ref = local_dir or self.name_model
            self.__logger__.info("Loading ESM-C from: %s on device=%s", load_ref, self.device)
            mdl = ESMC.from_pretrained(load_ref)
            mdl.to(self.device)
            mdl.eval()
            self.model = mdl
            self.__logger__.info("ESM-C model '%s' loaded successfully.", load_ref)
        except Exception as e:
            self.__logger__.error("Failed to load ESM-C model: %s", e)
            raise

    @torch.no_grad()
    def _embed_one(
        self,
        sequence: str,
        *,
        return_hidden_states: bool,
    ) -> tuple[torch.Tensor | None, Sequence[torch.Tensor] | None]:
        """Encode one sequence."""
        self.ensure_loaded()
        if self.model is None:
            msg = "ESM-C model not loaded."
            raise RuntimeError(msg)

        try:
            protein = ESMProtein(sequence=sequence)
            cfg = LogitsConfig(return_embeddings=True, return_hidden_states=return_hidden_states)

            if self.device.type == "cuda" and self._amp_dtype() is not None:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype()):
                    pt = self.model.encode(protein).to(self.device)
                    out = self.model.logits(pt, cfg)
            else:
                pt = self.model.encode(protein).to(self.device)
                out = self.model.logits(pt, cfg)
        except (TypeError, ValueError, RuntimeError) as e:
            self.__logger__.warning("Failed to embed sequence: %s", e)
            return None, None
        else:
            return out.embeddings, getattr(out, "hidden_states", None)

    def _process_sequence(
        self, seq: str, layers: LayerSpec, layer_agg: LayerAgg, pool: Pool,
    ) -> np.ndarray | None:
        """Embed and pool a single sequence."""
        emb, hs = self._embed_one(seq, return_hidden_states=True)

        if self._has_hidden_states(hs):
            hs_seq = cast("Sequence[torch.Tensor]", hs)
            select = self._parse_layers(layers, len(hs_seq))
            chosen = [hs_seq[j] for j in select]
            if layer_agg == "concat":
                stacked = torch.cat(chosen, dim=-1)
            elif layer_agg == "sum":
                stacked = torch.stack(chosen, dim=0).sum(dim=0)
            else:
                stacked = torch.stack(chosen, dim=0).mean(dim=0)
            dummy_attn = torch.ones(stacked.shape[:2], dtype=stacked.dtype, device=stacked.device)
            pooled = self._pool_tokens(stacked, dummy_attn, pool).squeeze(0)
        elif emb is not None:
            dummy_attn = torch.ones(emb.shape[:2], dtype=emb.dtype, device=emb.device)
            pooled = self._pool_tokens(emb, dummy_attn, pool).squeeze(0)
        else:
            return None

        # Ensure FP32 before NumPy conversion
        pooled = pooled.contiguous()
        if pooled.dtype in (torch.bfloat16, torch.float16):
            pooled = pooled.to(torch.float32)
        return pooled.detach().cpu().numpy()

    def _adjust_batch_size_on_oom(self, current_bs: int, i: int) -> int:
        new_bs = max(current_bs // 2, 1)
        self.__logger__.warning("OOM at idx %d. Reducing batch size %d → %d.", i, current_bs, new_bs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats()
        return new_bs

    def embedding_process(
        self,
        batch_size: int = 32,
        *,
        seq_len: int | None = None,
        layers: LayerSpec = "last",
        layer_agg: LayerAgg = "mean",
        pool: Pool = "mean",
    ) -> pd.DataFrame:
        """Embed all sequences with ESM-C and return pooled embeddings."""
        self.ensure_loaded()
        if self.dataset is None or self.column_seq not in self.dataset.columns:
            msg = f"Dataset invalid or column '{self.column_seq}' missing."
            raise ValueError(msg)

        sequences = self.dataset[self.column_seq].astype(str).tolist()
        if seq_len is not None:
            sequences = [s[:seq_len].ljust(seq_len, "X") for s in sequences]

        self.__logger__.info("Embedding %d sequences with ESM-C.", len(sequences))

        current_bs, out_vecs, i = max(int(batch_size), 1), [], 0
        while i < len(sequences):
            chunk = sequences[i : i + current_bs]
            try:
                for seq in tqdm(chunk, desc=f"[ESMC] idx {i}", leave=False):
                    vec = self._process_sequence(seq, layers, layer_agg, pool)
                    if vec is not None:
                        out_vecs.append(vec)
                i += current_bs
            except RuntimeError as e:
                is_oom = ("CUDA out of memory" in str(e)) or ("CUBLAS_STATUS_ALLOC_FAILED" in str(e))
                if self.oom_backoff and is_oom and current_bs > 1 and self.device.type == "cuda":
                    current_bs = self._adjust_batch_size_on_oom(current_bs, i)
                    continue
                raise

        if not out_vecs:
            msg = "No embeddings generated with ESM-C."
            raise RuntimeError(msg)

        self.release_resources()
        mat = np.stack(out_vecs, axis=0)
        cols = pd.Index([f"p_{k+1}" for k in range(mat.shape[1])])
        df_emb = pd.DataFrame(mat, columns=cols, index=self.dataset.index)
        df_emb[self.column_seq] = self.dataset[self.column_seq].to_numpy()
        return df_emb
