# sylphy/embedding_extraction/esmc_based.py
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm

from sylphy.types import PrecisionType

from .embedding_based import EmbeddingBased, LayerAgg, LayerSpec, Pool  # types & semantics


class ESMCBasedEmbedding(EmbeddingBased):
    """
    ESM-C backend using Meta's ESM SDK.

    Notes
    -----
    - ESMC exposes embeddings and (optionally) hidden states via LogitsConfig.
    - We provide per-sequence embedding with optional layer selection/aggregation.
    - No HuggingFace tokenizer is required for ESM-C.
    """

    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "esmc_300m",
        name_tokenizer: str | None = None,  # ignored for ESM-C
        dataset: pd.DataFrame | None = None,
        column_seq: str | None = "sequence",
        debug: bool = False,
        debug_mode: int = logging.INFO,
        precision: PrecisionType = "fp32",
        oom_backoff: bool = True,
    ) -> None:
        if dataset is None:
            raise ValueError("dataset must be provided")

        super().__init__(
            dataset=dataset,
            name_device=name_device,
            name_model=name_model,
            name_tokenizer="",  # force empty: ESM-C doesn't use tokenizer
            provider="other",
            revision=None,
            column_seq=column_seq or "sequence",
            debug=debug,
            debug_mode=debug_mode,
            name_logging=ESMCBasedEmbedding.__name__,
            trust_remote_code=False,
            precision=precision,
            oom_backoff=oom_backoff,
        )
        self.requires_tokenizer = False
        self._embedding_dim: int | None = None
        self.model: ESMC | None = None  # explicit type for clarity

    # -------- utilities --------
    def _has_hidden_states(self, hs) -> bool:
        if hs is None:
            return False
        if isinstance(hs, (list, tuple)):
            return len(hs) > 0
        if torch.is_tensor(hs):
            return hs.numel() > 0
        return False

    def ensure_loaded(self) -> None:
        """Idempotent loader. Safe to call many times."""
        if getattr(self, "model", None) is None:
            self.load_model_tokenizer()

    def load_model_tokenizer(self) -> None:
        try:
            # Try registry resolution; if not found, fall back to from_pretrained(name_model)
            local_dir: str | None = None
            try:
                local_dir = self._register_and_resolve()
            except Exception:
                local_dir = None

            load_ref = local_dir if local_dir else self.name_model
            self.__logger__.info("Loading ESM-C from: %s on device=%s", load_ref, self.device)
            mdl = ESMC.from_pretrained(load_ref)
            mdl.to(self.device)  # move to device
            mdl.eval()
            self.model = mdl
            self.status = True
            self.message = "ESM-C model loaded."
            self.__logger__.info("ESM-C model '%s' loaded successfully.", load_ref)
        except Exception as e:
            self.status = False
            self.message = f"Failed to load ESM-C model: {e}"
            self.__logger__.error(self.message)
            raise

    @torch.no_grad()
    def _embed_one(
        self,
        sequence: str,
        *,
        return_hidden_states: bool,
    ) -> tuple[torch.Tensor | None, Sequence[torch.Tensor] | None]:
        """
        Encode one sequence. Returns:
          - embeddings: (1, L, H) or None
          - hidden_states: list of (1, L, H) or None
        """
        try:
            self.ensure_loaded()
            assert self.model is not None, "ESM-C model not loaded."

            protein = ESMProtein(sequence=sequence)

            cfg = LogitsConfig(
                return_embeddings=True,
                return_hidden_states=return_hidden_states,
            )

            if self.device.type == "cuda" and self._amp_dtype() is not None:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype()):
                    pt = self.model.encode(protein)  # cpu tensor
                    pt = pt.to(self.device)
                    out = self.model.logits(pt, cfg)
            else:
                pt = self.model.encode(protein)
                pt = pt.to(self.device)
                out = self.model.logits(pt, cfg)

            emb = out.embeddings  # (1, L, H) or None
            hs = getattr(out, "hidden_states", None)
            return emb, hs
        except Exception as e:
            self.__logger__.warning("Failed to embed one sequence: %s", e)
            return None, None

    def embedding_process(
        self,
        batch_size: int = 32,
        *,
        seq_len: int | None = None,
        layers: LayerSpec = "last",
        layer_agg: LayerAgg = "mean",
        pool: Pool = "mean",
    ) -> pd.DataFrame:
        """
        Embed all sequences with ESM-C and return pooled embeddings per sequence.

        Parameters
        ----------
        seq_len : int, optional
            If provided, sequences are padded/truncated to this length *before* encoding.
        layers : LayerSpec
            "last" | "last4" | "all" | int | [ints]
        layer_agg : {"mean","sum","concat"}
            Aggregation across selected layers (ignored if hidden states unavailable).
        pool : {"mean","cls","eos"}
            Token pooling strategy.
        """
        # Ensure model is loaded (idempotent)
        self.ensure_loaded()

        if self.dataset is None:
            raise ValueError("Dataset is not loaded.")
        if self.column_seq not in self.dataset.columns:
            raise ValueError(f"Column '{self.column_seq}' not found in dataset.")

        sequences: list[str] = self.dataset[self.column_seq].astype(str).tolist()
        if seq_len is not None:
            sequences = [s[:seq_len].ljust(seq_len, "X") for s in sequences]

        self.__logger__.info(
            "Embedding %d sequences with ESM-C (device=%s, precision=%s, OOM backoff=%s).",
            len(sequences),
            self.device,
            self.precision,
            self.oom_backoff,
        )

        current_bs: int = int(batch_size)
        if current_bs < 1:
            current_bs = 1
        out_vecs: list[np.ndarray] = []
        i: int = 0

        while i < len(sequences):
            chunk = sequences[i : i + current_bs]
            try:
                # ESM-C SDK is per-sequence; iterate inside the chunk
                for seq in tqdm(chunk, desc=f"[ESMC] idx {i}", leave=False):
                    emb, hs = self._embed_one(seq, return_hidden_states=True)

                    if self._has_hidden_states(hs):  # hidden states available
                        assert hs is not None
                        n_layers = len(hs)
                        select = EmbeddingBased._parse_layers(layers, n_layers)
                        chosen = [hs[j] for j in select]  # each (1,L,H)
                        if layer_agg == "concat":
                            stacked = torch.cat(chosen, dim=-1)
                        elif layer_agg == "sum":
                            stacked = torch.stack(chosen, dim=0).sum(dim=0)
                        else:  # mean (default)
                            stacked = torch.stack(chosen, dim=0).mean(dim=0)
                        dummy_attn = torch.ones(stacked.shape[:2], dtype=stacked.dtype, device=stacked.device)
                        pooled = self._pool_tokens(stacked, dummy_attn, pool).squeeze(0)  # (H')
                    elif emb is not None:
                        dummy_attn = torch.ones(emb.shape[:2], dtype=emb.dtype, device=emb.device)
                        pooled = self._pool_tokens(emb, dummy_attn, pool).squeeze(0)  # (H)
                    else:
                        # Skip if neither emb nor hs are available
                        continue

                    # --- ensure FP32 before NumPy conversion ---
                    pooled = pooled.contiguous()
                    if pooled.dtype in (torch.bfloat16, torch.float16):
                        pooled = pooled.to(torch.float32)
                    vec = pooled.detach().cpu().numpy()

                    if self._embedding_dim is None:
                        self._embedding_dim = int(vec.shape[-1])
                    out_vecs.append(vec)

                i += current_bs

            except RuntimeError as e:
                is_oom = ("CUDA out of memory" in str(e)) or ("CUBLAS_STATUS_ALLOC_FAILED" in str(e))
                if not (self.oom_backoff and is_oom and current_bs > 1 and self.device.type == "cuda"):
                    raise
                new_bs = current_bs // 2
                if new_bs < 1:
                    new_bs = 1
                self.__logger__.warning("OOM at idx %d. Reducing batch size %d → %d.", i, current_bs, new_bs)
                current_bs = new_bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                continue

        if not out_vecs:
            raise RuntimeError("No embeddings generated with ESM-C.")

        # Release GPU memory
        self.release_resources()

        mat = np.stack(out_vecs, axis=0)  # (N, H')
        headers = [f"p_{k + 1}" for k in range(mat.shape[1])]
        df_emb = pd.DataFrame(mat, columns=pd.Index(headers), index=self.dataset.index)
        df_emb[self.column_seq] = self.dataset[self.column_seq].values
        self.__logger__.info("ESM-C embedding completed. Shape: %s", df_emb.shape)
        return df_emb
