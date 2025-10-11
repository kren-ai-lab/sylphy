# sylphy/embedding_extraction/esmc_based.py
from __future__ import annotations

import logging
from typing import Optional, List, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from .embedding_based import LayerSpec, LayerAgg, Pool  # types & semantics
from .embedding_based import EmbeddingBased


class ESMCBasedEmbedding(EmbeddingBased):
    """
    ESM-C backend using Meta's ESM SDK.

    Notes
    -----
    - ESMC exposes embeddings and (optionally) hidden states via LogitsConfig.
    - We provide per-sequence embedding with optional layer selection/aggregation.
    """

    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "esmc_300m",
        name_tokenizer: Optional[str] = None,
        dataset: Optional[pd.DataFrame] = None,
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
            name_tokenizer=name_tokenizer or "",
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
        self._embedding_dim: Optional[int] = None

    def _has_hidden_states(self, hs) -> bool:
        if hs is None:
            return False
        if isinstance(hs, (list, tuple)):
            return len(hs) > 0
        if torch.is_tensor(hs):
            return hs.numel() > 0
        return False

    def load_model_tokenizer(self) -> None:
        try:
            # Try registry resolution; if not found, fall back to from_pretrained(name_model)
            local_dir: Optional[str] = None
            try:
                local_dir = self._register_and_resolve()
            except Exception:
                local_dir = None

            load_ref = local_dir if local_dir else self.name_model
            self.__logger__.info("Loading ESM-C from: %s on device=%s", load_ref, self.device)
            self.model = ESMC.from_pretrained(load_ref).to(self.device)
            self.model.eval()
            self.__logger__.info("ESM-C model '%s' loaded successfully.", load_ref)
        except Exception as e:
            self.status = False
            self.message = f"Failed to load ESM-C model: {e}"
            self.__logger__.error(self.message)
            raise

    @staticmethod
    def _pool_tokens(x: torch.Tensor, pool: Pool) -> torch.Tensor:
        # ESMC outputs do not provide attention masks in the same way;
        # we assume full length (padded/truncated upstream if needed).
        if pool == "mean":
            return x.mean(dim=1)
        if pool == "cls":
            return x[:, 0, :]
        if pool == "eos":
            return x[:, -1, :]
        raise ValueError(f"Unknown token pool '{pool}'")

    @torch.no_grad()
    def _embed_one(
        self,
        sequence: str,
        *,
        return_hidden_states: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[Sequence[torch.Tensor]]]:
        """
        Encode one sequence. Returns:
          - embeddings: (1, L, H) or None
          - hidden_states: list of (1, L, H) or None
        """
        try:
            protein = ESMProtein(sequence=sequence)
            if self.device.type == "cuda" and self._amp_dtype() is not None:
                with torch.autocast(device_type="cuda", dtype=self._amp_dtype()):
                    pt = self.model.encode(protein).to(self.device)
                    cfg = LogitsConfig(return_embeddings=True, return_hidden_states=return_hidden_states)
                    out = self.model.logits(pt, cfg)
            else:
                pt = self.model.encode(protein).to(self.device)
                cfg = LogitsConfig(return_embeddings=True, return_hidden_states=return_hidden_states)
                out = self.model.logits(pt, cfg)

            emb = out.embeddings  # (1, L, H) or None
            hs = out.hidden_states if hasattr(out, "hidden_states") else None
            return emb, hs
        except Exception as e:
            self.__logger__.warning("Failed to embed one sequence: %s", e)
            return None, None

    def embedding_process(
        self,
        batch_size: int = 32,
        *,
        seq_len: Optional[int] = None,
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
        self.load_model_tokenizer()

        if self.dataset is None:
            raise ValueError("Dataset is not loaded.")
        if self.column_seq not in self.dataset.columns:
            raise ValueError(f"Column '{self.column_seq}' not found in dataset.")

        sequences: List[str] = self.dataset[self.column_seq].astype(str).tolist()
        if seq_len is not None:
            sequences = [s[:seq_len].ljust(seq_len, "X") for s in sequences]

        self.__logger__.info(
            "Embedding %d sequences with ESM-C (device=%s, precision=%s, OOM backoff=%s).",
            len(sequences), self.device, self.precision, self.oom_backoff
        )

        current_bs = max(1, int(batch_size))
        out_vecs: List[np.ndarray] = []
        i = 0

        while i < len(sequences):
            chunk = sequences[i : i + current_bs]
            try:
                # ESMC API is per-sequence; we iterate inside the chunk
                for seq in tqdm(chunk, desc=f"[ESMC] idx {i}", leave=False):
                    emb, hs = self._embed_one(seq, return_hidden_states=True)

                    if self._has_hidden_states(hs):  # hidden states available
                        n_layers = len(hs)
                        # Select and aggregate layers
                        select = EmbeddingBased._parse_layers(layers, n_layers)
                        chosen = [hs[j] for j in select]  # each (1,L,H)
                        if layer_agg == "concat":
                            stacked = torch.cat(chosen, dim=-1)
                        elif layer_agg == "sum":
                            stacked = torch.stack(chosen, dim=0).sum(dim=0)
                        else:
                            stacked = torch.stack(chosen, dim=0).mean(dim=0)
                        pooled = self._pool_tokens(stacked, pool=pool).squeeze(0)  # (H')
                    elif emb is not None:
                        pooled = self._pool_tokens(emb, pool=pool).squeeze(0)      # (H)
                    else:
                        continue

                    # --- FIX: NumPy does not support torch.bfloat16/float16 ---
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
                new_bs = max(1, current_bs // 2)
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

        mat = np.stack(out_vecs, axis=0)  # (N, H')
        headers = [f"p_{k+1}" for k in range(mat.shape[1])]
        df_emb = pd.DataFrame(mat, columns=headers, index=self.dataset.index)
        df_emb[self.column_seq] = self.dataset[self.column_seq].values
        self.__logger__.info("ESM-C embedding completed. Shape: %s", df_emb.shape)
        return df_emb
