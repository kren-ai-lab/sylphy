import pandas as pd
import numpy as np
import torch
from typing import Optional, List
from tqdm import tqdm
from protein_representation.core.config import ToolConfig

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from .embedding_based import EmbeddingBased

class ESMCBasedEmbedding(EmbeddingBased):

    def __init__(
        self,
        name_device: str = "cuda" if torch.cuda.is_available() else "cpu",
        name_model: str = "esmc_300m",             
        name_tokenizer: Optional[str] = None,      
        dataset: Optional[pd.DataFrame] = None,
        column_seq: Optional[str] = "sequence",
        debug: bool = True,
        debug_mode: int = ToolConfig.log_level,
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

    def load_model_tokenizer(self) -> None:
        """
        Try to resolve the model via the registry (provider='other') if the user has
        registered a local path/URL. Otherwise, fallback to ESMC.from_pretrained(name_model).
        """
        try:
            # Try registry resolve (will fail if not registered)
            local_dir: Optional[str] = None
            try:
                # _register_and_resolve is tailored for HF auto-registration,
                # so we do a guarded call here. If not found, we just ignore and fallback.
                local_dir = self._register_and_resolve()
            except Exception:
                local_dir = None

            load_ref = local_dir if local_dir else self.name_model

            self.__logger__.info(f"Loading ESM-C from: {load_ref} on device={self.device}")
            self.model = ESMC.from_pretrained(load_ref).to(self.device)
            self.model.eval()
            self.__logger__.info(f"ESM-C model '{load_ref}' loaded successfully.")

        except Exception as e:
            self.status = False
            self.message = f"Failed to load ESM-C model: {e}"
            self.__logger__.error(self.message)
            raise


    @staticmethod
    def pad_sequence(sequence: str, target_length: int, pad_char: str = "X") -> str:
        """
        Pads or truncates a protein sequence to a fixed length for uniform tensors.
        """
        s = (sequence or "")
        return s[:target_length].ljust(target_length, pad_char)


    def _amp_dtype(self):
        # Reuse the same helper idea as the base class (kept here to keep this class isolated)
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "bf16":
            return torch.bfloat16
        return None


    @torch.no_grad()
    def _embed_one(self, sequence: str, amp_dtype: Optional[torch.dtype]) -> Optional[np.ndarray]:
        """
        Embed a single sequence with ESM-C → mean over length → (H,)
        """
        try:
            protein = ESMProtein(sequence=sequence)
            # ESM-C: encode then logits call
            if self.device.type == "cuda" and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    protein_tensor = self.model.encode(protein).to(self.device)
                    cfg = LogitsConfig(return_embeddings=True, return_hidden_states=False)
                    output = self.model.logits(protein_tensor, cfg)
            else:
                protein_tensor = self.model.encode(protein).to(self.device)
                cfg = LogitsConfig(return_embeddings=True, return_hidden_states=False)
                output = self.model.logits(protein_tensor, cfg)

            # output.embeddings: (1, L, H)
            emb = output.embeddings
            if emb is None:
                return None
            pooled = emb.mean(dim=1).squeeze(0).detach().cpu().numpy()  # (H,)
            if self._embedding_dim is None:
                self._embedding_dim = pooled.shape[-1]
            return pooled
        except Exception as e:
            self.__logger__.warning(f"Failed to embed one sequence: {e}")
            return None

    def embedding_process(
        self,
        batch_size: int = 64,
        seq_len: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Embed all sequences with ESM-C (sequence-wise loop) and return pooled embeddings.

        Notes
        -----
        - Uses AMP (fp16/bf16) if precision set and CUDA available.
        - Auto-reduces batch size by half on CUDA OOM.
        - If seq_len is provided, sequences are padded/truncated to that length.
          Otherwise, uses max length in dataset (be careful with VRAM).
        """
        # Load model
        self.load_model_tokenizer()

        # Validate input
        if self.dataset is None:
            raise ValueError("Dataset is not loaded.")
        if self.column_seq not in self.dataset.columns:
            raise ValueError(f"Column '{self.column_seq}' not found in dataset.")

        sequences: List[str] = self.dataset[self.column_seq].astype(str).tolist()
        target_len = (max(len(s) for s in sequences) if seq_len is None else int(seq_len))
        padded = [self.pad_sequence(s, target_len) for s in sequences]

        self.__logger__.info(
            f"Embedding {len(padded)} sequences with ESM-C "
            f"(target_len={target_len}, device={self.device}, precision={self.precision}, OOM backoff={self.oom_backoff})."
        )

        amp_dtype = self._amp_dtype()
        current_bs = max(1, int(batch_size))
        all_vecs: List[np.ndarray] = []

        i = 0
        while i < len(padded):
            chunk = padded[i : i + current_bs]

            try:
                # ESM-C SDK is more sequence-oriented; we iterate inside the chunk
                vecs = []
                for seq in tqdm(chunk, desc=f"[ESMC] batch @ idx {i}", leave=False):
                    v = self._embed_one(seq, amp_dtype)
                    if v is not None:
                        vecs.append(v)

                if not vecs:
                    raise RuntimeError("No sequences embedded in current chunk.")

                all_vecs.extend(vecs)
                i += current_bs

            except RuntimeError as e:
                is_oom = ("CUDA out of memory" in str(e)) or ("CUBLAS_STATUS_ALLOC_FAILED" in str(e))
                if not (self.oom_backoff and is_oom and current_bs > 1 and self.device.type == "cuda"):
                    # Not an OOM we can handle → re-raise
                    raise
                # Backoff
                new_bs = max(1, current_bs // 2)
                self.__logger__.warning(
                    f"OOM detected near idx {i}. Reducing batch size {current_bs} → {new_bs} and retrying."
                )
                current_bs = new_bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                continue

        if not all_vecs:
            raise RuntimeError("No embeddings generated with ESM-C.")

        mat = np.stack(all_vecs, axis=0)  # (N, H)
        headers = [f"p_{k+1}" for k in range(mat.shape[1])]
        df_emb = pd.DataFrame(mat, columns=headers)

        self.__logger__.info("ESM-C embedding completed.")
        return df_emb
