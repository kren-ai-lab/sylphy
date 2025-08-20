import numpy as np
import pandas as pd
import torch
from abc import ABC
from typing import List, Optional, Literal
from transformers import AutoTokenizer, AutoModel, AutoConfig

from protein_representation.logging.logging_config import setup_logger
from protein_representation.core.model_registry import ModelSpec, resolve_model, register_model
from protein_representation.core.config import ToolConfig
from protein_representation.misc.utils_lib import UtilsLib

class EmbeddingBased(ABC):
    """
    Base class for embedding extraction from protein sequences using HF models.

    Key features:
    - Resolves model path via model registry (resolves -> auto-register if needed -> resolves).
    - Loads tokenizer/model from the resolved *local* directory (no global HF cache pollution).
    - Batch embedding with masked pooling ("mean", "cls", "max").
    - Optional metadata passthrough to the output DataFrame.
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
        name_logging: str = "bioclust.embedding",
        trust_remote_code: bool = False,      
        precision: Literal["fp32","fp16","bf16"] = "fp32",
        oom_backoff: bool = True, 
    ) -> None:

        self.dataset = dataset
        self.column_seq = column_seq

        self.name_device = name_device
        self.device = torch.device(self.name_device)

        self.name_model = name_model.strip()
        self.name_tokenizer = (name_tokenizer.strip() or self.name_model)
        self.provider = provider
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.precision = precision
        self.oom_backoff = oom_backoff

        self.__logger__ = setup_logger(
            name=name_logging,
            level=debug_mode,
            enable=debug
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
        Try to resolve the model via registry. If missing and provider is HF
        and name_model looks like 'org/model', register it, then resolve.
        Returns a *local directory path* to the model.
        """
        # try to resolve as given (could be a registry key already)
        try:
            local_dir = resolve_model(self.name_model)
            return str(local_dir)
        except KeyError:
            self.__logger__.info(f"Model '{self.name_model}' not found in registry; will try to auto-register…")

        # if not found, and looks like HF ref ("org/model"), register it
        if self.provider.lower() == "huggingface" and "/" in self.name_model:
            org, model = self.name_model.split("/", 1)
            # canonical key for the registry (you can choose other naming policies)
            canonical_name = model  # e.g., esm2_t6_8M_UR50D
            try:
                register_model(ModelSpec(
                    name=canonical_name,
                    provider="huggingface",
                    ref=f"{org}/{model}",
                    revision=self.revision
                ))
                self.__logger__.info(f"Registered HF model '{canonical_name}' → {org}/{model}")
                # Now resolve by the *canonical* name
                local_dir = resolve_model(canonical_name)
                # Keep both around: original user string and canonical key.
                # Prefer loading from the resolved *local path*.
                return str(local_dir)
            except Exception as e:
                self.status = False
                self.message = f"Failed to auto-register '{self.name_model}': {e}"
                self.__logger__.error(self.message)
                raise
        else:
            # If provider is 'other', you'd need a URL or local path in the registry.
            # We don't auto-register unknown 'other' refs to avoid surprises.
            msg = (
                f"Unknown model '{self.name_model}' and auto-registration is only supported "
                f"for Hugging Face refs shaped as 'org/model'. Provide a registered name or a valid HF ref."
            )
            self.status = False
            self.message = msg
            self.__logger__.error(msg)
            raise KeyError(msg)

    def load_model_tokenizer(self) -> None:
        """
        Resolve and load tokenizer/model from the local directory managed by the registry.
        """
        try:
            local_dir = self._register_and_resolve()

            # Load config first (optional, but useful to inspect hidden_size, etc.)
            _ = AutoConfig.from_pretrained(local_dir, trust_remote_code=self.trust_remote_code)

            self.__logger__.info(f"Loading tokenizer from: {local_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                do_lower_case=False,
                use_fast=True,
                trust_remote_code=self.trust_remote_code
            )

            self.__logger__.info(f"Loading model from: {local_dir} on device={self.device}")
            self.model = AutoModel.from_pretrained(
                local_dir,
                trust_remote_code=self.trust_remote_code
            ).to(self.device)
            self.model.eval()

        except Exception as e:
            self.status = False
            self.message = f"Failed to load tokenizer/model: {e}"
            self.__logger__.error(self.message)
            raise

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    def validate_input(self) -> None:
        if self.dataset is None:
            msg = "Dataset is not loaded."
            self.__logger__.error(msg)
            raise ValueError(msg)

        if self.column_seq not in self.dataset.columns:
            msg = f"Column '{self.column_seq}' not found in dataset."
            self.__logger__.error(msg)
            raise ValueError(msg)

        if self.tokenizer is None or self.model is None:
            msg = "Tokenizer or model has not been loaded. Call `load_model_tokenizer()`."
            self.__logger__.error(msg)
            raise RuntimeError(msg)

    # ---------------------------------------------------------------------
    # Embedding core
    # ---------------------------------------------------------------------

    def _amp_dtype(self):
        if self.precision == "fp16":
            return torch.float16
        if self.precision == "bf16":
            return torch.bfloat16
        return None
    
    @torch.no_grad()
    def embedding_batch(
        self,
        batch: List[str],
        max_length: int = 1024
    ) -> torch.Tensor:
        try:
            enc = self.tokenizer(
                batch,
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
                self.__logger__.debug(f"Using autocast dtype={amp_dtype} for batch forward.")
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.model(**enc, output_hidden_states=False)
            else:
                out = self.model(**enc, output_hidden_states=False)

            return out.last_hidden_state, enc.get("attention_mask", None)
        except Exception as e:
            msg = f"Error during embedding batch: {e}"
            self.__logger__.error(msg)
            raise RuntimeError(msg)


    def _pool_masked_mean(self, hidden: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> np.ndarray:
        """
        Mean over sequence length ignoring padding (uses attention_mask).
        hidden: (B, L, H), attn_mask: (B, L)
        """
        if attn_mask is None:
            return hidden.mean(dim=1).cpu().numpy()
        mask = attn_mask.unsqueeze(-1)  # (B, L, 1)
        summed = (hidden * mask).sum(dim=1)           # (B, H)
        counts = mask.sum(dim=1).clamp(min=1)         # (B, 1)
        return (summed / counts).cpu().numpy()

    def pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        strategy: Literal["mean", "cls", "max"] = "mean"
    ) -> np.ndarray:
        """
        Pools hidden states into a single embedding per sequence.
        """
        try:
            if strategy == "mean":
                return self._pool_masked_mean(hidden_states, attention_mask)
            elif strategy == "cls":
                # Some protein models don't use [CLS] semantically; mean is often better.
                return hidden_states[:, 0, :].cpu().numpy()
            elif strategy == "max":
                if attention_mask is not None:
                    # Set pads to very negative before max
                    neg_inf = torch.finfo(hidden_states.dtype).min
                    mask = attention_mask.unsqueeze(-1)
                    masked = hidden_states.masked_fill(mask == 0, neg_inf)
                    return masked.max(dim=1).values.cpu().numpy()
                return hidden_states.max(dim=1).values.cpu().numpy()
            else:
                raise ValueError(f"Invalid pooling strategy '{strategy}'.")
        except Exception as e:
            self.__logger__.error(f"Error during pooling: {e}")
            raise

    def embedding_process(
        self,
        batch_size: int = 32,
        pooling_strategy: Literal["mean", "cls", "max"] = "mean",
        max_length: int = 1024
    ) -> pd.DataFrame:
        self.validate_input()

        sequences = self.dataset[self.column_seq].tolist()
        all_embeddings: List[np.ndarray] = []

        current_bs = max(1, batch_size)
        i = 0
        self.__logger__.info(
            f"Embedding {len(sequences)} seqs with model='{self.name_model}' "
            f"on {self.device} (precision={self.precision}, OOM backoff={self.oom_backoff})."
        )

        while i < len(sequences):
            try:
                batch = sequences[i : i + current_bs]
                last_hidden, attn = self.embedding_batch(batch=batch, max_length=max_length)
                pooled = self.pooling(last_hidden, attn, strategy=pooling_strategy)
                all_embeddings.append(pooled)
                i += current_bs 
            except RuntimeError as e:
                is_oom = ("CUDA out of memory" in str(e)) or ("CUBLAS_STATUS_ALLOC_FAILED" in str(e))
                if not (self.oom_backoff and is_oom and current_bs > 1 and self.device.type == "cuda"):
                    raise
                # Backoff
                new_bs = max(1, current_bs // 2)
                self.__logger__.warning(
                    f"OOM detected at batch starting idx {i}. "
                    f"Reducing batch size {current_bs} → {new_bs} and retrying."
                )
                current_bs = new_bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                continue

        mat = np.concatenate(all_embeddings, axis=0)
        headers = [f"p_{i+1}" for i in range(mat.shape[1])]
        df_embedding = pd.DataFrame(mat, columns=headers)

        self.__logger__.info("Embedding process completed.")
        return df_embedding


    # ---------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------

    def export_embeddings(
        self,
        df_embedding: pd.DataFrame,
        path: str,
        file_format: Literal["csv", "npy"] = "csv"
    ) -> None:
        """
        Save the embedding matrix to disk.
        """
        UtilsLib.export_data(
            df_encoded=df_embedding,
            path=path,
            __logger__= self.__logger__,
            base_message= "Embedding extracted ",
            file_format= file_format
        )

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def cleaning_memory(self) -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            self.__logger__.info("Clearing CUDA memory cache.")
            torch.cuda.empty_cache()
