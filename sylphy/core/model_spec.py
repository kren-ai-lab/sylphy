# core/model_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Provider = Literal["huggingface", "other"]


@dataclass(frozen=True)
class ModelSpec:
    """
    Canonical specification for a model entry in the registry.

    Parameters
    ----------
    name : str
        Canonical short name (e.g., "prot_t5_xl_uniref50"). Must be unique within the registry.
    provider : {"huggingface", "other"}
        Provider identifier. Use "huggingface" for HF Hub models and "other" for local paths/URLs.
    ref : str
        Provider-specific reference. For Hugging Face this is "org/model". For "other", a URL or local path.
    subdir : Optional[str], default=None
        Optional subdirectory under the resolved local directory (useful for nested snapshots).
    revision : Optional[str], default=None
        For HF: branch, tag, or commit SHA to pin.
    alias_of : Optional[str], default=None
        If this spec is an alias, points to the canonical model name.

    Notes
    -----
    - Minimal and immutable to keep the registry deterministic and easy to serialize.
    """

    name: str
    provider: Provider
    ref: str
    subdir: Optional[str] = None
    revision: Optional[str] = None
    alias_of: Optional[str] = None

    def is_alias(self) -> bool:
        """Return True if this spec is an alias entry."""
        return self.alias_of is not None
