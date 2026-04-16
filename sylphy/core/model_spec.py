"""Define the immutable model specification used by the registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Provider = Literal["huggingface", "other"]


@dataclass(frozen=True)
class ModelSpec:
    """Canonical specification for a model entry in the registry.

    Args:
        name: Canonical short model name (for example, ``prot_t5_xl_uniref50``).
        provider: Provider identifier (``huggingface`` or ``other``).
        ref: Provider-specific model reference.
        subdir: Optional subdirectory under the resolved local path.
        revision: Optional revision for providers that support pinning.
        alias_of: Canonical model name if this entry is an alias.

    """

    name: str
    provider: Provider
    ref: str
    subdir: str | None = None
    revision: str | None = None
    alias_of: str | None = None

    def is_alias(self) -> bool:
        """Return True if this spec is an alias entry."""
        return self.alias_of is not None
