"""Implement ordinal residue encoding for protein sequences."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sylphy.constants import residues

if TYPE_CHECKING:
    import polars as pl

from .encoder_base import EncoderBase, encode_ordinal


class OrdinalEncoder(EncoderBase):
    """Encode residues as alphabet indices padded to ``max_length``."""

    def __init__(
        self,
        dataset: pl.DataFrame | None = None,
        sequence_column: str | None = "sequence",
        max_length: int = 1024,
        *,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        """Initialize the ordinal encoder."""
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column or "sequence",
            max_length=max_length,
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=OrdinalEncoder.__name__,
        )
        self._alpha = residues(extended=self.allow_extended or self.allow_unknown)

    def run_process(self) -> None:
        """Encode all validated sequences using ordinal representation."""
        try:
            self.__logger__.info("Starting ordinal encoding for %d sequences.", len(self.dataset))
            sequences = self.dataset[self.sequence_column].to_list()
            matrix = encode_ordinal(sequences, self._alpha, self.max_length)
            self._finalize_encoding(sequences, matrix)
            self.__logger__.info("Ordinal encoding completed with %d features.", self.coded_dataset.width)
        except Exception as e:
            msg = f"[ERROR] Ordinal encoding failed: {e}"
            self.__logger__.exception(msg)
            raise RuntimeError(msg) from e
