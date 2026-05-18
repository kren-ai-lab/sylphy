"""Implement one-hot encoding for protein sequences."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sylphy.constants import residues

if TYPE_CHECKING:
    import polars as pl

from .encoder_base import EncoderBase, encode_onehot


class OneHotEncoder(EncoderBase):
    """Encode sequences as one-hot residue vectors padded to ``max_length``.

    Supports canonical or extended alphabets configured by base-class flags.
    """

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
        """Initialize the one-hot encoder."""
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column or "sequence",
            max_length=max_length,
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=OneHotEncoder.__name__,
        )
        self._alpha = residues(extended=self.allow_extended or self.allow_unknown)

    def run_process(self) -> None:
        """Encode all validated sequences using one-hot representation."""
        try:
            self.__logger__.info("Starting one-hot encoding for %d sequences.", len(self.dataset))
            sequences = self.dataset[self.sequence_column].to_list()
            matrix = encode_onehot(sequences, self._alpha, self.max_length)
            self._finalize_encoding(sequences, matrix)
            self.__logger__.info("One-hot encoding completed with %d features.", self.coded_dataset.width)
        except Exception as e:
            msg = f"[ERROR] One-hot encoding failed: {e}"
            self.__logger__.exception(msg)
            raise RuntimeError(msg) from e
