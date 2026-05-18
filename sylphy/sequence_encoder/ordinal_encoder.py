"""Implement ordinal residue encoding for protein sequences."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from sylphy.constants import get_index, residues

from .encoder_base import EncoderBase


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

    def _encode_sequence(self, sequence: str) -> list[int]:
        coded: list[int] = []
        for r in sequence:
            try:
                coded.append(
                    get_index(
                        r,
                        extended=(self.allow_extended or self.allow_unknown),
                        allow_unknown=self.allow_unknown,
                    ),
                )
            except KeyError:
                coded.append(0)
        coded += [0] * (self.max_length - len(coded))
        return coded

    def run_process(self) -> None:
        """Encode all validated sequences using ordinal representation."""
        try:
            self.__logger__.info("Starting ordinal encoding for %d sequences.", len(self.dataset))
            sequences = self.dataset[self.sequence_column].to_list()
            matrix = np.array([self._encode_sequence(seq) for seq in sequences], dtype=np.int32)
            col_names = [f"p_{i}" for i in range(matrix.shape[1])]
            self.coded_dataset = pl.from_numpy(matrix, schema=col_names).with_columns(
                pl.Series(self.sequence_column, sequences)
            )
            self.__logger__.info("Ordinal encoding completed with %d features.", self.coded_dataset.width)
        except Exception as e:
            msg = f"[ERROR] Ordinal encoding failed: {e}"
            self.__logger__.exception(msg)
            raise RuntimeError(msg) from e
