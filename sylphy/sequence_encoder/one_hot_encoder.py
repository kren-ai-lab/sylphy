"""Implement one-hot encoding for protein sequences."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from sylphy.constants import get_index, residues

from .encoder_base import EncoderBase


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
        self._A = len(self._alpha)

    def _encode_sequence(self, sequence: str) -> list[int]:
        coded: list[int] = []
        for r in sequence:
            v = [0] * self._A
            try:
                pos = get_index(
                    r,
                    extended=(self.allow_extended or self.allow_unknown),
                    allow_unknown=self.allow_unknown,
                )
                v[pos] = 1
            except KeyError:
                self.__logger__.debug("Unknown residue '%s' encoded as zero vector.", r)
            coded.extend(v)
        coded += [0] * (self.max_length * self._A - len(coded))
        return coded

    def run_process(self) -> None:
        """Encode all validated sequences using one-hot representation."""
        try:
            self.__logger__.info("Starting one-hot encoding for %d sequences.", len(self.dataset))
            sequences = self.dataset[self.sequence_column].to_list()
            matrix = np.array([self._encode_sequence(seq) for seq in sequences], dtype=np.uint8)
            col_names = [f"p_{i}" for i in range(matrix.shape[1])]
            self.coded_dataset = pl.from_numpy(matrix, schema=col_names).with_columns(
                pl.Series(self.sequence_column, sequences)
            )
            self.__logger__.info("One-hot encoding completed with %d features.", self.coded_dataset.width)
        except Exception as e:
            msg = f"[ERROR] One-hot encoding failed: {e}"
            self.__logger__.exception(msg)
            raise RuntimeError(msg) from e
