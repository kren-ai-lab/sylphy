from __future__ import annotations

import logging

import pandas as pd

from sylphy.constants import get_index, residues

from .base_encoder import Encoders


class OneHotEncoder(Encoders):
    """
    One-hot encode sequences; |alphabet|-dim per residue, zero-padded to `max_length`.
    Supports canonical or extended alphabet via base class flags.
    """

    def __init__(
        self,
        dataset: pd.DataFrame | None = None,
        sequence_column: str | None = "sequence",
        max_length: int = 1024,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
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

    def __vector_for_residue(self, residue: str) -> list[int]:
        v = [0] * self._A
        try:
            pos = get_index(
                residue,
                extended=(self.allow_extended or self.allow_unknown),
                allow_unknown=self.allow_unknown,
            )
            v[pos] = 1
        except Exception:
            # Unknown residue: keep zero vector
            pass
        return v

    def __zero_padding(self, current_length: int) -> list[int]:
        target = self.max_length * self._A
        return [0] * (target - current_length)

    def __encode_sequence(self, sequence: str) -> list[int]:
        coded: list[int] = []
        for r in sequence:
            coded.extend(self.__vector_for_residue(r))
        if len(sequence) < self.max_length:
            coded += self.__zero_padding(len(coded))
        return coded

    def run_process(self) -> None:
        if not self.status:
            self.__logger__.warning("Encoding skipped; dataset validation failed.")
            return

        try:
            self.__logger__.info("Starting one-hot encoding for %d sequences.", len(self.dataset))
            matrix = [
                self.__encode_sequence(self.dataset.at[i, self.sequence_column]) for i in self.dataset.index  # type: ignore[bad-argument-type]
            ]
            header = pd.Index([f"p_{i}" for i in range(len(matrix[0]))])
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values
            self.__logger__.info("One-hot encoding completed with %d features.", self.coded_dataset.shape[1])
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] One-hot encoding failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message) from e
