from __future__ import annotations

import logging
from typing import Optional, List

import pandas as pd

from .base_encoder import Encoders
from sylphy.constants import get_index, residues


class OrdinalEncoder(Encoders):
    """
    Ordinally encode residues as their index in the selected amino-acid alphabet,
    zero-padding up to `max_length`.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = "sequence",
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
            name_logging=OrdinalEncoder.__name__,
        )
        self._alpha = residues(extended=self.allow_extended or self.allow_unknown)

    def __zero_padding(self, current_length: int) -> List[int]:
        return [0] * (self.max_length - current_length)

    def __encode_sequence(self, sequence: str) -> List[int]:
        coded: List[int] = []
        for r in sequence:
            try:
                coded.append(
                    get_index(r, extended=(self.allow_extended or self.allow_unknown),
                              allow_unknown=self.allow_unknown)
                )
            except Exception:
                coded.append(0)
        if len(sequence) < self.max_length:
            coded += self.__zero_padding(len(coded))
        return coded

    def run_process(self) -> None:
        if not self.status:
            self.__logger__.warning("Encoding skipped; dataset validation failed.")
            return

        try:
            self.__logger__.info("Starting ordinal encoding for %d sequences.", len(self.dataset))
            matrix = [self.__encode_sequence(self.dataset.at[i, self.sequence_column]) for i in self.dataset.index]
            header = [f"p_{i}" for i in range(len(matrix[0]))]
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values
            self.__logger__.info("Ordinal encoding completed with %d features.", self.coded_dataset.shape[1])
        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Ordinal encoding failed: {e}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
