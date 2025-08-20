import pandas as pd
from typing import Optional, List

from .base_encoder import Encoders
from bioclust.misc.constants import Constant
import logging

class OrdinalEncoder(Encoders):
    """
    Encodes amino acid sequences using ordinal encoding.

    Each residue is mapped to its corresponding index in the amino acid alphabet.
    Sequences shorter than `max_length` are zero-padded.

    Inherits from
    ----------
    Encoders : Provides validation and filtering of input sequences.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing amino acid sequences.
    sequence_column : str, optional
        Column name that contains the sequences.
    max_length : int, default=1024
        Maximum sequence length. Shorter sequences will be padded with zeros.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = "sequence",
        max_length: int = 1024,
        debug:bool=False,
        debug_mode:int=logging.INFO
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column,
            max_length=max_length,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=OrdinalEncoder.__name__
        )

    def __zero_padding(self, current_length: int) -> List[int]:
        """
        Generates zero-padding to reach the expected encoded vector length.

        Parameters
        ----------
        current_length : int
            Current length of the encoded sequence vector.

        Returns
        -------
        List[int]
            Padding list with zeros.
        """
        return [0] * (self.max_length - current_length)

    def __coding_sequence(self, sequence: str) -> List[int]:
        """
        Encodes a sequence into ordinal indices based on residue position.

        Parameters
        ----------
        sequence : str
            Amino acid sequence to encode.

        Returns
        -------
        List[int]
            Ordinal-encoded and padded sequence vector.
        """
        coded_vector = []
        for residue in sequence:
            try:
                coded_vector.append(Constant.POSITION_RESIDUES[residue])
            except KeyError:
                self.__logger__.warning("Residue '%s' not recognized. Skipping residue.", residue)
                coded_vector.append(0)

        if len(sequence) < self.max_length:
            coded_vector += self.__zero_padding(len(coded_vector))

        return coded_vector

    def run_process(self) -> None:
        """
        Performs ordinal encoding of all sequences in the dataset.

        The resulting encoded vectors are stored in `self.coded_dataset`.

        Raises
        ------
        RuntimeError
            If the encoding fails due to unexpected errors.
        """
        if not self.status:
            self.__logger__.warning("Encoding was not performed because the dataset is marked invalid.")
            return

        try:
            self.__logger__.info("Starting ordinal encoding for %d sequences.", len(self.dataset))

            matrix_coded = [
                self.__coding_sequence(self.dataset.at[index, self.sequence_column])
                for index in self.dataset.index
            ]

            header = [f"p_{i}" for i in range(len(matrix_coded[0]))]
            self.coded_dataset = pd.DataFrame(data=matrix_coded, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values

            self.__logger__.info("Ordinal encoding completed successfully with %d features.",
                            self.coded_dataset.shape[1])

        except Exception as e:
            self.status = False
            self.message = f"[ERROR] Ordinal encoding failed: {str(e)}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
