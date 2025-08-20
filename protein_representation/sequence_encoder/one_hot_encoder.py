import pandas as pd
from typing import Optional, List
import logging

from .base_encoder import Encoders
from bioclust.misc.constants import Constant

class OneHotEncoder(Encoders):
    """
    Encodes amino acid sequences into one-hot encoded vectors.

    Each residue is converted into a 20-dimensional binary vector where only the position
    corresponding to the residue is set to 1. Sequences are padded with zeros up to
    `max_length` if they are shorter.

    Inherits from
    ----------
    Encoders : Performs validation and filtering of sequences.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing amino acid sequences.
    sequence_column : str, optional
        Column name where the sequence is stored.
    max_length : int, default=1024
        Maximum sequence length. Sequences shorter than this will be zero-padded.
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
            name_logging=OneHotEncoder.__name__
        )

    def __generate_vector_by_residue(self, residue: str) -> List[int]:
        """
        Generates a one-hot vector for a given residue.

        Parameters
        ----------
        residue : str
            A single-letter amino acid code.

        Returns
        -------
        List[int]
            A one-hot encoded list of length 20.
        """
        vector_coded = [0] * 20
        try:
            position = Constant.POSITION_RESIDUES[residue]
            vector_coded[position] = 1
        except KeyError:
            self.__logger__.warning("Residue '%s' not found in POSITION_RESIDUES mapping.", residue)
        return vector_coded

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
        return [0] * (self.max_length * 20 - current_length)

    def __coding_sequence(self, sequence: str) -> List[int]:
        """
        Encodes a full sequence into a one-hot representation, padded as needed.

        Parameters
        ----------
        sequence : str
            Amino acid sequence to encode.

        Returns
        -------
        List[int]
            One-hot encoded and padded sequence vector.
        """
        coded_vector = []
        for residue in sequence:
            coded_vector.extend(self.__generate_vector_by_residue(residue))

        if len(sequence) < self.max_length:
            coded_vector += self.__zero_padding(len(coded_vector))

        return coded_vector

    def run_process(self) -> None:
        """
        Performs one-hot encoding of all sequences in the dataset.

        The resulting encoded vectors are stored in `self.coded_dataset`.

        Raises
        ------
        RuntimeError
            If the encoding fails due to invalid input or execution error.
        """
        if not self.status:
            self.__logger__.warning("Encoding was not performed because the dataset was marked invalid.")
            return

        try:
            self.__logger__.info("Starting one-hot encoding for %d sequences.", len(self.dataset))

            matrix_coded = [
                self.__coding_sequence(self.dataset.at[index, self.sequence_column])
                for index in self.dataset.index
            ]

            header = [f"p_{i}" for i in range(len(matrix_coded[0]))]
            self.coded_dataset = pd.DataFrame(data=matrix_coded, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values

            self.__logger__.info("One-hot encoding completed successfully with %d features.",
                            self.coded_dataset.shape[1])

        except Exception as e:
            self.status = False
            self.message = f"[ERROR] One-hot encoding failed: {str(e)}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
