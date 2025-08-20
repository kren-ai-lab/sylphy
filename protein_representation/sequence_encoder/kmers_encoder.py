import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, List

from .base_encoder import Encoders
import logging

class KMersEncoders(Encoders):
    """
    Encodes sequences into k-mer-based TF-IDF feature vectors.

    This encoder splits amino acid sequences into overlapping k-mers of a given length,
    transforms them into string representations, and then uses TF-IDF vectorization
    to encode them into a numerical format.

    Inherits from
    ----------
    Encoders : Provides validation and filtering of input sequences.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing amino acid sequences.
    sequence_column : str
        Column name that contains the sequences.
    size_kmer : int, default=3
        The length of k-mers to generate from each sequence.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = "sequence",
        size_kmer: int = 3,
        debug:bool=False,
        debug_mode:int=logging.INFO
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=KMersEncoders.__name__
        )
        self.size_kmer = size_kmer

    def kmer(self, seq: str, kmer_length: int = 3) -> str:
        """
        Generate a space-separated string of k-mers from a sequence.

        Parameters
        ----------
        seq : str
            The input amino acid sequence.
        kmer_length : int, optional
            The size of each k-mer (default is 3).

        Returns
        -------
        str
            A string with k-mers separated by spaces.
        """
        return " ".join(seq[i: i + kmer_length] for i in range(len(seq) - kmer_length + 1))

    def process_dataset(self) -> None:
        """
        Transforms the dataset of sequences into TF-IDF encoded k-mer vectors.
        The encoded feature matrix is stored in `self.coded_dataset`.

        Raises
        ------
        RuntimeError
            If k-mer encoding or vectorization fails.
        """
        if not self.status:
            self.__logger__.warning("Encoding aborted due to failed validation.")
            return

        try:
            self.__logger__.info("Starting k-mer encoding with size_kmer=%d", self.size_kmer)

            self.dataset["kmer_sequence"] = self.dataset[self.sequence_column].apply(
                lambda x: self.kmer(x, self.size_kmer)
            )

            vectorizer = TfidfVectorizer()
            transformed_data = vectorizer.fit_transform(self.dataset["kmer_sequence"]).astype("float32")

            self.coded_dataset = pd.DataFrame(
                data=transformed_data.toarray(),
                columns=[col.upper() for col in vectorizer.get_feature_names_out()]
            )

            # Add preserved columns
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values

            # Optional cleanup
            if "sequence" in self.coded_dataset.columns and self.sequence_column != "sequence":
                self.coded_dataset.drop(columns=["sequence"], inplace=True)

            self.__logger__.info("TF-IDF k-mer encoding completed successfully with %d features.",
                            self.coded_dataset.shape[1])

        except Exception as e:
            self.status = False
            self.message = f"[ERROR] K-mer encoding failed: {str(e)}"
            self.__logger__.exception(self.message)
            raise RuntimeError(self.message)
