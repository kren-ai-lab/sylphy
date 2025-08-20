import os
import io
import requests

import pandas as pd
from typing import Optional, List

from .base_encoder import Encoders
from bioclust.misc.constants import Constant
from bioclust.core.config import get_config
import logging

class PhysicochemicalEncoder(Encoders):
    """
    Encodes sequences based on a specified physicochemical property.

    This encoder maps each residue to a numerical value corresponding to a selected
    physicochemical property. Sequences are padded with zeros up to `max_length`.

    Inherits from:
        Encoders: Performs validation and filtering of input sequences.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset containing amino acid sequences.
    sequence_column : str
        Column name that contains the sequences.
    max_length : int, default=1024
        Maximum sequence length. Sequences shorter than this will be zero-padded.
    type_descriptor : str, default="aaindex"
        Type of descriptor file to use. Options: "aaindex", "group_based".
    name_property : str, default="ANDN920101"
        Name of the physicochemical property to use for encoding.
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = None,
        max_length: int = 1024,
        type_descriptor: str = "aaindex",
        name_property: str = "ANDN920101",
        debug:bool=False,
        debug_mode:int=logging.INFO
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column,
            max_length=max_length,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=PhysicochemicalEncoder.__name__
        )

        self.name_property = name_property
        self.df_properties = self._load_descriptor_file(type_descriptor)

        if self.name_property not in self.df_properties.columns:
            msg = f"Property '{self.name_property}' not found in descriptor file."
            self.__logger__.error(msg)
            raise ValueError(msg)

    def _load_descriptor_file(self, type_descriptor: str = "aaindex") -> pd.DataFrame:
        """
        Load the descriptor file from cache or download if not available.

        Parameters
        ----------
        type_descriptor : str
            Type of descriptor to load. Must be either "aaindex" or "group_based".

        Returns
        -------
        pd.DataFrame
            DataFrame with descriptor values indexed by residues.

        Raises
        ------
        ValueError
            If the descriptor type is not recognized.
        RuntimeError
            If the descriptor file cannot be read or downloaded.
        """
        if type_descriptor not in ["aaindex", "group_based"]:
            msg = f"Unsupported descriptor type: {type_descriptor}. Must be 'aaindex' or 'group_based'."
            self.__logger__.error(msg)
            raise ValueError(msg)

        base_url = (
            Constant.BASE_URL_AAINDEX if type_descriptor == "aaindex"
            else Constant.BASE_URL_CLUSTERS_DESCRIPTORS
        )

        cfg = get_config()
        cache_dir = os.path.join(cfg.cache_paths.data(), type_descriptor)
        os.makedirs(cache_dir, exist_ok=True)
        self.__logger__.info("Using cache directory at: %s", cache_dir)

        filepath = os.path.join(cache_dir, base_url.split("/")[-1])
        self.__logger__.info("Using filepath %s", filepath)
        if not os.path.exists(filepath):
            try:
                self.__logger__.warning("Descriptor file not found. Downloading from %s", base_url)
                s = requests.get(base_url).content
                df = pd.read_csv(io.StringIO(s.decode('utf-8')))
                df.to_csv(filepath, index=False)
                self.__logger__.info("Descriptor file downloaded and cached at %s", filepath)
            except Exception as e:
                self.__logger__.error("Failed to download descriptor file: %s", str(e))
                raise RuntimeError("Failed to load or download descriptor file.") from e

        try:
            return pd.read_csv(filepath, index_col=0)
        except Exception as e:
            self.__logger__.error("Failed to read descriptor file from cache: %s", str(e))
            raise RuntimeError("Failed to read cached descriptor file.") from e

    def __encoding_residue(self, residue: str) -> float:
        """
        Get the physicochemical value for a residue.

        Parameters
        ----------
        residue : str
            Single-letter amino acid code.

        Returns
        -------
        float
            Property value from df_properties.
        """
        try:
            return self.df_properties.at[residue, self.name_property]
        except KeyError:
            self.__logger__.warning("Residue '%s' not found in property table. Assigning 0.0", residue)
            return 0.0
        except Exception as e:
            self.__logger__.error("Unexpected error during residue encoding: %s", str(e))
            return 0.0

    def __encoding_sequence(self, sequence: str) -> List[float]:
        """
        Encode a full sequence into its physicochemical property vector.

        Parameters
        ----------
        sequence : str
            The input amino acid sequence.

        Returns
        -------
        List[float]
            Encoded sequence vector padded with zeros.
        """
        try:
            sequence = sequence.upper()
            sequence_encoding = [self.__encoding_residue(residue) for residue in sequence]
            padding_length = self.max_length - len(sequence_encoding)

            if padding_length > 0:
                sequence_encoding += [0.0] * padding_length

            return sequence_encoding

        except Exception as e:
            self.__logger__.error("Failed to encode sequence '%s': %s", sequence, str(e))
            return [0.0] * self.max_length

    def __encoding_dataset(self) -> None:
        """
        Apply encoding to the full dataset of sequences.

        Populates `self.coded_dataset` with the transformed matrix.
        """
        try:
            self.__logger__.info("Encoding and processing dataset...")
            matrix_data = [
                self.__encoding_sequence(self.dataset.at[index, self.sequence_column])
                for index in self.dataset.index
            ]

            header = [f"p_{i}" for i in range(len(matrix_data[0]))]
            self.coded_dataset = pd.DataFrame(matrix_data, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values

            self.__logger__.info("Encoding complete. Dataset shape: %s", self.coded_dataset.shape)

        except Exception as e:
            self.__logger__.error("Error encoding dataset: %s", str(e))
            raise RuntimeError("Failed to encode dataset.") from e

    def run_process(self) -> None:
        """
        Runs the encoding process if sequence validation passed.
        """
        if self.status:
            self.__logger__.info("Running encoding process for physicochemical encoding.")
            self.__encoding_dataset()
        else:
            self.__logger__.warning("Encoding aborted. Dataset validation failed.")
