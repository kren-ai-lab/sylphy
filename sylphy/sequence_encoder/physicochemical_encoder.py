# protein_representation/sequence_encoder/physicochemical_encoder.py
from __future__ import annotations

import io
import logging
import os
from typing import Optional, List

import pandas as pd
import requests

from .base_encoder import Encoders
from sylphy.constants.tool_constants import BASE_URL_AAINDEX, BASE_URL_CLUSTERS_DESCRIPTORS
from sylphy.core.config import get_config


class PhysicochemicalEncoder(Encoders):
    """
    Encode sequences using a selected physicochemical property (e.g., AAIndex).
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        sequence_column: Optional[str] = None,
        max_length: int = 1024,
        type_descriptor: str = "aaindex",
        name_property: str = "ANDN920101",
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column,
            max_length=max_length,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=PhysicochemicalEncoder.__name__,
        )

        self.name_property = name_property
        self.df_properties = self._load_descriptor_file(type_descriptor)

        if self.name_property not in self.df_properties.columns:
            msg = f"Property '{self.name_property}' not found in descriptor file."
            self.__logger__.error(msg)
            raise ValueError(msg)

    def _load_descriptor_file(self, type_descriptor: str = "aaindex") -> pd.DataFrame:
        if type_descriptor not in {"aaindex", "group_based"}:
            msg = "Unsupported descriptor type: %s. Must be 'aaindex' or 'group_based'." % type_descriptor
            self.__logger__.error(msg)
            raise ValueError(msg)

        base_url = (
            BASE_URL_AAINDEX if type_descriptor == "aaindex"
            else BASE_URL_CLUSTERS_DESCRIPTORS
        )

        cfg = get_config()
        cache_dir = os.path.join(cfg.cache_paths.data(), type_descriptor)
        os.makedirs(cache_dir, exist_ok=True)
        self.__logger__.info("Using cache directory at: %s", cache_dir)

        filename = base_url.split("/")[-1]
        filepath = os.path.join(cache_dir, filename)
        self.__logger__.info("Using descriptor file %s", filepath)

        if not os.path.exists(filepath):
            try:
                self.__logger__.warning("Descriptor file not found. Downloading from %s", base_url)
                s = requests.get(base_url, timeout=60)
                s.raise_for_status()
                df = pd.read_csv(io.StringIO(s.content.decode("utf-8")))
                df.to_csv(filepath, index=False)
                self.__logger__.info("Descriptor cached at %s", filepath)
            except Exception as e:
                self.__logger__.error("Failed to download descriptor file: %s", e)
                raise RuntimeError("Failed to load or download descriptor file.") from e

        try:
            return pd.read_csv(filepath, index_col=0)
        except Exception as e:
            self.__logger__.error("Failed to read descriptor file from cache: %s", e)
            raise RuntimeError("Failed to read cached descriptor file.") from e

    def __encoding_residue(self, residue: str) -> float:
        try:
            return float(self.df_properties.at[residue, self.name_property])
        except KeyError:
            self.__logger__.warning("Residue '%s' not in property table. Using 0.0", residue)
            return 0.0
        except Exception as e:
            self.__logger__.error("Unexpected error during residue encoding: %s", e)
            return 0.0

    def __encoding_sequence(self, sequence: str) -> List[float]:
        try:
            seq = sequence.upper()
            vec = [self.__encoding_residue(r) for r in seq]
            pad = self.max_length - len(vec)
            if pad > 0:
                vec.extend([0.0] * pad)
            return vec
        except Exception as e:
            self.__logger__.error("Failed to encode sequence '%s': %s", sequence, e)
            return [0.0] * self.max_length

    def __encoding_dataset(self) -> None:
        try:
            self.__logger__.info("Encoding dataset with physicochemical property: %s", self.name_property)
            matrix = [self.__encoding_sequence(self.dataset.at[i, self.sequence_column]) for i in self.dataset.index]
            header = [f"p_{i}" for i in range(len(matrix[0]))]
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].values
            self.__logger__.info("Encoding complete. Dataset shape: %s", self.coded_dataset.shape)
        except Exception as e:
            self.__logger__.error("Error encoding dataset: %s", e)
            raise RuntimeError("Failed to encode dataset.") from e

    def run_process(self) -> None:
        if self.status:
            self.__logger__.info("Running physicochemical encoding.")
            self.__encoding_dataset()
        else:
            self.__logger__.warning("Encoding aborted. Dataset validation failed.")
