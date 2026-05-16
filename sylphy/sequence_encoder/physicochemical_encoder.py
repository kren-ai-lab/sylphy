"""Implement physicochemical-property encoding for sequence datasets."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import cast

import pandas as pd
import requests

from sylphy.constants import BASE_URL_AAINDEX, BASE_URL_CLUSTERS_DESCRIPTORS
from sylphy.core import get_config

from .encoder_base import EncoderBase


class PhysicochemicalEncoder(EncoderBase):
    """Encode sequences using a selected physicochemical property (e.g., AAIndex).

    Notes:
        Residues missing from the property table are encoded as ``0.0``.

    """

    def __init__(
        self,
        dataset: pd.DataFrame | None = None,
        sequence_column: str | None = None,
        max_length: int = 1024,
        type_descriptor: str = "aaindex",
        name_property: str = "ANDN920101",
        *,
        allow_extended: bool = False,
        allow_unknown: bool = False,
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        """Initialize the physicochemical encoder."""
        super().__init__(
            dataset=dataset,
            sequence_column=sequence_column or "sequence",
            max_length=max_length,
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
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
        """Load descriptor data from cache or download it when missing."""
        if type_descriptor not in {"aaindex", "group_based"}:
            msg = f"Unsupported descriptor type: {type_descriptor}. Must be 'aaindex' or 'group_based'."
            self.__logger__.error(msg)
            raise ValueError(msg)

        base_url = BASE_URL_AAINDEX if type_descriptor == "aaindex" else BASE_URL_CLUSTERS_DESCRIPTORS

        cfg = get_config()
        cache_dir = Path(cfg.cache_paths.data()) / type_descriptor
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.__logger__.info("Using cache directory at: %s", cache_dir)

        filename = base_url.split("/")[-1]
        filepath = cache_dir / filename
        self.__logger__.info("Using descriptor file %s", filepath)

        if not filepath.exists():
            try:
                self.__logger__.warning("Descriptor file not found. Downloading from %s", base_url)
                with requests.Session() as sess:
                    s = sess.get(base_url, timeout=60)
                    s.raise_for_status()
                    df = pd.read_csv(io.StringIO(s.content.decode("utf-8")))
                df.to_csv(filepath, index=False)
                self.__logger__.info("Descriptor cached at %s", filepath)
            except Exception as e:
                self.__logger__.error("Failed to download descriptor file: %s", e)
                msg_0 = "Failed to load or download descriptor file."
                raise RuntimeError(msg_0) from e

        try:
            df = pd.read_csv(filepath, index_col=0)
        except Exception as e:
            self.__logger__.error("Failed to read descriptor file from cache: %s", e)
            msg_0 = "Failed to read cached descriptor file."
            raise RuntimeError(msg_0) from e
        else:
            df.index = df.index.astype(str).str.upper()
            return df

    def __encoding_residue(self, residue: str) -> float:
        try:
            value = cast("float | int | str", self.df_properties.loc[residue, self.name_property])
            return float(value)
        except KeyError:
            self.__logger__.warning("Residue '%s' not in property table. Using 0.0", residue)
            return 0.0
        except (TypeError, ValueError) as e:
            self.__logger__.error("Unexpected error during residue encoding: %s", e)
            return 0.0

    def __encoding_sequence(self, sequence: str) -> list[float]:
        try:
            seq = sequence.upper()
            vec = [self.__encoding_residue(r) for r in seq]
            pad = self.max_length - len(vec)
            if pad > 0:
                vec.extend([0.0] * pad)
            return vec[: self.max_length]
        except (TypeError, ValueError) as e:
            self.__logger__.error("Failed to encode sequence '%s': %s", sequence, e)
            return [0.0] * self.max_length

    def __encoding_dataset(self) -> None:
        try:
            self.__logger__.info("Encoding dataset with physicochemical property: %s", self.name_property)
            matrix = [
                self.__encoding_sequence(cast("str", self.dataset.loc[i, self.sequence_column]))
                for i in self.dataset.index
            ]
            header = pd.Index([f"p_{i}" for i in range(len(matrix[0]))])
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.dataset[self.sequence_column].to_numpy()
            self.__logger__.info("Encoding complete. Dataset shape: %s", self.coded_dataset.shape)
        except Exception as e:
            self.__logger__.error("Error encoding dataset: %s", e)
            msg = "Failed to encode dataset."
            raise RuntimeError(msg) from e

    def run_process(self) -> None:
        """Encode validated sequences using the configured descriptor property."""
        self.__logger__.info("Running physicochemical encoding.")
        self.__encoding_dataset()
