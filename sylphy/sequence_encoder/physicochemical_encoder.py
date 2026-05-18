"""Implement physicochemical-property encoding for sequence datasets."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import polars as pl
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
        dataset: pl.DataFrame | None = None,
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
        # _prop_map: residue (uppercase) → float value for name_property
        self._prop_map = self._load_prop_map(type_descriptor)

    def _load_prop_map(self, type_descriptor: str = "aaindex") -> dict[str, float]:
        """Load descriptor data and return a residue → value mapping for the chosen property."""
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
                    resp = sess.get(base_url, timeout=60)
                    resp.raise_for_status()
                    raw_text = resp.content.decode("utf-8")
                df = pl.read_csv(io.StringIO(raw_text))
                df.write_csv(filepath)
                self.__logger__.info("Descriptor cached at %s", filepath)
            except Exception as e:
                self.__logger__.error("Failed to download descriptor file: %s", e)
                msg = "Failed to load or download descriptor file."
                raise RuntimeError(msg) from e

        try:
            df = pl.read_csv(filepath)
        except Exception as e:
            self.__logger__.error("Failed to read descriptor file from cache: %s", e)
            msg = "Failed to read cached descriptor file."
            raise RuntimeError(msg) from e

        # First column is the residue key
        residue_col = df.columns[0]
        df = df.with_columns(pl.col(residue_col).cast(pl.String).str.to_uppercase())

        if self.name_property not in df.columns:
            msg = f"Property '{self.name_property}' not found in descriptor file."
            self.__logger__.error(msg)
            raise ValueError(msg)

        return dict(
            zip(
                df[residue_col].to_list(),
                df[self.name_property].cast(pl.Float64).to_list(),
                strict=True,
            )
        )

    def _encode_residue(self, residue: str) -> float:
        r = residue.upper()
        if r not in self._prop_map:
            self.__logger__.warning("Residue '%s' not in property table. Using 0.0", residue)
            return 0.0
        val = self._prop_map[r]
        try:
            return float(val)
        except (TypeError, ValueError) as e:
            self.__logger__.error("Unexpected error during residue encoding: %s", e)
            return 0.0

    def _encode_sequence(self, sequence: str) -> list[float]:
        vec = [self._encode_residue(r) for r in sequence.upper()]
        pad = self.max_length - len(vec)
        if pad > 0:
            vec.extend([0.0] * pad)
        return vec[: self.max_length]

    def _encode_all(self) -> None:
        try:
            self.__logger__.info("Encoding dataset with physicochemical property: %s", self.name_property)
            sequences = self.dataset[self.sequence_column].to_list()
            matrix = np.array([self._encode_sequence(seq) for seq in sequences], dtype=np.float32)
            col_names = [f"p_{i}" for i in range(matrix.shape[1])]
            self.coded_dataset = pl.from_numpy(matrix, schema=col_names).with_columns(
                pl.Series(self.sequence_column, sequences)
            )
            self.__logger__.info("Encoding complete. Shape: %s", self.coded_dataset.shape)
        except Exception as e:
            self.__logger__.error("Error encoding dataset: %s", e)
            msg = "Failed to encode dataset."
            raise RuntimeError(msg) from e

    def run_process(self) -> None:
        """Encode validated sequences using the configured descriptor property."""
        self.__logger__.info("Running physicochemical encoding.")
        self._encode_all()
