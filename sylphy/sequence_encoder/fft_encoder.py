from __future__ import annotations

import logging
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from scipy.fft import fft

from sylphy.logging import add_context, get_logger
from sylphy.misc.utils_lib import UtilsLib


class FFTEncoder:
    """
    Apply FFT to each numeric row (after removing the `sequence_column`).

    Notes
    -----
    Expects a numeric matrix in the dataframe except for the `sequence_column`,
    which is preserved and re-attached to the output.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        sequence_column: str = "sequence",
        debug: bool = False,
        debug_mode: int = logging.INFO,
    ) -> None:
        self.sequence_column = sequence_column
        _ = get_logger("sylphy")
        self.__logger__ = logging.getLogger("sylphy.sequence_encoder.FFTEncoder")
        self.__logger__.setLevel(debug_mode if debug else logging.NOTSET)
        add_context(self.__logger__, component="sequence_encoder", encoder="FFTEncoder")

        # Keep a copy and store sequences aside
        self.dataset = dataset.copy()
        self.sequence_list = self.dataset[self.sequence_column].values
        self.dataset = self.dataset.drop(columns=[self.sequence_column])

        # Determine FFT size (next power of two >= number of numeric columns)
        self.max_length = len(self.dataset.columns)
        self.init_process()

        self.coded_dataset: Optional[pd.DataFrame] = None

    def __get_near_pow(self) -> None:
        self.__logger__.info("Computing nearest power-of-two for padding.")
        self.stop_value = int(2 ** int(np.ceil(np.log2(max(1, self.max_length)))))
        self.__logger__.info("FFT stop value set to %d.", self.stop_value)

    def __complete_zero_padding(self) -> None:
        self.__logger__.info("Applying zero-padding up to %d.", self.stop_value)
        pad = self.stop_value - self.max_length
        if pad > 0:
            padding_df = pd.DataFrame(
                data=np.zeros((self.dataset.shape[0], pad), dtype=float),
                columns=[f"p_{i + self.max_length}" for i in range(pad)],
                index=self.dataset.index,
            )
            self.dataset = pd.concat([self.dataset, padding_df], axis=1)

    def init_process(self) -> None:
        self.__logger__.info("Initializing FFT encoding process.")
        self.__get_near_pow()
        self.__complete_zero_padding()

    def __create_row(self, index: int) -> List[float]:
        return self.dataset.iloc[index].tolist()

    def __apply_fft(self, index: int) -> List[float]:
        try:
            row = self.__create_row(index)
            yf = fft(row)
            return np.abs(yf[: self.stop_value // 2]).tolist()
        except Exception as e:
            self.__logger__.error("Error applying FFT at index %d: %s", index, e)
            return [0.0] * (self.stop_value // 2)

    def encoding_dataset(self) -> None:
        try:
            self.__logger__.info("Encoding dataset with FFT.")
            matrix = [self.__apply_fft(i) for i in self.dataset.index]
            header = [f"p_{i}" for i in range(len(matrix[0]))]
            self.coded_dataset = pd.DataFrame(matrix, columns=header)
            self.coded_dataset[self.sequence_column] = self.sequence_list
            self.__logger__.info("FFT encoding complete. Output shape: %s", self.coded_dataset.shape)
        except Exception as e:
            self.__logger__.error("Failed to encode dataset with FFT: %s", e)
            raise RuntimeError("FFT encoding failed.") from e

    def run_process(self) -> None:
        """Convenience to mirror other encoders."""
        self.encoding_dataset()

    def export_encoder(
        self,
        df_encoder: pd.DataFrame,
        path: str,
        file_format: Literal["csv", "npy", "npz", "parquet"] = "csv",
    ) -> None:
        UtilsLib.export_data(
            df_encoded=df_encoder,
            path=path,
            base_message="FFT encoder output",
            file_format=file_format,
        )
