import numpy as np
import pandas as pd
from scipy.fft import fft
from typing import Optional, List, Literal
from .logging_config import setup_logger
import logging
from bioclust.misc.utils_lib import UtilsLib

class FFTEncoder:
    """
    Applies Fast Fourier Transform (FFT) encoding to a numeric dataset.

    This class processes a dataset by applying zero-padding to the nearest power-of-two
    length and encoding each row using the FFT algorithm. It supports ignoring specific
    columns (e.g., metadata) and returns the transformed dataset.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset to be encoded using FFT.
    ignore_columns : list, optional
        Columns to exclude from FFT transformation.

    Attributes
    ----------
    dataset : pd.DataFrame
        Input dataset with zero-padding applied.
    ignore_columns : list
        Columns excluded from FFT.
    max_length : int
        Number of columns considered for FFT (excluding ignored columns).
    stop_value : int
        Nearest power-of-two length for FFT computation.
    coded_dataset : pd.DataFrame
        FFT-encoded dataset.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        sequence_column:str="sequence",
        debug:bool=False,
        debug_mode:int=logging.INFO

    ) -> None:
        self.dataset = dataset.copy()
        self.sequence_column = sequence_column
        self.coded_dataset: Optional[pd.DataFrame] = None
        
        self.sequence_list = self.dataset[self.sequence_column].values
        self.dataset = self.dataset.drop(columns=[self.sequence_column])

        self.__logger__ = setup_logger(
            name=FFTEncoder.__name__,
            level=debug_mode,
            enable=debug)

        self.max_length = len(self.dataset.columns) - 1
        self.init_process()

    def __get_near_pow(self) -> None:
        """Determine the next power of two greater than or equal to max_length."""
        self.__logger__.info("Computing nearest power-of-two for padding.")
        self.stop_value = 2 ** int(np.ceil(np.log2(self.max_length)))
        self.__logger__.info("FFT stop value set to %d.", self.stop_value)

    def __complete_zero_padding(self) -> None:
        """Apply zero-padding to each row to reach length of next power of two."""
        self.__logger__.info("Applying zero-padding up to %d.", self.stop_value)
        padding_needed = self.stop_value - self.max_length

        if padding_needed > 0:
            padding_df = pd.DataFrame(
                data=np.zeros((self.dataset.shape[0], padding_needed)),
                columns=[f"p_{i+self.max_length}" for i in range(padding_needed)],
                index=self.dataset.index
            )
            self.dataset = pd.concat([self.dataset, padding_df], axis=1)

    def init_process(self) -> None:
        """Execute all preprocessing steps before FFT encoding."""
        self.__logger__.info("Initializing FFT encoding process.")
        self.__get_near_pow()
        self.__complete_zero_padding()

    def __create_row(self, index: int) -> List[float]:
        """Convert a row to list format."""
        return self.dataset.iloc[index].tolist()

    def __apply_fft(self, index: int) -> List[float]:
        """Apply FFT to a row vector and return magnitude spectrum."""
        try:
            row = self.__create_row(index)
            yf = fft(row)
            yf_magnitude = np.abs(yf[:self.stop_value // 2])
            return yf_magnitude.tolist()
        except Exception as e:
            self.__logger__.error("Error applying FFT at index %d: %s", index, str(e))
            return [0.0] * (self.stop_value // 2)

    def encoding_dataset(self) -> None:
        """
        Apply FFT row-wise and construct the encoded DataFrame.

        Populates `self.coded_dataset` with the FFT-encoded values.
        """
        try:
            self.__logger__.info("Encoding dataset with FFT.")
            matrix_response = [self.__apply_fft(index) for index in self.dataset.index]
            header = [f"p_{i}" for i in range(len(matrix_response[0]))]

            self.coded_dataset = pd.DataFrame(matrix_response, columns=header)
            self.coded_dataset[self.sequence_column] = self.sequence_list
            
            self.__logger__.info("FFT encoding complete. Output shape: %s", self.coded_dataset.shape)
        except Exception as e:
            self.__logger__.error("Failed to encode dataset with FFT: %s", str(e))
            raise RuntimeError("FFT encoding failed.") from e

    def export_encoder(
        self,
        df_encoder: pd.DataFrame,
        path: str,
        file_format: Literal["csv", "npy"] = "csv"
    ) -> None:
        """
        Save the encoder matrix to disk.
        """
        UtilsLib.export_data(
            df_encoded=df_encoder,
            path=path,
            __logger__= self.__logger__,
            base_message= "Encoder generated ",
            file_format= file_format
        )