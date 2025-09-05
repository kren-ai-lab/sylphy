import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Literal, Any
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances
import uuid
import datetime as dt
import os
import shutil
from pathlib import Path

__logger__ = logging.getLogger("utils.UtilsLib")


class UtilsLib:
    """
    Utility library for common data manipulation tasks including stratified sampling,
    distance estimation, and statistical summarization.
    """

    @classmethod
    def random_selection(
        cls,
        df: pd.DataFrame,
        label_name: str = "",
        labels: Optional[List[str]] = None,
        n_samples: int = 100,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Randomly select `n_samples` rows from a DataFrame, optionally using stratified sampling
        by a specified label column.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to sample from.
        label_name : str, optional
            Column name used for stratified sampling.
        labels : list of str, optional
            List of labels to stratify by.
        n_samples : int, default=100
            Number of samples to select per label (if labels provided) or total otherwise.
        random_state : int, default=42
            Seed for reproducible sampling.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the selected samples.

        Raises
        ------
        ValueError
            If `label_name` is missing or any provided `labels` are not found in the column.
        """
        if labels is None or len(labels) == 0:
            __logger__.info("Performing random sampling without stratification.")
            return shuffle(df, random_state=random_state).head(n_samples)

        if label_name not in df.columns:
            __logger__.error(f"Label column '{label_name}' not found in DataFrame.")
            raise ValueError(f"Label column '{label_name}' not found in DataFrame.")

        unknown_labels = [label for label in labels if label not in df[label_name].unique()]
        if unknown_labels:
            __logger__.error(f"Unknown labels in column '{label_name}': {unknown_labels}")
            raise ValueError(
                f"The following labels were not found in column '{label_name}': {unknown_labels}"
            )

        selected_dfs = []
        for label in labels:
            subset = df[df[label_name] == label]
            sampled_subset = shuffle(subset, random_state=random_state).head(n_samples)
            selected_dfs.append(sampled_subset)
            __logger__.info(f"Sampled {len(sampled_subset)} rows for label '{label}'.")

        result = pd.concat(selected_dfs, axis=0).reset_index(drop=True)
        __logger__.info(f"Total sampled rows: {len(result)}")
        return result

    @classmethod
    def estimated_distance(
        cls,
        matrix_data: np.ndarray,
        metric: Literal[
            'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean',
            'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
            'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        ] = "euclidean"
    ) -> np.ndarray:
        """
        Compute pairwise distances between rows of a numeric matrix.

        Parameters
        ----------
        matrix_data : np.ndarray
            2D numeric matrix with shape (n_samples, n_features).
        metric : str, default='euclidean'
            Distance metric to use.

        Returns
        -------
        np.ndarray
            Square matrix of pairwise distances.

        Raises
        ------
        TypeError
            If `matrix_data` is not a NumPy ndarray.
        ValueError
            If `metric` is not supported.
        """
        if not isinstance(matrix_data, np.ndarray):
            __logger__.error("Input must be a NumPy ndarray.")
            raise TypeError("Input must be a NumPy ndarray.")

        supported_metrics = {
            'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'nan_euclidean',
            'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
            'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        }

        if metric not in supported_metrics:
            __logger__.error(f"Unsupported metric '{metric}' specified.")
            raise ValueError(f"Unsupported metric '{metric}'. Must be one of: {sorted(supported_metrics)}")

        __logger__.info(f"Computing pairwise distances using metric '{metric}'")
        return pairwise_distances(matrix_data, metric=metric)

    @classmethod
    def create_jobid(cls) -> str:
        ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        jid = f"{ts}-{uuid.uuid4().hex[:12]}"
        __logger__.info(f"Creating job_id {jid}")
        return jid
    
    @classmethod
    def delete_folder(cls, path_to_folder:str) -> None:
        
        folder = Path(path_to_folder)
        if folder.exists() and folder.is_dir():
            shutil.rmtree(folder)
    
    @classmethod
    def export_data(
        self,
        df_encoded: pd.DataFrame,
        path: str,
        __logger__:Any,
        base_message : str = "Encoded data",
        file_format: Literal["csv", "npy"] = "csv",
    ) -> None:
        """
        Save the coded matrix to disk.
        """
        try:
            if file_format == "csv":
                df_encoded.to_csv(path, index=False)
                __logger__.info(f"{base_message} exported to CSV: {path}")
            elif file_format == "npy":
                np.save(path, df_encoded.values)
                __logger__.info(f"{base_message} exported to NPY: {path}")
            else:
                raise ValueError(f"Unsupported file format '{file_format}'.")
        except Exception as e:
            __logger__.error(f"Failed to export {base_message}: {e}")
            raise