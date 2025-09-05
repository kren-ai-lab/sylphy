# protein_representation/reductions/linear_reductions.py
from __future__ import annotations

from typing import Optional, Tuple, Union
import logging
import traceback
import numpy as np
import pandas as pd

from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    SparsePCA,
    FastICA,
    TruncatedSVD,
    MiniBatchNMF,
    NMF,
    MiniBatchSparsePCA,
    FactorAnalysis,
    LatentDirichletAllocation,
)

from .reduction_methods import Reductions


class LinearReduction(Reductions):
    """
    Linear dimensionality reductions (PCA/ICA/NMF/LDA/etc.) with a consistent API.

    All methods log their parameters and shapes, return either NumPy arrays or
    DataFrames depending on ``return_type``. For estimators where the trained model
    is also valuable, we return ``(fitted_model, transformed)``.
    """

    def __init__(
        self,
        dataset: Union[np.ndarray, pd.DataFrame],
        *,
        return_type: str = "numpy",
        debug: bool = True,
        debug_mode: int = logging.INFO,
    ) -> None:
        super().__init__(
            dataset=dataset,
            return_type=return_type,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=LinearReduction.__name__,
        )
        self.__logger__.info("Initialized LinearReduction with dataset shape=%s", self.dataset.shape)

    def _apply_model(
        self,
        model,
        method_name: str,
        n_components: Optional[int] = None,
    ) -> Tuple[object, Optional[Union[np.ndarray, pd.DataFrame]]]:
        """Fit/transform wrapper with logging and error handling."""
        try:
            params = getattr(model, "get_params", lambda: {})()
            self.__logger__.info("Applying %s with params=%s", method_name, params)
            transformed = model.fit_transform(self.dataset)
            k = n_components if n_components is not None else getattr(model, "n_components", transformed.shape[1])
            self.__logger__.info("%s successful. Output shape=%s", method_name, transformed.shape)
            return model, self.generate_dataset_post_reduction(transformed, k)
        except Exception as e:
            self.__logger__.error("%s failed: %s", method_name, e)
            self.__logger__.debug(traceback.format_exc())
            return model, None

    # Public API
    def apply_pca(self, **kwargs) -> Tuple[PCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(PCA(**kwargs), "PCA", kwargs.get("n_components"))

    def apply_incremental_pca(self, **kwargs) -> Tuple[IncrementalPCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(IncrementalPCA(**kwargs), "IncrementalPCA", kwargs.get("n_components"))

    def apply_sparse_pca(self, **kwargs) -> Tuple[SparsePCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(SparsePCA(**kwargs), "SparsePCA", kwargs.get("n_components"))

    def apply_minibatch_sparse_pca(self, **kwargs) -> Tuple[MiniBatchSparsePCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(MiniBatchSparsePCA(**kwargs), "MiniBatchSparsePCA", kwargs.get("n_components"))

    def apply_fast_ica(self, **kwargs) -> Tuple[FastICA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(FastICA(**kwargs), "FastICA", kwargs.get("n_components"))

    def apply_truncated_svd(self, **kwargs) -> Tuple[TruncatedSVD, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(TruncatedSVD(**kwargs), "TruncatedSVD", kwargs.get("n_components"))

    def apply_factor_analysis(self, **kwargs) -> Tuple[FactorAnalysis, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(FactorAnalysis(**kwargs), "FactorAnalysis", kwargs.get("n_components"))

    def apply_nmf(self, **kwargs) -> Tuple[NMF, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(NMF(**kwargs), "NMF", kwargs.get("n_components"))

    def apply_minibatch_nmf(self, **kwargs) -> Tuple[MiniBatchNMF, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(MiniBatchNMF(**kwargs), "MiniBatchNMF", kwargs.get("n_components"))

    def apply_latent_dirichlet_allocation(self, **kwargs) -> Tuple[LatentDirichletAllocation, Optional[Union[np.ndarray, pd.DataFrame]]]:
        return self._apply_model(LatentDirichletAllocation(**kwargs), "LatentDirichletAllocation", kwargs.get("n_components"))
