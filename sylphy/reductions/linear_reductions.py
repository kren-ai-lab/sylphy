from __future__ import annotations

from typing import Optional, Tuple, Union, Any, Type
import logging
import traceback
import inspect
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

from .reduction_methods import Reductions, ReturnType, Preprocess


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
        return_type: ReturnType = "numpy",
        preprocess: Preprocess = "none",
        random_state: Optional[int] = None,
        debug: bool = True,
        debug_mode: int = logging.INFO,
    ) -> None:
        super().__init__(
            dataset=dataset,
            return_type=return_type,
            preprocess=preprocess,
            random_state=random_state,
            debug=debug,
            debug_mode=debug_mode,
            name_logging=LinearReduction.__name__,
        )
        self.__logger__.info("Initialized LinearReduction with dataset shape=%s", self.dataset.shape)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _init_with_seed(self, cls: Type, kwargs: dict) -> Any:
        sig = inspect.signature(cls.__init__)
        params = set(sig.parameters.keys())
        k = dict(kwargs)
        if "random_state" in params and "random_state" not in k:
            k["random_state"] = self.random_state
        return cls(**k)

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

    # -----------------------------
    # Public API
    # -----------------------------
    def apply_pca(self, **kwargs) -> Tuple[PCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(PCA, kwargs)
        return self._apply_model(model, "PCA", kwargs.get("n_components"))

    def apply_incremental_pca(self, **kwargs) -> Tuple[IncrementalPCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(IncrementalPCA, kwargs)
        return self._apply_model(model, "IncrementalPCA", kwargs.get("n_components"))

    def apply_sparse_pca(self, **kwargs) -> Tuple[SparsePCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(SparsePCA, kwargs)
        return self._apply_model(model, "SparsePCA", kwargs.get("n_components"))

    def apply_minibatch_sparse_pca(self, **kwargs) -> Tuple[MiniBatchSparsePCA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(MiniBatchSparsePCA, kwargs)
        return self._apply_model(model, "MiniBatchSparsePCA", kwargs.get("n_components"))

    def apply_fast_ica(self, **kwargs) -> Tuple[FastICA, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(FastICA, kwargs)
        return self._apply_model(model, "FastICA", kwargs.get("n_components"))

    def apply_truncated_svd(self, **kwargs) -> Tuple[TruncatedSVD, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(TruncatedSVD, kwargs)
        return self._apply_model(model, "TruncatedSVD", kwargs.get("n_components"))

    def apply_factor_analysis(self, **kwargs) -> Tuple[FactorAnalysis, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(FactorAnalysis, kwargs)
        return self._apply_model(model, "FactorAnalysis", kwargs.get("n_components"))

    def apply_nmf(self, **kwargs) -> Tuple[NMF, Optional[Union[np.ndarray, pd.DataFrame]]]:
        # NMF requires non-negative data; users should ensure this upstream or via preprocess
        model = self._init_with_seed(NMF, kwargs)
        return self._apply_model(model, "NMF", kwargs.get("n_components"))

    def apply_minibatch_nmf(self, **kwargs) -> Tuple[MiniBatchNMF, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(MiniBatchNMF, kwargs)
        return self._apply_model(model, "MiniBatchNMF", kwargs.get("n_components"))

    def apply_latent_dirichlet_allocation(self, **kwargs) -> Tuple[LatentDirichletAllocation, Optional[Union[np.ndarray, pd.DataFrame]]]:
        model = self._init_with_seed(LatentDirichletAllocation, kwargs)
        return self._apply_model(model, "LatentDirichletAllocation", kwargs.get("n_components"))
