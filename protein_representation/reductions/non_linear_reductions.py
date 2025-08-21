# protein_representation/reductions/non_linear_reductions.py
from __future__ import annotations

from typing import Optional, Union
import logging
import traceback
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning
from clustpy.partition import DipExt
import umap.umap_ as umap

from .reduction_methods import Reductions


class NonLinearReductions(Reductions):
    """
    Non-linear dimensionality reductions with a unified interface.

    Supported methods
    -----------------
    - t-SNE, Isomap, MDS, LLE, Spectral Embedding, UMAP
    - DictionaryLearning, MiniBatchDictionaryLearning
    - DipExt (ClustPy)
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
            name_logging=NonLinearReductions.__name__,
        )
        self.__logger__.info("Initialized NonLinearReductions with dataset shape=%s", self.dataset.shape)

    def _apply_model(
        self,
        model,
        method_name: str,
        n_components: Optional[int] = None
    ) -> Union[np.ndarray, pd.DataFrame, None]:
        try:
            params = getattr(model, "get_params", lambda: {})()
            self.__logger__.info("Applying %s with params=%s", method_name, params)
            transformed = model.fit_transform(self.dataset)
            k = n_components if n_components is not None else getattr(model, "n_components", transformed.shape[1])
            self.__logger__.info("%s successful. Output shape=%s", method_name, transformed.shape)
            return self.generate_dataset_post_reduction(transformed, k)
        except Exception as e:
            self.__logger__.error("%s failed: %s", method_name, e)
            self.__logger__.debug(traceback.format_exc())
            return None

    # Public API (one wrapper per method)
    def apply_tsne(self, **kwargs):
        return self._apply_model(TSNE(**kwargs), "t-SNE", kwargs.get("n_components"))

    def apply_isomap(self, **kwargs):
        return self._apply_model(Isomap(**kwargs), "Isomap", kwargs.get("n_components"))

    def apply_mds(self, **kwargs):
        return self._apply_model(MDS(**kwargs), "MDS", kwargs.get("n_components"))

    def apply_lle(self, **kwargs):
        return self._apply_model(LocallyLinearEmbedding(**kwargs), "LLE", kwargs.get("n_components"))

    def apply_spectral(self, **kwargs):
        return self._apply_model(SpectralEmbedding(**kwargs), "SpectralEmbedding", kwargs.get("n_components"))

    def apply_umap(self, **kwargs):
        return self._apply_model(umap.UMAP(**kwargs), "UMAP", kwargs.get("n_components"))

    def apply_dictionary_learning(self, **kwargs):
        return self._apply_model(DictionaryLearning(**kwargs), "DictionaryLearning", kwargs.get("n_components"))

    def apply_mini_batch_dictionary_learning(self, **kwargs):
        return self._apply_model(MiniBatchDictionaryLearning(**kwargs), "MiniBatchDictionaryLearning", kwargs.get("n_components"))

    def apply_dip_ext(self, **kwargs):
        return self._apply_model(DipExt(**kwargs), "DipExt", kwargs.get("n_components"))
