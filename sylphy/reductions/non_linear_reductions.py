from __future__ import annotations

from typing import Optional, Union, Any, Type
import logging
import traceback
import inspect
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning
from clustpy.partition import DipExt
import umap.umap_ as umap

from .reduction_methods import Reductions, ReturnType, Preprocess


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
            name_logging=NonLinearReductions.__name__,
        )
        self.__logger__.info("Initialized NonLinearReductions with dataset shape=%s", self.dataset.shape)

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

    # -----------------------------
    # Public API (one wrapper per method)
    # -----------------------------
    def apply_tsne(self, **kwargs):
        n = self.dataset.shape[0]
        perplexity = kwargs.get("perplexity", 30)
        if perplexity >= max(1, n):
            new_p = max(5, min(30, max(1, n // 3)))
            self.__logger__.warning("t-SNE perplexity=%s >= n=%s; using %s instead.", perplexity, n, new_p)
            kwargs["perplexity"] = new_p
        model = self._init_with_seed(TSNE, kwargs)
        return self._apply_model(model, "t-SNE", kwargs.get("n_components"))

    def apply_isomap(self, **kwargs):
        model = Isomap(**kwargs)  # no random_state
        return self._apply_model(model, "Isomap", kwargs.get("n_components"))

    def apply_mds(self, **kwargs):
        model = self._init_with_seed(MDS, kwargs)
        return self._apply_model(model, "MDS", kwargs.get("n_components"))

    def apply_lle(self, **kwargs):
        model = self._init_with_seed(LocallyLinearEmbedding, kwargs)
        return self._apply_model(model, "LLE", kwargs.get("n_components"))

    def apply_spectral(self, **kwargs):
        model = self._init_with_seed(SpectralEmbedding, kwargs)
        return self._apply_model(model, "SpectralEmbedding", kwargs.get("n_components"))

    def apply_umap(self, **kwargs):
        # umap-learn supports 'random_state'
        model = self._init_with_seed(umap.UMAP, kwargs)
        return self._apply_model(model, "UMAP", kwargs.get("n_components"))

    def apply_dictionary_learning(self, **kwargs):
        model = self._init_with_seed(DictionaryLearning, kwargs)
        return self._apply_model(model, "DictionaryLearning", kwargs.get("n_components"))

    def apply_mini_batch_dictionary_learning(self, **kwargs):
        model = self._init_with_seed(MiniBatchDictionaryLearning, kwargs)
        return self._apply_model(model, "MiniBatchDictionaryLearning", kwargs.get("n_components"))

    def apply_dip_ext(self, **kwargs):
        model = DipExt(**kwargs)  # library-specific, no random_state in __init__
        return self._apply_model(model, "DipExt", kwargs.get("n_components"))
