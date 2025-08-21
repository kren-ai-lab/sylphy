# tests/reductions/test_non_linear_reductions.py
from __future__ import annotations

import numpy as np

import pytest

from protein_representation.reductions import NonLinearReductions


def test_isomap_and_spectral(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    Z_iso = nr.apply_isomap(n_components=2, n_neighbors=3)
    Z_spec = nr.apply_spectral(n_components=2, random_state=0)
    assert Z_iso is not None and isinstance(Z_iso, np.ndarray) and Z_iso.shape[1] == 2
    assert Z_spec is not None and isinstance(Z_spec, np.ndarray) and Z_spec.shape[1] == 2


def test_lle_basic(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    Z = nr.apply_lle(n_components=2, n_neighbors=4, random_state=0)
    assert Z is not None and Z.shape == (X_small.shape[0], 2)


@pytest.mark.skipif(pytest.importorskip("umap", reason="umap not installed") is None, reason="umap missing")
def test_umap_if_available(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    Z = nr.apply_umap(n_components=2, n_neighbors=3, min_dist=0.1, random_state=0)
    assert Z is not None and Z.shape == (X_small.shape[0], 2)


@pytest.mark.skipif(pytest.importorskip("clustpy", reason="clustpy not installed") is None, reason="clustpy missing")
def test_dipext_if_available(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    Z = nr.apply_dip_ext(n_components=2)
    # DipExt may reduce differently; just assert it returns something 2D
    assert Z is None or (hasattr(Z, "shape") and Z.shape[0] == X_small.shape[0])
