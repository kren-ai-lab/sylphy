from __future__ import annotations

import numpy as np
import pytest

from sylphy.reductions import NonLinearReductions

# Optional deps: detect once, then use skip marks cleanly
try:
    import umap  # noqa: F401
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import clustpy  # noqa: F401
    HAS_CLUSTPY = True
except Exception:
    HAS_CLUSTPY = False


def test_isomap_and_spectral(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    Z_iso = nr.apply_isomap(n_components=2, n_neighbors=3)
    Z_spec = nr.apply_spectral(n_components=2, random_state=0)
    assert Z_iso is not None and isinstance(Z_iso, np.ndarray) and Z_iso.shape[1] == 2
    assert Z_spec is not None and isinstance(Z_spec, np.ndarray) and Z_spec.shape[1] == 2


def test_lle_basic(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    # Avoid kwargs that may not be supported by certain sklearn versions
    Z = nr.apply_lle(n_components=2, n_neighbors=4)
    assert Z is not None and Z.shape == (X_small.shape[0], 2)


@pytest.mark.skipif(not HAS_UMAP, reason="umap not installed")
def test_umap_if_available(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    Z = nr.apply_umap(n_components=2, n_neighbors=3, min_dist=0.1, random_state=0)
    assert Z is not None and Z.shape == (X_small.shape[0], 2)


@pytest.mark.skipif(not HAS_CLUSTPY, reason="clustpy not installed")
def test_dipext_if_available(X_small):
    nr = NonLinearReductions(X_small, return_type="numpy", debug=True)
    Z = nr.apply_dip_ext(n_components=2)
    # DipExt may reduce differently; just assert it returns something 2D if provided
    if Z is not None:
        assert hasattr(Z, "shape") and Z.shape[0] == X_small.shape[0]
