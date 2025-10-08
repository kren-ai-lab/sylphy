from __future__ import annotations

import sylphy.reductions as r


def test_public_surface():
    """The reductions module should expose the main symbols."""
    assert hasattr(r, "Reductions")
    assert hasattr(r, "LinearReduction")
    assert hasattr(r, "NonLinearReductions")
    assert hasattr(r, "reduce_dimensionality")
    assert hasattr(r, "get_available_methods")
    assert hasattr(r, "is_linear_method")
    assert hasattr(r, "is_nonlinear_method")


def test_get_available_and_kind_helpers():
    """Helper queries for available methods and kind classification."""
    all_methods = r.get_available_methods()
    assert "pca" in all_methods and "isomap" in all_methods

    linear = r.get_available_methods("linear")
    nonlinear = r.get_available_methods("nonlinear")
    assert "pca" in linear and "pca" not in nonlinear
    assert "isomap" in nonlinear and "isomap" not in linear

    assert r.is_linear_method("pca") is True
    assert r.is_nonlinear_method("pca") is False
    assert r.is_nonlinear_method("isomap") is True
