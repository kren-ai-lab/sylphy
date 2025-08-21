# tests/reductions/test_public_api_and_helpers.py
from __future__ import annotations

import protein_representation.reductions as r


def test_public_surface():
    assert hasattr(r, "Reductions")
    assert hasattr(r, "LinearReduction")
    assert hasattr(r, "NonLinearReductions")
    assert hasattr(r, "reduce_dimensionality")
    assert hasattr(r, "get_available_methods")
    assert hasattr(r, "is_linear_method")
    assert hasattr(r, "is_nonlinear_method")


def test_get_available_and_kind_helpers():
    all_methods = r.get_available_methods()
    assert "pca" in all_methods and "isomap" in all_methods

    linear = r.get_available_methods("linear")
    nonlinear = r.get_available_methods("nonlinear")
    assert "pca" in linear and "pca" not in nonlinear
    assert "isomap" in nonlinear and "isomap" not in linear

    assert r.is_linear_method("pca") is True
    assert r.is_nonlinear_method("pca") is False
    assert r.is_nonlinear_method("isomap") is True
