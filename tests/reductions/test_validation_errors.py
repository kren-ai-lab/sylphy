from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sylphy.reductions import LinearReduction


def test_non_numeric_raises_type_error():
    """String-only frames should be rejected."""
    df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
    with pytest.raises((TypeError, ValueError)):
        LinearReduction(df)


def test_1d_array_raises_value_error():
    """1D inputs are invalid; require 2D feature matrices."""
    arr = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        LinearReduction(arr)
