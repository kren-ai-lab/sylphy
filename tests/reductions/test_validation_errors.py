# tests/reductions/test_validation_errors.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sylphy.reductions import LinearReduction


def test_non_numeric_raises_type_error():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})
    with pytest.raises(TypeError):
        LinearReduction(df)


def test_1d_array_raises_value_error():
    arr = np.array([1, 2, 3], dtype=float)
    with pytest.raises(ValueError):
        LinearReduction(arr)
