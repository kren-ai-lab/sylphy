from __future__ import annotations

import pandas as pd
import pytest

from sylphy.sequence_encoder import OrdinalEncoder


def test_missing_sequence_column_raises_value_error() -> None:
    """Verify encoder raises ValueError when sequence column is missing."""
    df = pd.DataFrame({"seq": ["AAA"]})
    with pytest.raises(ValueError, match="Column 'sequence' not found in dataset"):
        OrdinalEncoder(dataset=df, sequence_column="sequence", max_length=3, debug=True)


def test_none_dataset_raises_value_error() -> None:
    """Verify encoder raises ValueError when dataset is None."""
    with pytest.raises(ValueError, match="No dataset provided"):
        OrdinalEncoder(dataset=None)
