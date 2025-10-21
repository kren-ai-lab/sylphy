from __future__ import annotations

import pandas as pd

from sylphy.sequence_encoder import OrdinalEncoder


def test_missing_sequence_column_sets_status_false():
    """Verify encoder sets status=False when sequence column is missing."""
    df = pd.DataFrame({"seq": ["AAA"]})
    enc = OrdinalEncoder(dataset=df, sequence_column="sequence", max_length=3, debug=True)
    assert enc.status is False
    enc.run_process()
    assert hasattr(enc, "coded_dataset")
    assert enc.coded_dataset.empty
