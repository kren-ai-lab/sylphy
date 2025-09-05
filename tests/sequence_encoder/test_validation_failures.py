# tests/sequence_encoder/test_validation_failures.py
from __future__ import annotations

import pandas as pd

from sylphy.sequence_encoder import OrdinalEncoder


def test_missing_sequence_column_sets_status_false():
    df = pd.DataFrame({"seq": ["AAA"]})  # wrong column name
    enc = OrdinalEncoder(dataset=df, sequence_column="sequence", max_length=3, debug=True)
    assert enc.status is False
    # run_process should no-op but not crash
    enc.run_process()
    assert enc.coded_dataset.empty
