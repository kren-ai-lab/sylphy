# tests/sequence_encoder/test_physicochemical_encoder.py
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from sylphy.sequence_encoder import PhysicochemicalEncoder


def test_physicochemical_reads_cached_file(monkeypatch, tmp_path):
    """
    Avoid network: create the cached AAIndex file that the encoder expects.
    The code will skip download when the file already exists.
    """
    # Point config cache data() to tmp
    class _CachePaths:
        def data(self):  # matches usage in encoder
            return str(tmp_path / "data")

    class _Cfg:
        cache_paths = _CachePaths()

    from sylphy.core import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config", lambda: _Cfg(), raising=True)

    # Build the expected filepath: <data>/<type_descriptor>/<filename>
    # filename is taken from Constant.BASE_URL_AAINDEX basename; set in conftest.
    type_descriptor = "aaindex"
    filename = "aaindex.csv"
    cache_dir = Path(_Cfg().cache_paths.data()) / type_descriptor
    cache_dir.mkdir(parents=True, exist_ok=True)
    filepath = cache_dir / filename

    # Make a small CSV with index=Residue and a column ANDN920101
    content = "res,ANDN920101\nA,1.0\nC,2.0\nD,3.0\n"
    filepath.write_text(content, encoding="utf-8")

    df = pd.DataFrame({"sequence": ["ACD", "DA"]})
    enc = PhysicochemicalEncoder(
        dataset=df,
        sequence_column="sequence",
        max_length=5,
        type_descriptor=type_descriptor,
        name_property="ANDN920101",
        debug=True,
    )
    enc.run_process()
    X = enc.coded_dataset
    # shape = max_length features + sequence
    assert X.shape == (2, 5 + 1)
    # First sequence "ACD" → values [1.0, 2.0, 3.0, 0, 0]
    row0 = X.iloc[0, :5].tolist()
    assert row0[:3] == [1.0, 2.0, 3.0]
    assert row0[3:] == [0.0, 0.0]


def test_physicochemical_raises_on_unknown_property(monkeypatch, tmp_path):
    # Same cache stubbing as above
    class _CachePaths:
        def data(self):
            return str(tmp_path / "data")

    class _Cfg:
        cache_paths = _CachePaths()

    from sylphy.core import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config", lambda: _Cfg(), raising=True)

    # prepare a valid aaindex.csv
    cache_dir = Path(_Cfg().cache_paths.data()) / "aaindex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "aaindex.csv").write_text("res,PROP\nA,1\n", encoding="utf-8")

    df = pd.DataFrame({"sequence": ["A"]})
    with pytest.raises(ValueError):
        PhysicochemicalEncoder(dataset=df, name_property="NON_EXISTENT")
