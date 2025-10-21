from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sylphy.constants.tool_constants import BASE_URL_AAINDEX
from sylphy.sequence_encoder import PhysicochemicalEncoder


def test_physicochemical_reads_cached_file(monkeypatch, tmp_path):
    """
    Avoid network: create the cached AAIndex CSV that the encoder expects.
    The encoder should load from cache and not attempt to download.
    """

    # Point `get_config().cache_paths.data()` to a temp directory
    class _CachePaths:
        def data(self):  # matches usage in encoder
            return str(tmp_path / "data")

    class _Cfg:
        cache_paths = _CachePaths()

    # Patch get_config where it's imported in the encoder module
    from sylphy.sequence_encoder import physicochemical_encoder

    monkeypatch.setattr(physicochemical_encoder, "get_config", lambda: _Cfg(), raising=True)

    # Build the exact filepath the encoder will use:
    # <data>/<type_descriptor>/<basename(BASE_URL_AAINDEX)>
    type_descriptor = "aaindex"
    filename = Path(BASE_URL_AAINDEX).name  # e.g., "aaindex_encoders.csv"
    cache_dir = Path(_Cfg().cache_paths.data()) / type_descriptor
    cache_dir.mkdir(parents=True, exist_ok=True)
    filepath = cache_dir / filename

    # Minimal CSV: residue column and one property column
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

    # Patch get_config where it's imported in the encoder module
    from sylphy.sequence_encoder import physicochemical_encoder

    monkeypatch.setattr(physicochemical_encoder, "get_config", lambda: _Cfg(), raising=True)

    # Prepare a valid aaindex CSV with a different property name
    cache_dir = Path(_Cfg().cache_paths.data()) / "aaindex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / Path(BASE_URL_AAINDEX).name).write_text("res,PROP\nA,1\n", encoding="utf-8")

    df = pd.DataFrame({"sequence": ["A"]})
    with pytest.raises(ValueError):
        PhysicochemicalEncoder(dataset=df, name_property="NON_EXISTENT")
