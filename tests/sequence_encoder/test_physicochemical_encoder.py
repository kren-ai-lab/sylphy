from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sylphy.constants.tool_constants import BASE_URL_AAINDEX
from sylphy.sequence_encoder import PhysicochemicalEncoder


@pytest.fixture
def mock_cache(monkeypatch, tmp_path):
    """Redirect AAIndex cache to a temporary directory."""

    class _CachePaths:
        def data(self):
            return str(tmp_path / "data")

    class _Cfg:
        cache_paths = _CachePaths()

    from sylphy.sequence_encoder import physicochemical_encoder

    monkeypatch.setattr(physicochemical_encoder, "get_config", lambda: _Cfg(), raising=True)
    return tmp_path / "data"


def test_physicochemical_reads_cached_file(mock_cache):
    """Verify encoder loads from cached AAIndex file without network access."""
    cache_dir = mock_cache / "aaindex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filepath = cache_dir / Path(BASE_URL_AAINDEX).name

    filepath.write_text("res,ANDN920101\nA,1.0\nC,2.0\nD,3.0\n", encoding="utf-8")

    df = pd.DataFrame({"sequence": ["ACD", "DA"]})
    enc = PhysicochemicalEncoder(
        dataset=df,
        sequence_column="sequence",
        max_length=5,
        type_descriptor="aaindex",
        name_property="ANDN920101",
    )
    enc.run_process()

    assert enc.coded_dataset.shape == (2, 6)
    assert enc.coded_dataset.iloc[0, :5].tolist() == [1.0, 2.0, 3.0, 0.0, 0.0]


def test_physicochemical_raises_on_unknown_property(mock_cache):
    cache_dir = mock_cache / "aaindex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / Path(BASE_URL_AAINDEX).name).write_text("res,PROP\nA,1\n", encoding="utf-8")

    df = pd.DataFrame({"sequence": ["A"]})
    with pytest.raises(ValueError):
        PhysicochemicalEncoder(dataset=df, name_property="NON_EXISTENT")
