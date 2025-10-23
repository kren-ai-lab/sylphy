from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sylphy.constants.tool_constants import BASE_URL_AAINDEX
from sylphy.sequence_encoder import (
    FFTEncoder,
    FrequencyEncoder,
    KMersEncoders,
    OneHotEncoder,
    OrdinalEncoder,
    PhysicochemicalEncoder,
    create_encoder,
)


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


def test_factory_known_aliases(toy_df, mock_cache):
    """Verify factory creates correct encoder for all known aliases."""
    # Pre-populate cache with minimal AAIndex file to avoid network requests
    cache_dir = mock_cache / "aaindex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    filepath = cache_dir / Path(BASE_URL_AAINDEX).name
    filepath.write_text(
        "res,ANDN920101\nA,1.0\nC,2.0\nD,3.0\nG,4.0\nK,5.0\nL,6.0\nM,7.0\nN,8.0\nP,9.0\nQ,10.0\nW,11.0\nY,12.0\nV,13.0\n",
        encoding="utf-8",
    )

    cases: dict[str, type] = {
        "ordinal": OrdinalEncoder,
        "one_hot": OneHotEncoder,
        "onehot": OneHotEncoder,
        "frequency": FrequencyEncoder,
        "kmers": KMersEncoders,
        "kmer": KMersEncoders,
        "tfidf": KMersEncoders,
        "physicochemical": PhysicochemicalEncoder,
        "physchem": PhysicochemicalEncoder,
        "aaindex": PhysicochemicalEncoder,
        "fft": FFTEncoder,
    }
    for key, cls in cases.items():
        enc = create_encoder(key, dataset=toy_df, max_length=8)
        assert isinstance(enc, cls)


def test_factory_unknown_raises():
    """Verify factory raises ValueError for unknown encoder names."""
    with pytest.raises(ValueError):
        create_encoder("nope", dataset=pd.DataFrame({"sequence": ["AAA"]}))
