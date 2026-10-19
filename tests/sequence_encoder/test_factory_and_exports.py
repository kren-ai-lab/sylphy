from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from sylphy.constants.tool_constants import BASE_URL_AAINDEX
from sylphy.sequence_encoder import (
    FFTEncoder,
    FrequencyEncoder,
    KMerEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PhysicochemicalEncoder,
    create_encoder,
)


@pytest.fixture
def mock_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect AAIndex cache to a temporary directory."""

    class _CachePaths:
        def data(self: object) -> str:
            return str(tmp_path / "data")

    class _Cfg:
        cache_paths = _CachePaths()

    from sylphy.sequence_encoder import physicochemical_encoder

    monkeypatch.setattr(physicochemical_encoder, "get_config", _Cfg, raising=True)
    return tmp_path / "data"


def test_factory_known_aliases(toy_df: pl.DataFrame, mock_cache: Path) -> None:
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
        "kmers": KMerEncoder,
        "kmer": KMerEncoder,
        "tfidf": KMerEncoder,
        "physicochemical": PhysicochemicalEncoder,
        "physchem": PhysicochemicalEncoder,
        "aaindex": PhysicochemicalEncoder,
        "fft": FFTEncoder,
    }
    for key, cls in cases.items():
        enc = create_encoder(key, dataset=toy_df, max_length=8)
        assert isinstance(enc, cls)


def test_factory_unknown_raises() -> None:
    """Verify factory raises ValueError for unknown encoder names."""
    with pytest.raises(ValueError, match=r"Unknown encoder 'nope'"):
        create_encoder("nope", dataset=pl.DataFrame({"sequence": ["AAA"]}))


def test_export_encoder_matches_cli_format(tmp_path: Path) -> None:
    """API export is keyed by id and carries no sequence column."""
    df = pl.DataFrame({"id": ["a", "b"], "sequence": ["ACD", "WYV"]})
    enc = create_encoder("ordinal", dataset=df, max_length=4)
    enc.run_process()
    out = tmp_path / "o.csv"
    enc.export_encoder(out)
    got = pl.read_csv(out)
    assert got.columns[0] == "id"
    assert "sequence" not in got.columns
    assert got["id"].to_list() == ["a", "b"]


def test_export_encoder_rejects_bad_ids(tmp_path: Path) -> None:
    """Missing or duplicated ids raise the library's DataError."""
    from sylphy.exceptions import DataError

    kw = {"sequence_column": "sequence", "max_length": 4}
    for df in (
        pl.DataFrame({"sequence": ["ACD", "WYV"]}),
        pl.DataFrame({"id": ["x", "x"], "sequence": ["ACD", "WYV"]}),
    ):
        enc = OrdinalEncoder(dataset=df, **kw)
        enc.run_process()
        with pytest.raises(DataError):
            enc.export_encoder(tmp_path / "o.csv")
