"""Shared ecosystem CLI contract: record ids, ``--config``, ``validate`` and provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from sylphy.cli._shared import snapshot_commit
from sylphy.cli.embed import app as embed_app
from sylphy.cli.encode import app as encode_app
from sylphy.cli.main import app as main_app

SCHEMAS = Path(__file__).parent.parent / "contract"
ENCODE = ["--method", "ordinal", "--max-length", "5"]


def _schema(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text(encoding="utf-8")))


def _csv(tmp_path: Path, ids: list[str], col: str = "id") -> Path:
    inp = tmp_path / "in.csv"
    pl.DataFrame({col: ids, "sequence": ["ACD", "WYYVV", "KLMNP"][: len(ids)]}).write_csv(inp)
    return inp


def _plain(text: str) -> str:
    """Strip ANSI codes, Rich borders and whitespace so assertions ignore CI colour and width."""
    return re.sub(r"\x1b\[[0-9;]*m|[│╭╮╰╯─]|\s", "", text)


def _encode(inp: Path, out: Path, *extra: str) -> object:
    return CliRunner().invoke(encode_app, [*ENCODE, "-i", str(inp), "-o", str(out), *extra])


def test_csv_ids_preserved_in_order(tmp_path: Path) -> None:
    out = tmp_path / "o.csv"
    assert _encode(_csv(tmp_path, ["007", "b", "a"]), out).exit_code == 0  # type: ignore[attr-defined]
    got = pl.read_csv(out, infer_schema=False)
    assert got.columns[0] == "id"
    assert got["id"].to_list() == ["007", "b", "a"]
    assert got.width == 1 + 5  # ordinal: max_length features, no sequence column


def test_fasta_ids_first_token(tmp_path: Path) -> None:
    inp = tmp_path / "in.fasta"
    inp.write_text(">p1 some description\nACD\nEF\n>p2\nWYY\n", encoding="utf-8")
    out = tmp_path / "o.parquet"
    assert _encode(inp, out).exit_code == 0  # type: ignore[attr-defined]
    assert pl.read_parquet(out)["id"].to_list() == ["p1", "p2"]


def test_custom_id_col(tmp_path: Path) -> None:
    out = tmp_path / "o.csv"
    assert _encode(_csv(tmp_path, ["x", "y"], col="name"), out, "--id-col", "name").exit_code == 0  # type: ignore[attr-defined]
    assert pl.read_csv(out)["id"].to_list() == ["x", "y"]


@pytest.mark.parametrize(
    ("ids", "col", "needle"),
    [(["a", "b"], "name", "--id-col"), (["a", "a"], "id", "Duplicate"), (["a", ""], "id", "Empty")],
)
def test_bad_ids_exit_2(tmp_path: Path, ids: list[str], col: str, needle: str) -> None:
    res = _encode(_csv(tmp_path, ids, col), tmp_path / "o.csv")
    assert res.exit_code == 2  # type: ignore[attr-defined]
    assert needle in _plain(res.output)  # type: ignore[attr-defined]


def test_npz_and_npy_carry_ids(tmp_path: Path) -> None:
    inp = _csv(tmp_path, ["a", "b", "c"])
    npz = tmp_path / "o.npz"
    assert _encode(inp, npz).exit_code == 0  # type: ignore[attr-defined]
    with np.load(npz, allow_pickle=False) as z:
        assert z["ids"].tolist() == ["a", "b", "c"]
        assert z["columns"].shape[0] == z["values"].shape[1] == 5
    npy = tmp_path / "o.npy"
    assert _encode(inp, npy).exit_code == 0  # type: ignore[attr-defined]
    assert np.load(npy, allow_pickle=False).shape == (3, 5)
    assert (tmp_path / "o.ids.txt").read_text(encoding="utf-8").split() == ["a", "b", "c"]


def test_config_precedence(tmp_path: Path) -> None:
    inp = _csv(tmp_path, ["a", "b"])
    cfg = tmp_path / "c.yaml"
    cfg.write_text("method: ordinal\nmax_length: 7\n", encoding="utf-8")
    out = tmp_path / "o.csv"
    base = ["-i", str(inp), "-o", str(out), "--config", str(cfg)]
    assert CliRunner().invoke(encode_app, base).exit_code == 0
    assert pl.read_csv(out).width == 1 + 7  # file overrides default (1024)
    assert CliRunner().invoke(encode_app, [*base, "--max-length", "4"]).exit_code == 0
    assert pl.read_csv(out).width == 1 + 4  # flag overrides file


@pytest.mark.parametrize("body", ["nope: 1\n", "max_length: three\n", "input_path: x\n", "- a\n"])
def test_bad_config_exit_2(tmp_path: Path, body: str) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(body, encoding="utf-8")
    res = CliRunner().invoke(
        encode_app, ["-i", str(_csv(tmp_path, ["a"])), "-o", "o.csv", "--config", str(cfg)]
    )
    assert res.exit_code == 2


@pytest.mark.parametrize(
    ("target", "body", "valid"),
    [
        ("encode", "max_length: 9\nmethod: kmers\n", True),
        ("embed", "revision: abc\nbatch_size: 2\n", True),
        ("encode", "method: nope\nbatch_size: 2\nmax_length: x\n", False),
        ("embed", "batch_size: 0\n", False),
    ],
)
def test_validate_json_conforms(tmp_path: Path, target: str, body: str, valid: bool) -> None:  # noqa: FBT001
    cfg = tmp_path / "c.yaml"
    cfg.write_text(body, encoding="utf-8")
    res = CliRunner().invoke(main_app, ["validate", target, str(cfg), "--json"])
    assert res.exit_code == (0 if valid else 2)
    issues = json.loads(res.stdout)
    _schema("validate-errors.schema.json").validate(issues)
    assert (issues == []) is valid


def test_validate_plain_lists_problems_on_stderr(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("nope: 1\n", encoding="utf-8")
    res = CliRunner().invoke(main_app, ["validate", "encode", str(cfg)])
    assert res.exit_code == 2
    assert "nope: unknown key" in res.output


def test_encode_provenance_conforms(tmp_path: Path) -> None:
    out = tmp_path / "o.csv"
    assert _encode(_csv(tmp_path, ["a", "b"]), out).exit_code == 0  # type: ignore[attr-defined]
    doc = json.loads((tmp_path / "o.csv.provenance.json").read_text(encoding="utf-8"))
    _schema("provenance.schema.json").validate(doc)
    assert doc["command"] == "encode"
    assert doc["parameters"]["max_length"] == 5
    assert "input_path" not in doc["parameters"]
    assert doc["outputs"][0]["n_rows"] == 2
    assert doc["outputs"][0]["n_cols"] == 5


def test_embed_provenance_and_revision(tmp_path: Path) -> None:
    out = tmp_path / "e.npy"
    res = CliRunner().invoke(
        embed_app,
        ["-i", str(_csv(tmp_path, ["a", "b"])), "-o", str(out), "-d", "cpu", "-b", "2", "--revision", "v1"],
    )
    assert res.exit_code == 0, res.output
    doc = json.loads((tmp_path / "e.npy.provenance.json").read_text(encoding="utf-8"))
    _schema("provenance.schema.json").validate(doc)
    ext = doc["extensions"]
    assert ext["revision_requested"] == "v1"
    assert ext["revision_resolved"] == "c0ffee"  # from the stubbed .../snapshots/<commit> dir
    assert ext["device"] == "cpu"
    assert ext["batch_size_effective"] == 2
    assert doc["outputs"][0]["path"] == str(out)


def test_snapshot_commit() -> None:
    assert snapshot_commit("/c/models--o--m/snapshots/abc123") == "abc123"
    assert snapshot_commit("/c/models--o--m/snapshots/abc123/sub") == "abc123"
    assert snapshot_commit("/some/local/dir") is None
    assert snapshot_commit(None) is None


def test_revision_reaches_snapshot_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    from sylphy.cli.embed import _pin_revision
    from sylphy.core import model_registry as reg

    seen: dict[str, object] = {}

    def fake_download(ref: str, revision: str | None = None) -> str:
        seen.update(ref=ref, revision=revision)
        return "/c/models--o--m/snapshots/abc"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    _pin_revision("org/model", "v2")
    reg._resolve_huggingface(reg.get_model_spec("org/model"))  # noqa: SLF001
    reg.unregister("org/model")
    assert seen == {"ref": "org/model", "revision": "v2"}


def test_fft_keeps_ids(tmp_path: Path) -> None:
    out = tmp_path / "f.csv"
    res = CliRunner().invoke(
        encode_app, ["--method", "fft", "-i", str(_csv(tmp_path, ["a", "b", "c"])), "-o", str(out), "-l", "5"]
    )
    assert res.exit_code == 0, res.output
    got = pl.read_csv(out)
    assert got["id"].to_list() == ["a", "b", "c"]
    assert "sequence" not in got.columns
