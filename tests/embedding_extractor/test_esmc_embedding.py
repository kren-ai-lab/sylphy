"""Tests for ESMCEmbedding: no-ljust truncation and batched GPU tensor accumulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from sylphy.embedding_extractor.esmc_embedding import ESMCEmbedding

if TYPE_CHECKING:
    import pytest


def _make_inst(sequences: list[str], monkeypatch: pytest.MonkeyPatch) -> ESMCEmbedding:
    from esm.models.esmc import ESMC  # type: ignore[attr-defined]

    df = pl.DataFrame({"sequence": sequences})
    inst = ESMCEmbedding(dataset=df, name_device="cpu")
    monkeypatch.setattr(inst, "ensure_loaded", lambda: None)
    monkeypatch.setattr(inst, "release_resources", lambda: None)
    inst.model = ESMC()  # type: ignore[call-arg]  # ty: ignore[missing-argument]
    return inst


def test_embedding_process_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """embedding_process returns (N, H+1) DataFrame with sequence column first."""
    inst = _make_inst(["ACE", "MDEF", "KL"], monkeypatch)
    result = inst.embedding_process(batch_size=4)

    assert result.shape[0] == 3
    assert result.columns[0] == "sequence"
    assert result.shape[1] > 1


def test_seq_len_truncates_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """seq_len truncates sequences but never pads with X."""
    inst = _make_inst(["ACDEFG"], monkeypatch)
    seen: list[str] = []

    original_embed = inst._embed_one  # noqa: SLF001

    def _capture(seq: str, *, return_hidden_states: bool = True) -> object:
        seen.append(seq)
        return original_embed(seq, return_hidden_states=return_hidden_states)

    monkeypatch.setattr(inst, "_embed_one", _capture)
    inst.embedding_process(seq_len=3)

    assert seen == ["ACD"]


def test_process_sequence_returns_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    """_process_sequence returns a float32 torch.Tensor (deferred CPU transfer)."""
    inst = _make_inst(["ACE"], monkeypatch)
    result = inst._process_sequence("ACE", "last", "mean", "mean")  # noqa: SLF001

    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32


def test_embedding_values_consistent_across_batch_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Results are identical with batch_size=1 vs batch_size=4."""
    seqs = ["ACE", "MDEF", "KL", "ACDE"]
    inst1 = _make_inst(seqs, monkeypatch)
    inst2 = _make_inst(seqs, monkeypatch)

    r1 = inst1.embedding_process(batch_size=1).select(pl.exclude("sequence")).to_numpy()
    r2 = inst2.embedding_process(batch_size=4).select(pl.exclude("sequence")).to_numpy()

    np.testing.assert_array_equal(r1, r2)
