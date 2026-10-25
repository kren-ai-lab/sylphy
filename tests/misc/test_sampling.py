"""Stratified sampling in ``UtilsLib.random_selection``."""

from __future__ import annotations

import os
import subprocess
import sys

import polars as pl
import pytest

from sylphy.misc.utils_lib import UtilsLib

DF = pl.DataFrame({"id": [f"s{i}" for i in range(10)], "label": ["a"] * 7 + ["b"] * 3})

_SCRIPT = """
import polars as pl
from sylphy.misc.utils_lib import UtilsLib
df = pl.DataFrame({"id": [f"s{i}" for i in range(10)], "label": ["a"] * 7 + ["b"] * 3})
out = UtilsLib.random_selection(df, "label", n_samples=5, per_label=%s, replace=True, random_state=7)
print(out["id"].to_list())
"""


@pytest.mark.parametrize("per_label", [True, False])
def test_seeded_sampling_with_replacement_is_reproducible_across_processes(*, per_label: bool) -> None:
    def run(hash_seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        code = _SCRIPT % per_label
        return subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        ).stdout

    assert run("1") == run("2")


def test_per_label_with_replacement_counts() -> None:
    out = UtilsLib.random_selection(DF, "label", n_samples=5, per_label=True, replace=True, random_state=7)
    assert out.columns == DF.columns
    assert out["label"].value_counts().sort("label")["count"].to_list() == [5, 5]
    assert set(out.filter(pl.col("label") == "b")["id"]) <= {"s7", "s8", "s9"}


def test_global_with_replacement_counts() -> None:
    out = UtilsLib.random_selection(DF, "label", n_samples=10, per_label=False, replace=True, random_state=7)
    assert out["label"].value_counts().sort("label")["count"].to_list() == [7, 3]
