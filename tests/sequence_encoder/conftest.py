# tests/sequence_encoder/conftest.py
from __future__ import annotations

import os
import sys
import types
from typing import Iterator

import pandas as pd
import pytest

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch) -> Iterator[None]:
    """
    - Keep logs inside tmp.
    - Remove any PR_LOG_* residue between tests.
    """
    for k in list(os.environ.keys()):
        if k.startswith("PR_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PR_LOG_FILE", str(tmp_path / "seqenc.log"))
    yield


@pytest.fixture(autouse=True)
def _stub_constants(monkeypatch):
    """
    Provide a predictable amino-acid mapping for tests.
    If protein_representation.misc.constants.Constant exists, override its attributes.
    If not, create a minimal stub module so imports succeed.
    """
    residues = list("ACDEFGHIKLMNPQRSTVWY")
    pos = {aa: i for i, aa in enumerate(residues)}
    # Create or patch the module
    mod_name = "protein_representation.misc.constants"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        pkg = types.ModuleType("protein_representation.misc")
        sys.modules["protein_representation.misc"] = pkg
        mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = mod

    class Constant:  # minimal surface needed by encoders
        LIST_RESIDUES = residues
        POSITION_RESIDUES = pos
        BASE_URL_AAINDEX = "https://example.com/aaindex.csv"
        BASE_URL_CLUSTERS_DESCRIPTORS = "https://example.com/groups.csv"

    monkeypatch.setattr(mod, "Constant", Constant, raising=False)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    # Canonical short sequences; include variability
    return pd.DataFrame(
        {"sequence": ["ACD", "WYYVV", "KLMNPQ", "GGG"]}  # lengths: 3,5,6,3
    )
