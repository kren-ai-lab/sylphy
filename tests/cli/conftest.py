# tests/cli/conftest.py
from __future__ import annotations

import os
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _quiet_logs_and_temp(tmp_path, monkeypatch) -> Iterator[None]:
    for k in list(os.environ.keys()):
        if k.startswith("PR_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PR_LOG_FILE", str(tmp_path / "cli.log"))
    yield


@pytest.fixture(autouse=True)
def _stub_model_registry(tmp_path, monkeypatch):
    model_dir = tmp_path / "fake_model_dir"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    from sylphy.core import model_registry as reg

    monkeypatch.setattr(reg, "resolve_model", lambda name: model_dir, raising=True)
    yield


@pytest.fixture(autouse=True)
def _stub_constants(monkeypatch):
    import sys
    import types

    residues = list("ACDEFGHIKLMNPQRSTVWY")
    pos = {aa: i for i, aa in enumerate(residues)}
    mod_name = "protein_representation.misc.constants"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        pkg = types.ModuleType("protein_representation.misc")
        sys.modules["protein_representation.misc"] = pkg
        mod = types.ModuleType(mod_name)
        sys.modules[mod_name] = mod

    class Constant:
        LIST_RESIDUES = residues
        POSITION_RESIDUES = pos
        BASE_URL_AAINDEX = "https://example.com/aaindex.csv"
        BASE_URL_CLUSTERS_DESCRIPTORS = "https://example.com/groups.csv"

    monkeypatch.setattr(mod, "Constant", Constant, raising=False)
