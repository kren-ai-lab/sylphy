# tests/cli/conftest.py
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Iterator, Dict, List

import pandas as pd
import pytest
import torch


@pytest.fixture(autouse=True)
def _quiet_logs_and_temp(tmp_path, monkeypatch) -> Iterator[None]:
    for k in list(os.environ.keys()):
        if k.startswith("PR_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PR_LOG_FILE", str(tmp_path / "cli.log"))
    yield


class _FakeConfig:
    @classmethod
    def from_pretrained(cls, local_dir: str, trust_remote_code: bool = False):
        obj = cls()
        obj.hidden_size = 4
        return obj


class _FakeTokenizer:
    pad_token_id: int = 0
    pad_token: str = "[PAD]"
    eos_token: str = "[EOS]"

    @classmethod
    def from_pretrained(cls, local_dir: str, **kwargs):
        return cls()

    def add_special_tokens(self, mapping: Dict[str, str]) -> None:
        if "pad_token" in mapping:
            self.pad_token = mapping["pad_token"]
            self.pad_token_id = 0

    def __call__(self,
                 sequences: List[str],
                 return_tensors: str = "pt",
                 truncation: bool = True,
                 padding: bool = True,
                 add_special_tokens: bool = False,
                 max_length: int = 1024) -> Dict[str, torch.Tensor]:
        def enc(s: str) -> List[int]:
            ids = [max(1, (ord(ch.upper()) - 64)) for ch in s]  # A..Z → 1..26
            return ids[:max_length] if truncation else ids

        batch_ids = [enc(s) for s in sequences]
        max_len = min(max(len(x) for x in batch_ids), max_length) if padding else max(len(x) for x in batch_ids)
        padded, mask = [], []
        for ids in batch_ids:
            ids = ids[:max_len]
            pad = [self.pad_token_id] * (max_len - len(ids))
            padded.append(ids + pad)
            mask.append([1] * len(ids) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _FakeModel:
    OOM_THRESHOLD: int | None = None
    FORWARD_CALLS: int = 0

    @classmethod
    def from_pretrained(cls, local_dir: str, trust_remote_code: bool = False):
        return cls()

    def to(self, device: torch.device):  # no-op
        return self

    def eval(self):  # no-op
        return self

    def __call__(self, **enc) -> _FakeModelOutput:
        _FakeModel.FORWARD_CALLS += 1
        input_ids: torch.Tensor = enc["input_ids"]
        B, L = input_ids.shape
        if self.OOM_THRESHOLD is not None and B > int(self.OOM_THRESHOLD):
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        H = 4
        last_hidden = torch.arange(1, H + 1, dtype=torch.float32).view(1, 1, H).repeat(B, L, 1)
        return _FakeModelOutput(last_hidden)


@pytest.fixture(autouse=True)
def _install_fake_transformers(monkeypatch) -> Iterator[None]:

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = _FakeTokenizer
    mod.AutoModel = _FakeModel
    mod.AutoConfig = _FakeConfig
    monkeypatch.setitem(sys.modules, "transformers", mod)

    yield

    _FakeModel.OOM_THRESHOLD = None
    _FakeModel.FORWARD_CALLS = 0


@pytest.fixture(autouse=True)
def _stub_model_registry(tmp_path, monkeypatch):

    model_dir = tmp_path / "fake_model_dir"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    from protein_representation.core import model_registry as reg
    monkeypatch.setattr(reg, "resolve_model", lambda name: model_dir, raising=True)
    yield



@pytest.fixture(autouse=True)
def _stub_constants(monkeypatch):

    import sys, types
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
