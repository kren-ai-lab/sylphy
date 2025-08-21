# tests/embedding_extraction/conftest.py
from __future__ import annotations

import types
import sys
import os
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, List

import torch
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _quiet_and_temp_logs(tmp_path, monkeypatch) -> Iterator[None]:
    # Keep logs out of user HOME and avoid console spam
    for k in list(os.environ.keys()):
        if k.startswith("PR_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PR_LOG_FILE", str(tmp_path / "embeddings.log"))
    yield


# ----------------------------
# Fake transformers stack
# ----------------------------

class _FakeConfig:
    @classmethod
    def from_pretrained(cls, local_dir: str, trust_remote_code: bool = False):
        # Return a dumb object; the base doesn't use it further
        obj = cls()
        obj.hidden_size = 4
        return obj


class _FakeTokenizer:
    pad_token_id: int = 0
    pad_token: str = "[PAD]"
    eos_token: str = "[EOS]"

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, local_dir: str, do_lower_case: bool = False,
                        use_fast: bool = True, trust_remote_code: bool = False):
        return cls()

    def add_special_tokens(self, mapping: Dict[str, str]) -> None:
        if "pad_token" in mapping:
            self.pad_token = mapping["pad_token"]
            self.pad_token_id = 0  # keep 0 as PAD

    def __call__(self,
                 sequences: List[str],
                 return_tensors: str = "pt",
                 truncation: bool = True,
                 padding: bool = True,
                 add_special_tokens: bool = False,
                 max_length: int = 1024) -> Dict[str, torch.Tensor]:
        # Simple char->id mapping: A..Z -> 1..26 (PAD=0)
        def encode(s: str) -> List[int]:
            ids = [max(1, (ord(ch.upper()) - 64)) for ch in s]  # A->1 ...
            if truncation:
                ids = ids[:max_length]
            return ids

        batch_ids = [encode(s) for s in sequences]
        max_len = max(len(x) for x in batch_ids) if padding else max(len(x) for x in batch_ids)
        max_len = min(max_len, max_length)

        padded: List[List[int]] = []
        mask: List[List[int]] = []
        for ids in batch_ids:
            ids = ids[:max_len]
            pad = [self.pad_token_id] * (max_len - len(ids))
            padded.append(ids + pad)
            mask.append([1] * len(ids) + [0] * len(pad))

        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(mask, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _FakeModel:
    # Control OOM behavior in tests by setting this class var
    OOM_THRESHOLD: int | None = None  # if set (e.g., 2), raise OOM when batch > threshold
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
            # Simulate CUDA OOM regardless of actual device
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        # Return a deterministic tensor: ones * index to keep it simple
        H = 4
        last_hidden = torch.arange(1, H + 1, dtype=torch.float32).view(1, 1, H).repeat(B, L, 1)
        return _FakeModelOutput(last_hidden)


@pytest.fixture(autouse=True)
def _install_fake_transformers(monkeypatch) -> Iterator[None]:
    """
    Install a minimal fake 'transformers' module so tests don't require the real package.
    """
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = _FakeTokenizer
    mod.AutoModel = _FakeModel
    mod.AutoConfig = _FakeConfig

    # Insert into sys.modules
    monkeypatch.setitem(sys.modules, "transformers", mod)

    yield

    # Cleanup counters between tests
    _FakeModel.OOM_THRESHOLD = None
    _FakeModel.FORWARD_CALLS = 0


@pytest.fixture(autouse=True)
def _stub_resolve_model(tmp_path, monkeypatch):
    """
    Stub the registry resolver to return a local temp directory so the base class
    can "load" from it without network.
    """
    # Create an empty 'model dir'
    model_dir = tmp_path / "fake_model_dir"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    from protein_representation.core import model_registry as reg
    monkeypatch.setattr(reg, "resolve_model", lambda name: model_dir, raising=True)
    yield


@pytest.fixture
def toy_df() -> pd.DataFrame:
    return pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
