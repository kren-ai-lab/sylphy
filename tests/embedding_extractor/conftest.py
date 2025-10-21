from __future__ import annotations

"""
Test bootstrap for embedding_extractor tests.

- Injects a fake `transformers` module (with all aliases needed by backends).
- Injects a fake `esm` package exposing:
    - esm.models.esmc.ESMC
    - esm.sdk.api.ESMProtein, esm.sdk.api.LogitsConfig
    - esm.sdk.forge.ESM3ForgeInferenceClient
- Stubs model resolution to a temporary directory (no network).
- Keeps logging quiet and confined to tmp.
"""

import os
import sys
import types
from typing import Dict, Iterator, List, Tuple

import pytest
import torch

# ----------------------------
# Fake Transformers (installed before importing the code under test)
# ----------------------------


class _FakeConfig:
    hidden_size: int = 4

    @classmethod
    def from_pretrained(cls, local_dir: str, trust_remote_code: bool = False):
        return cls()


class _FakeTokenizer:
    pad_token_id: int = 0
    pad_token: str = "[PAD]"
    eos_token: str = "[EOS]"

    @classmethod
    def from_pretrained(
        cls,
        local_dir: str,
        do_lower_case: bool = False,
        use_fast: bool = True,
        trust_remote_code: bool = False,
    ):
        return cls()

    def add_special_tokens(self, mapping: Dict[str, str]) -> None:
        if "pad_token" in mapping:
            self.pad_token = mapping["pad_token"]
            self.pad_token_id = 0  # PAD=0

    def get_vocab(self) -> Dict[str, int]:
        return {"<cls>": 1, self.pad_token: 0, self.eos_token: 2}

    def __call__(
        self,
        sequences: List[str],
        return_tensors: str = "pt",
        truncation: bool = True,
        padding: bool = True,
        add_special_tokens: bool = True,
        max_length: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        def encode(s: str) -> List[int]:
            ids = [max(1, (ord(ch.upper()) - 64)) for ch in s if s and ch.strip()]
            return ids[:max_length] if truncation else ids

        sequences = sequences or [""]
        batch_ids = [encode(s) for s in sequences]
        max_len = min(max((len(x) for x in batch_ids), default=1), max_length)
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


class _FakeOutput:
    def __init__(self, last_hidden_state: torch.Tensor, hidden_states: Tuple[torch.Tensor, ...]):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _FakeModel:
    """
    Tiny fake HF-like model:
    - hidden size H=4
    - exposes 4 hidden layers in `hidden_states`
    - supports simulated CUDA OOM via class attribute `OOM_THRESHOLD`
    """

    OOM_THRESHOLD: int | None = None
    FORWARD_CALLS: int = 0

    @classmethod
    def from_pretrained(cls, local_dir: str, trust_remote_code: bool = False):
        return cls()

    def to(self, device: torch.device):
        return self

    def eval(self):
        return self

    def __call__(self, **enc) -> _FakeOutput:
        _FakeModel.FORWARD_CALLS += 1
        x: torch.Tensor = enc["input_ids"]
        B, L = x.shape
        if self.OOM_THRESHOLD is not None and B > int(self.OOM_THRESHOLD):
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        H = 4
        # <<< FIX: create on the same device as the input >>>
        base = torch.arange(1, H + 1, dtype=torch.float32, device=x.device).view(1, 1, H).repeat(B, L, 1)
        hidden_states = tuple(base + float(i) for i in range(4))  # 4 layers
        last_hidden = hidden_states[-1]
        return _FakeOutput(last_hidden, hidden_states)


# Build and inject the fake `transformers` module
_fake_tf = types.ModuleType("transformers")
_fake_tf.AutoTokenizer = _FakeTokenizer
_fake_tf.AutoModel = _FakeModel
_fake_tf.AutoConfig = _FakeConfig
# Aliases needed by backends
_fake_tf.T5Tokenizer = _FakeTokenizer
_fake_tf.T5EncoderModel = _FakeModel
_fake_tf.BertModel = _FakeModel
_fake_tf.AutoModelForCausalLM = _FakeModel
_fake_tf.EsmTokenizer = _FakeTokenizer


# Some libs import this symbol directly (ensure it exists)
class PreTrainedTokenizerFast:  # thin placeholder
    pass


_fake_tf.PreTrainedTokenizerFast = PreTrainedTokenizerFast

sys.modules["transformers"] = _fake_tf


# ----------------------------
# Fake ESM package (make it a real package with __path__)
# ----------------------------

_fake_esm = types.ModuleType("esm")
_fake_esm.__path__ = []  # mark as package

# esm.models and esm.models.esmc
_fake_esm_models = types.ModuleType("esm.models")
_fake_esm_models.__path__ = []
_fake_esm_models_esmc = types.ModuleType("esm.models.esmc")


class ESMC:
    """Minimal ESMC stub with HF-like call signature."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **enc):
        x: torch.Tensor = enc["input_ids"]
        B, L = x.shape
        H = 4
        # <<< FIX: same device as the input >>>
        base = torch.arange(1, H + 1, dtype=torch.float32, device=x.device).view(1, 1, H).repeat(B, L, 1)
        hidden_states = tuple(base + float(i) for i in range(4))
        last_hidden = hidden_states[-1]

        class _Out:
            last_hidden_state = last_hidden
            hidden_states = hidden_states

        return _Out()


_fake_esm_models_esmc.ESMC = ESMC

# esm.sdk, esm.sdk.api, esm.sdk.forge
_fake_esm_sdk = types.ModuleType("esm.sdk")
_fake_esm_sdk.__path__ = []
_fake_esm_sdk_api = types.ModuleType("esm.sdk.api")
_fake_esm_sdk_forge = types.ModuleType("esm.sdk.forge")


# Minimal API classes used by your backends
class ESMProtein: ...


class LogitsConfig: ...


class ESM3ForgeInferenceClient: ...


_fake_esm_sdk_api.ESMProtein = ESMProtein
_fake_esm_sdk_api.LogitsConfig = LogitsConfig
_fake_esm_sdk_forge.ESM3ForgeInferenceClient = ESM3ForgeInferenceClient

# Register esm package and submodules
sys.modules["esm"] = _fake_esm
sys.modules["esm.models"] = _fake_esm_models
sys.modules["esm.models.esmc"] = _fake_esm_models_esmc
sys.modules["esm.sdk"] = _fake_esm_sdk
sys.modules["esm.sdk.api"] = _fake_esm_sdk_api
sys.modules["esm.sdk.forge"] = _fake_esm_sdk_forge


# ----------------------------
# Pytest fixtures (env & resolver)
# ----------------------------


@pytest.fixture(autouse=True)
def _quiet_logs(tmp_path, monkeypatch) -> Iterator[None]:
    """Confine logging to tmp and disable any pre-existing SYLPHY_LOG_* env vars."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "embeddings.log"))
    yield


@pytest.fixture(autouse=True)
def _stub_resolve_model(tmp_path, monkeypatch) -> Iterator[None]:
    """
    Make the model resolver return a local temp directory so backends can load
    without network or real artifacts.
    """
    from sylphy.core import model_registry as reg

    local = tmp_path / "fake_model_dir"
    local.mkdir(parents=True, exist_ok=True)
    (local / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reg, "resolve_model", lambda name: local, raising=True)
    yield
    _FakeModel.OOM_THRESHOLD = None
    _FakeModel.FORWARD_CALLS = 0
