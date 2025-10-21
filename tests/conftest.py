"""Shared fake modules and fixtures for all test suites."""
from __future__ import annotations

import sys
import types
from typing import Dict, Iterator, List, Tuple

import pytest
import torch


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
        **kwargs,
    ):
        return cls()

    def add_special_tokens(self, mapping: Dict[str, str]) -> None:
        if "pad_token" in mapping:
            self.pad_token = mapping["pad_token"]
            self.pad_token_id = 0

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
    def __init__(
        self, last_hidden_state: torch.Tensor, hidden_states: Tuple[torch.Tensor, ...] | None = None
    ):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states or (last_hidden_state,)


class _FakeModel:
    """Fake HF model with H=4, OOM simulation, and forward call tracking."""

    OOM_THRESHOLD: int | None = None
    FORWARD_CALLS: int = 0

    def __init__(self):
        self._device = torch.device("cpu")

    @classmethod
    def from_pretrained(cls, local_dir: str, trust_remote_code: bool = False):
        return cls()

    def to(self, device):
        if isinstance(device, str):
            self._device = torch.device(device)
        else:
            self._device = device
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
        base = torch.arange(1, H + 1, dtype=torch.float32, device=x.device).view(1, 1, H).repeat(B, L, 1)
        hidden_states = tuple(base + float(i) for i in range(4))  # 4 layers
        last_hidden = hidden_states[-1]
        return _FakeOutput(last_hidden, hidden_states)


_fake_tf = types.ModuleType("transformers")
_fake_tf.AutoTokenizer = _FakeTokenizer
_fake_tf.AutoModel = _FakeModel
_fake_tf.AutoConfig = _FakeConfig
_fake_tf.T5Tokenizer = _FakeTokenizer
_fake_tf.T5EncoderModel = _FakeModel
_fake_tf.BertModel = _FakeModel
_fake_tf.AutoModelForCausalLM = _FakeModel
_fake_tf.EsmTokenizer = _FakeTokenizer


class PreTrainedTokenizerFast:
    pass


_fake_tf.PreTrainedTokenizerFast = PreTrainedTokenizerFast

sys.modules["transformers"] = _fake_tf


_fake_esm = types.ModuleType("esm")
_fake_esm.__path__ = []

_fake_esm_models = types.ModuleType("esm.models")
_fake_esm_models.__path__ = []
_fake_esm_models_esmc = types.ModuleType("esm.models.esmc")


class ESMC:
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
        base = torch.arange(1, H + 1, dtype=torch.float32, device=x.device).view(1, 1, H).repeat(B, L, 1)
        hidden_states = tuple(base + float(i) for i in range(4))
        last_hidden = hidden_states[-1]

        class _Out:
            last_hidden_state = last_hidden
            hidden_states = hidden_states

        return _Out()


_fake_esm_models_esmc.ESMC = ESMC

_fake_esm_sdk = types.ModuleType("esm.sdk")
_fake_esm_sdk.__path__ = []
_fake_esm_sdk_api = types.ModuleType("esm.sdk.api")
_fake_esm_sdk_forge = types.ModuleType("esm.sdk.forge")


class ESMProtein:
    pass


class LogitsConfig:
    pass


class ESM3ForgeInferenceClient:
    pass


_fake_esm_sdk_api.ESMProtein = ESMProtein
_fake_esm_sdk_api.LogitsConfig = LogitsConfig
_fake_esm_sdk_forge.ESM3ForgeInferenceClient = ESM3ForgeInferenceClient

sys.modules["esm"] = _fake_esm
sys.modules["esm.models"] = _fake_esm_models
sys.modules["esm.models.esmc"] = _fake_esm_models_esmc
sys.modules["esm.sdk"] = _fake_esm_sdk
sys.modules["esm.sdk.api"] = _fake_esm_sdk_api
sys.modules["esm.sdk.forge"] = _fake_esm_sdk_forge


@pytest.fixture(autouse=True)
def _reset_fake_model_state() -> Iterator[None]:
    """Reset fake model state between tests."""
    _FakeModel.OOM_THRESHOLD = None
    _FakeModel.FORWARD_CALLS = 0
    yield
    _FakeModel.OOM_THRESHOLD = None
    _FakeModel.FORWARD_CALLS = 0
