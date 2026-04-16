"""Shared fake modules and fixtures for all test suites."""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


class _FakeConfig:
    hidden_size: int = 4

    @classmethod
    def from_pretrained(cls, _local_dir: str, *, trust_remote_code: bool = False) -> _FakeConfig:  # noqa: ARG003
        return cls()


class _FakeTokenizer:
    pad_token_id: int = 0
    pad_token: str = "[PAD]"  # noqa: S105
    eos_token: str = "[EOS]"  # noqa: S105

    @classmethod
    def from_pretrained(
        cls,
        _local_dir: str,
        *,
        do_lower_case: bool = False,  # noqa: ARG003
        use_fast: bool = True,  # noqa: ARG003
        trust_remote_code: bool = False,  # noqa: ARG003
        **kwargs: object,  # noqa: ARG003
    ) -> _FakeTokenizer:
        return cls()

    def add_special_tokens(self, mapping: dict[str, str]) -> None:
        if "pad_token" in mapping:
            self.pad_token = mapping["pad_token"]
            self.pad_token_id = 0

    def get_vocab(self) -> dict[str, int]:
        return {"<cls>": 1, self.pad_token: 0, self.eos_token: 2}

    def __call__(
        self,
        sequences: list[str],
        *,
        return_tensors: str = "pt",  # noqa: ARG002
        truncation: bool = True,
        padding: bool = True,  # noqa: ARG002
        add_special_tokens: bool = True,  # noqa: ARG002
        max_length: int = 1024,
    ) -> dict[str, torch.Tensor]:
        def encode(s: str) -> list[int]:
            ids = [max(1, (ord(ch.upper()) - 64)) for ch in s if s and ch.strip()]
            return ids[:max_length] if truncation else ids

        sequences = sequences or [""]
        batch_ids = [encode(s) for s in sequences]
        max_len = min(max((len(x) for x in batch_ids), default=1), max_length)
        padded, mask = [], []
        for row_ids in batch_ids:
            ids = row_ids[:max_len]
            pad = [self.pad_token_id] * (max_len - len(ids))
            padded.append(ids + pad)
            mask.append([1] * len(ids) + [0] * len(pad))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _FakeOutput:
    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        hidden_states: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states or (last_hidden_state,)


class _FakeModel:
    """Fake HF model with H=4, OOM simulation, and forward call tracking."""

    OOM_THRESHOLD: int | None = None
    FORWARD_CALLS: int = 0

    def __init__(self) -> None:
        pass

    @classmethod
    def from_pretrained(cls, _local_dir: str, *, trust_remote_code: bool = False) -> _FakeModel:  # noqa: ARG003
        return cls()

    def to(self, _device: object) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self

    def __call__(self, **enc: object) -> _FakeOutput:
        _FakeModel.FORWARD_CALLS += 1
        x: torch.Tensor = cast("torch.Tensor", enc["input_ids"])
        B, L = x.shape
        if self.OOM_THRESHOLD is not None and int(self.OOM_THRESHOLD) < B:
            msg = "simulated OOM"
            raise torch.cuda.OutOfMemoryError(msg)
        H = 4
        base = torch.arange(1, H + 1, dtype=torch.float32, device=x.device).view(1, 1, H).repeat(B, L, 1)
        hidden_states = tuple(base + float(i) for i in range(4))  # 4 layers
        last_hidden = hidden_states[-1]
        return _FakeOutput(last_hidden, hidden_states)


_fake_tf = cast("Any", types.ModuleType("transformers"))
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


_fake_esm = cast("Any", types.ModuleType("esm"))
_fake_esm.__path__ = []

_fake_esm_models = cast("Any", types.ModuleType("esm.models"))
_fake_esm_models.__path__ = []
_fake_esm_models_esmc = cast("Any", types.ModuleType("esm.models.esmc"))


class _ESMCOut:
    def __init__(self, last_hidden_state: torch.Tensor, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class ESMC:
    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> ESMC:  # noqa: ARG003
        return cls()

    def to(self, _device: object) -> ESMC:
        return self

    def eval(self) -> ESMC:
        return self

    def __call__(self, **enc: object) -> _ESMCOut:
        x: torch.Tensor = cast("torch.Tensor", enc["input_ids"])
        B, L = x.shape
        H = 4
        base = torch.arange(1, H + 1, dtype=torch.float32, device=x.device).view(1, 1, H).repeat(B, L, 1)
        hidden_states = tuple(base + float(i) for i in range(4))
        last_hidden = hidden_states[-1]

        return _ESMCOut(last_hidden, hidden_states)


_fake_esm_models_esmc.ESMC = ESMC

_fake_esm_sdk = cast("Any", types.ModuleType("esm.sdk"))
_fake_esm_sdk.__path__ = []
_fake_esm_sdk_api = cast("Any", types.ModuleType("esm.sdk.api"))
_fake_esm_sdk_forge = cast("Any", types.ModuleType("esm.sdk.forge"))


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
