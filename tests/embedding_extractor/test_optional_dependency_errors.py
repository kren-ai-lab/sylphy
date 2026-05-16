from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from sylphy.embedding_extractor.prot5_based import ProtT5Embedding


def test_prot5_missing_sentencepiece_suggests_embeddings_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"sequence": ["AAAA"]})
    inst = ProtT5Embedding(dataset=df, name_device="cpu")
    monkeypatch.setattr(inst, "_register_and_resolve", lambda: "fake-model-dir")

    def _boom(*_args: object, **_kwargs: object) -> object:
        msg = "T5Tokenizer requires the SentencePiece library but it was not found in your environment."
        raise ImportError(
            msg,
        )

    tokenizer_cls = cast("Any", __import__("transformers")).T5Tokenizer
    monkeypatch.setattr(tokenizer_cls, "from_pretrained", _boom)

    with pytest.raises(ImportError, match=r"sylphy\[embeddings\]"):
        inst.load_model_tokenizer()
