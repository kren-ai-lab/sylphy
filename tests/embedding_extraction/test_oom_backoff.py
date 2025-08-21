# tests/embedding_extraction/test_oom_backoff.py
from __future__ import annotations

import pandas as pd

from protein_representation.embedding_extractor import EmbeddingFactory

# Access the fake model to set its OOM behavior
from transformers import AutoModel as _FakeModel  # our conftest-installed stub


def test_cuda_oom_backoff_retries_and_succeeds(monkeypatch):
    # Configure fake model to raise OOM when batch > 2
    _FakeModel.OOM_THRESHOLD = 2
    _FakeModel.FORWARD_CALLS = 0

    df = pd.DataFrame({"sequence": ["AAAA", "BBBB", "CCCCC", "DD"]})
    inst = EmbeddingFactory(
        model_name="facebook/esm2_t6_8M_UR50D",
        dataset=df,
        column_seq="sequence",
        name_device="cpu",  # CPU is fine; we just simulate OOM via fake model
        debug=True,
        debug_mode=10,
        precision="fp32",
        oom_backoff=True,
    )
    inst.load_hf_tokenizer_and_model()

    # Call encode directly with an initially too-large batch size (4)
    out = inst.encode_batch_last_hidden(df["sequence"].tolist(), max_length=32, batch_size=4, pool="mean")
    # Should succeed via retry (batch halves to 2) without raising
    assert out.shape == (4, 4)
    # At least 2 calls (one failed at bs=4, one succeeded at bs=2)
    assert _FakeModel.FORWARD_CALLS >= 2
