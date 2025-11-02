from __future__ import annotations

import sys
from typing import Any, cast

import pandas as pd

from sylphy.embedding_extractor import EmbeddingFactory


def test_cuda_oom_backoff_retries_and_succeeds():
    """Verify OOM backoff halves batch size when encountering simulated CUDA OOM errors."""
    transformers_mod = cast(Any, sys.modules["transformers"])
    _FakeModel = transformers_mod.AutoModel
    _FakeModel.OOM_THRESHOLD = 2
    _FakeModel.FORWARD_CALLS = 0

    df = pd.DataFrame({"sequence": ["AAAA", "BBBB", "CCCCC", "DD"]})
    inst = EmbeddingFactory(
        model_name="facebook/esm2_t6_8M_UR50D",
        dataset=df,
        column_seq="sequence",
        name_device="cuda",
        debug=True,
        debug_mode=10,
        precision="fp32",
        oom_backoff=True,
    )
    inst.load_hf_tokenizer_and_model()
    inst.run_process(max_length=32, batch_size=4, layers="last", layer_agg="mean", pool="mean")

    assert inst.coded_dataset.shape == (4, 5)
    assert _FakeModel.FORWARD_CALLS >= 2
