from __future__ import annotations
"""
OOM backoff: simulate CUDA OOM via the fake model and verify automatic halving.
"""

import pandas as pd

from sylphy.embedding_extractor import EmbeddingFactory
from transformers import AutoModel as _FakeModel  # defined in conftest


def test_cuda_oom_backoff_retries_and_succeeds():
    """
    Simulate CUDA OOM when batch > 2 and verify run_process halves batch size and succeeds.
    NOTE: Backoff is enabled only when device.type == "cuda", so we pass name_device="cuda".
    Our fake model's .to(...) is a no-op, so this is safe without a real GPU.
    """
    _FakeModel.OOM_THRESHOLD = 2
    _FakeModel.FORWARD_CALLS = 0

    df = pd.DataFrame({"sequence": ["AAAA", "BBBB", "CCCCC", "DD"]})
    inst = EmbeddingFactory(
        model_name="facebook/esm2_t6_8M_UR50D",
        dataset=df,
        column_seq="sequence",
        name_device="cuda",  # <-- important for backoff path
        debug=True,
        debug_mode=10,
        precision="fp32",
        oom_backoff=True,
    )
    inst.load_hf_tokenizer_and_model()

    # Start with a too-large batch size (4). OOM should trigger and back off to 2.
    inst.run_process(max_length=32, batch_size=4, layers="last", layer_agg="mean", pool="mean")

    X = inst.coded_dataset
    assert X.shape == (4, 5)  # 4 sequences; hidden size 4 + 'sequence'
    assert _FakeModel.FORWARD_CALLS >= 2  # one failed attempt + retries at bs=2
