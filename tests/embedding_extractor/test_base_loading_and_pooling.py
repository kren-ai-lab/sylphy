from __future__ import annotations
"""
End-to-end checks for loading + pooling + layer selection using the fake HF stack.
"""

import numpy as np
import pandas as pd

from sylphy.embedding_extractor import EmbeddingFactory


def _make_inst(df: pd.DataFrame):
    return EmbeddingFactory(
        model_name="facebook/esm2_t6_8M_UR50D",
        dataset=df,
        column_seq="sequence",
        name_device="cpu",
        debug=True,
        debug_mode=10,
        precision="fp32",
        oom_backoff=True,
    )


def test_load_and_pool_mean_cls_eos_and_layers():
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inst = _make_inst(df)
    inst.load_hf_tokenizer_and_model()

    seqs = df["sequence"].tolist()

    # Pooling on last layer
    for pool in ("mean", "cls", "eos"):
        out = inst.encode_batch_layers(seqs, max_length=16, layers="last", layer_agg="mean", pool=pool)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 4)  # B x H, with H=4 from the fake model

    # Multi-layer: last4 with concat -> 4 * H
    out_concat = inst.encode_batch_layers(seqs, max_length=16, layers="last4", layer_agg="concat", pool="mean")
    assert out_concat.shape == (3, 16)

    # Multi-layer: last4 with mean aggregation -> H
    out_mean = inst.encode_batch_layers(seqs, max_length=16, layers="last4", layer_agg="mean", pool="mean")
    assert out_mean.shape == (3, 4)


def test_run_process_end_to_end():
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inst = _make_inst(df)
    inst.load_hf_tokenizer_and_model()
    inst.run_process(max_length=16, batch_size=2, layers="last", layer_agg="mean", pool="mean")

    X = inst.coded_dataset
    assert X.shape[0] == 3
    # hidden size 4 + 'sequence' column
    assert X.shape[1] == 5
    assert "sequence" in X.columns
