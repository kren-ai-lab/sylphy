from __future__ import annotations

import numpy as np
import pandas as pd

from sylphy.embedding_extractor import EmbeddingFactory


def test_load_and_pool_mean_cls_eos_and_layers():
    """Verify different pooling strategies and layer aggregation methods."""
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inst = EmbeddingFactory(
        model_name="facebook/esm2_t6_8M_UR50D",
        dataset=df,
        column_seq="sequence",
        name_device="cpu",
        precision="fp32",
    )
    inst.load_hf_tokenizer_and_model()
    seqs = df["sequence"].tolist()

    for pool in ("mean", "cls", "eos"):
        out = inst.encode_batch_layers(seqs, max_length=16, layers="last", layer_agg="mean", pool=pool)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3, 4)

    out_concat = inst.encode_batch_layers(
        seqs, max_length=16, layers="last4", layer_agg="concat", pool="mean"
    )
    assert out_concat.shape == (3, 16)

    out_mean = inst.encode_batch_layers(seqs, max_length=16, layers="last4", layer_agg="mean", pool="mean")
    assert out_mean.shape == (3, 4)


def test_run_process_end_to_end():
    """Verify complete embedding extraction pipeline produces expected output."""
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inst = EmbeddingFactory(
        model_name="facebook/esm2_t6_8M_UR50D",
        dataset=df,
        column_seq="sequence",
        name_device="cpu",
        precision="fp32",
    )
    inst.load_hf_tokenizer_and_model()
    inst.run_process(max_length=16, batch_size=2, layers="last", layer_agg="mean", pool="mean")

    assert inst.coded_dataset.shape == (3, 5)
    assert "sequence" in inst.coded_dataset.columns
