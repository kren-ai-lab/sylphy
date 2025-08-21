# tests/embedding_extraction/test_base_loading_and_pooling.py
from __future__ import annotations

import numpy as np
import pandas as pd

from protein_representation.embedding_extractor import EmbeddingFactory


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


def test_load_and_pool_mean_cls_eos():
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inst = _make_inst(df)
    inst.load_hf_tokenizer_and_model()

    # Batch of 3 sequences
    seqs = df["sequence"].tolist()
    for pool in ("mean", "cls", "eos"):
        out = inst.encode_batch_last_hidden(seqs, max_length=16, batch_size=3, pool=pool)
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == 3
        # Hidden size = 4 in our fake model
        assert out.shape[1] == 4


def test_run_process_end_to_end():
    df = pd.DataFrame({"sequence": ["AAAA", "BBB", "CCCCC"]})
    inst = _make_inst(df)
    inst.load_hf_tokenizer_and_model()
    inst.run_process(max_length=16, batch_size=2, pool="mean")

    X = inst.coded_dataset
    assert X.shape[0] == 3  # three sequences
    # hidden size 4 + sequence column
    assert X.shape[1] == 5
    assert "sequence" in X.columns
