from __future__ import annotations

import polars as pl
import pytest

from sylphy.embedding_extractor import create_embedding
from sylphy.embedding_extractor.ankh2_embedding import Ankh2Embedding
from sylphy.embedding_extractor.esm_embedding import ESMEmbedding
from sylphy.embedding_extractor.esmc_embedding import ESMCEmbedding
from sylphy.embedding_extractor.mistral_embedding import MistralEmbedding
from sylphy.embedding_extractor.prot_bert_embedding import ProtBertEmbedding
from sylphy.embedding_extractor.prot_t5_embedding import ProtT5Embedding


@pytest.mark.parametrize(
    ("model_name", "cls"),
    [
        ("facebook/esm2_t6_8M_UR50D", ESMEmbedding),
        ("ElnaggarLab/ankh2-ext1", Ankh2Embedding),
        ("Rostlab/prot_t5_xl_uniref50", ProtT5Embedding),
        ("Rostlab/prot_bert", ProtBertEmbedding),
        ("RaphaelMourad/Mistral-Prot-v1-134M", MistralEmbedding),
        ("esmc_300m", ESMCEmbedding),
    ],
)
def test_factory_selects_backend(model_name: str, cls: type) -> None:
    """Verify factory routes model names to the correct backend class."""
    df = pl.DataFrame({"sequence": ["AAAA"]})
    inst = create_embedding(model_name=model_name, dataset=df, column_seq="sequence", name_device="cpu")
    assert isinstance(inst, cls)


def test_factory_unknown_raises() -> None:
    """Verify factory raises ValueError for unknown model names."""
    df = pl.DataFrame({"sequence": ["AAAA"]})
    with pytest.raises(ValueError, match=r"Unknown model name"):
        create_embedding(model_name="unknown/model", dataset=df, column_seq="sequence", name_device="cpu")
