from __future__ import annotations

import pandas as pd
import pytest

from sylphy.embedding_extractor import EmbeddingFactory
from sylphy.embedding_extractor.ankh2_based import Ankh2BasedEmbedding
from sylphy.embedding_extractor.bert_based import BertBasedEmbedding
from sylphy.embedding_extractor.esm_based import ESMBasedEmbedding
from sylphy.embedding_extractor.esmc_based import ESMCBasedEmbedding
from sylphy.embedding_extractor.mistral_based import MistralBasedEmbedding
from sylphy.embedding_extractor.prot5_based import Prot5Based


@pytest.mark.parametrize(
    "model_name, cls",
    [
        ("facebook/esm2_t6_8M_UR50D", ESMBasedEmbedding),
        ("ElnaggarLab/ankh2-ext1", Ankh2BasedEmbedding),
        ("Rostlab/prot_t5_xl_uniref50", Prot5Based),
        ("Rostlab/prot_bert", BertBasedEmbedding),
        ("RaphaelMourad/Mistral-Prot-v1-134M", MistralBasedEmbedding),
        ("esmc_300m", ESMCBasedEmbedding),
    ],
)
def test_factory_selects_backend(model_name, cls):
    """Verify factory routes model names to the correct backend class."""
    df = pd.DataFrame({"sequence": ["AAAA"]})
    inst = EmbeddingFactory(model_name=model_name, dataset=df, column_seq="sequence", name_device="cpu")
    assert isinstance(inst, cls)


def test_factory_unknown_raises():
    """Verify factory raises ValueError for unknown model names."""
    df = pd.DataFrame({"sequence": ["AAAA"]})
    with pytest.raises(ValueError):
        EmbeddingFactory(model_name="unknown/model", dataset=df, column_seq="sequence", name_device="cpu")
