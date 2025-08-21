# tests/embedding_extractor/test_factory_routing_and_logging.py
from __future__ import annotations

import pandas as pd
import pytest

from protein_representation.embedding_extractor import EmbeddingFactory
from protein_representation.embedding_extractor.esm_based import ESMBasedEmbedding
from protein_representation.embedding_extractor.ankh2_based import Ankh2BasedEmbedding
from protein_representation.embedding_extractor.prot5_based import Prot5Based
from protein_representation.embedding_extractor.bert_based import BertBasedEmbedding
from protein_representation.embedding_extractor.mistral_based import MistralBasedEmbedding
from protein_representation.embedding_extractor.esmc_based import ESMCBasedEmbedding


@pytest.mark.parametrize(
    "model_name, cls",
    [
        ("facebook/esm2_t6_8M_UR50D", ESMBasedEmbedding),
        ("ElnaggarLab/ankh2-ext1", Ankh2BasedEmbedding),
        ("Rostlab/prot_t5_xl_uniref50", Prot5Based),
        ("Rostlab/prot_bert", BertBasedEmbedding),
        ("RaphaelMourad/Mistral-Prot-v1-15M", MistralBasedEmbedding),
        ("esmc_300m", ESMCBasedEmbedding),
    ],
)
def test_factory_selects_backend_and_logs(model_name, cls, caplog):
    df = pd.DataFrame({"sequence": ["AAAA"]})
    caplog.set_level("INFO", logger="protein_representation.embedding_extractor.factory")
    inst = EmbeddingFactory(
        model_name=model_name,
        dataset=df,
        column_seq="sequence",
        name_device="cpu",
        debug=True,
        debug_mode=20,
    )
    assert isinstance(inst, cls)
    # Ensure a selection log line was emitted
    assert any("Selecting" in rec.getMessage() for rec in caplog.records)


def test_factory_unknown_raises():
    df = pd.DataFrame({"sequence": ["AAAA"]})
    with pytest.raises(ValueError):
        EmbeddingFactory(
            model_name="unknown/model",
            dataset=df,
            column_seq="sequence",
            name_device="cpu",
        )
