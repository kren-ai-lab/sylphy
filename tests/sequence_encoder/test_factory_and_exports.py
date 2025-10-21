from __future__ import annotations

import pandas as pd
import pytest

from sylphy.sequence_encoder import (
    FFTEncoder,
    FrequencyEncoder,
    KMersEncoders,
    OneHotEncoder,
    OrdinalEncoder,
    PhysicochemicalEncoder,
    create_encoder,
)


def test_factory_known_aliases(toy_df):
    cases: dict[str, type] = {
        "ordinal": OrdinalEncoder,
        "one_hot": OneHotEncoder,
        "onehot": OneHotEncoder,
        "frequency": FrequencyEncoder,
        "kmers": KMersEncoders,
        "kmer": KMersEncoders,
        "tfidf": KMersEncoders,
        "physicochemical": PhysicochemicalEncoder,
        "physchem": PhysicochemicalEncoder,
        "aaindex": PhysicochemicalEncoder,
        "fft": FFTEncoder,
    }
    for key, cls in cases.items():
        enc = create_encoder(key, dataset=toy_df, max_length=8)
        assert isinstance(enc, cls)


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        create_encoder("nope", dataset=pd.DataFrame({"sequence": ["AAA"]}))
