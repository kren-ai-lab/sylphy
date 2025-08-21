# protein_representation/sequence_encoder/factory.py
from __future__ import annotations

from typing import Literal, Mapping, Type

from .ordinal_encoder import OrdinalEncoder
from .one_hot_encoder import OneHotEncoder
from .frequency_encoder import FrequencyEncoder
from .kmers_encoder import KMersEncoders
from .physicochemical_encoder import PhysicochemicalEncoder
from .fft_encoder import FFTEncoder

EncoderName = Literal[
    "ordinal", "one_hot", "onehot", "frequency", "kmers", "kmer", "tfidf",
    "physicochemical", "physchem", "aaindex", "fft",
]

_REGISTRY: Mapping[str, Type] = {
    # canonical
    "ordinal": OrdinalEncoder,
    "one_hot": OneHotEncoder,
    "frequency": FrequencyEncoder,
    "kmers": KMersEncoders,
    "physicochemical": PhysicochemicalEncoder,
    "fft": FFTEncoder,
    # aliases
    "onehot": OneHotEncoder,
    "kmer": KMersEncoders,
    "tfidf": KMersEncoders,
    "physchem": PhysicochemicalEncoder,
    "aaindex": PhysicochemicalEncoder,
}


def create_encoder(name: EncoderName, **kwargs):
    """
    Factory for sequence encoders.

    Parameters
    ----------
    name : {'ordinal','one_hot','onehot','frequency','kmers','kmer','tfidf',
            'physicochemical','physchem','aaindex','fft'}
        Encoder key or alias.
    **kwargs :
        Passed through to the encoder class constructor.

    Returns
    -------
    Encoders | FFTEncoder
        An initialized encoder instance.
    """
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {sorted(set(_REGISTRY))}")
    cls = _REGISTRY[key]
    return cls(**kwargs)
