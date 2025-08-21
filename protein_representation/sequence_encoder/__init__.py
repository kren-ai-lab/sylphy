# protein_representation/sequence_encoder/__init__.py
"""
Sequence Encoding Toolkit
=========================

Encoders to transform protein/peptide sequences into numerical representations:
one-hot, ordinal, frequency, k-mer TF-IDF, physicochemical (AAIndex), and FFT.
"""

from .base_encoder import Encoders
from .ordinal_encoder import OrdinalEncoder
from .one_hot_encoder import OneHotEncoder
from .frequency_encoder import FrequencyEncoder
from .kmers_encoder import KMersEncoders
from .physicochemical_encoder import PhysicochemicalEncoder
from .fft_encoder import FFTEncoder
from .factory import create_encoder, EncoderName

__all__ = [
    "Encoders",
    "OrdinalEncoder",
    "OneHotEncoder",
    "FrequencyEncoder",
    "KMersEncoders",
    "PhysicochemicalEncoder",
    "FFTEncoder",
    "create_encoder",
    "EncoderName",
]
