"""
Sequence Encoding Toolkit
=========================

This package provides a unified interface for encoding protein or peptide sequences
into numerical formats using various strategies, including one-hot encoding, ordinal
mapping, physicochemical properties, k-mer frequencies, and FFT-based transformation.

Modules
-------
- base_encoder               : Abstract base class with validation and preprocessing.
- ordinal_encoder            : Encodes residues by their ordinal position.
- one_hot_encoder            : Encodes residues using binary vectors (one-hot).
- frequency_encoder          : Encodes residues by normalized frequency per position.
- kmers_encoder              : Applies TF-IDF vectorization on k-mer subsequences.
- physicochemical_encoder    : Uses physicochemical properties from AAIndex.
- fft_encoder                : Applies Fast Fourier Transform to numeric vectors.

Author: KREN AI LAB
License: GNU GENERAL PUBLIC LICENSE
"""

__version__ = "1.0.0"
__author__ = "KREN AI LAB"
__email__ = "krenai@umag.cl"
__license__ = "GNU GENERAL PUBLIC LICENSE"

from .base_encoder import Encoders
from .ordinal_encoder import OrdinalEncoder
from .one_hot_encoder import OneHotEncoder
from .frequency_encoder import FrequencyEncoder
from .kmers_encoder import KMersEncoders
from .physicochemical_encoder import PhysicochemicalEncoder
from .fft_encoder import FFTEncoder

__all__ = [
    "Encoders",
    "OrdinalEncoder",
    "OneHotEncoder",
    "FrequencyEncoder",
    "KMersEncoders",
    "PhysicochemicalEncoder",
    "FFTEncoder"
]
