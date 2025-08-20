"""
Miscellaneous Utilities for Clustering Sequence Analysis
========================================================

This package contains shared constants and utility functions used across
bioinformatics pipelines for clustering amino acid sequence characterization, 
preprocessing, and sampling strategies.

Modules
-------
- constants   : Defines the canonical amino acid alphabet and related descriptors.
- utils_lib   : Utility functions for random selection and stratified sampling.

Author: KREN AI LAB
License: GNU GENERAL PUBLIC LICENSE
"""

__version__ = "1.0.0"
__author__ = "KREN AI LAB"
__email__ = "krenai@umag.cl"
__license__ = "GNU GENERAL PUBLIC LICENSE"

from .constants import Constant
from .utils_lib import UtilsLib

__all__ = [
    "Constant",
    "UtilsLib"
]
