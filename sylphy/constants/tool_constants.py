# tool_constants.py
from __future__ import annotations

# Environment variable prefix used across the project (e.g., SYLPHY_CACHE_ROOT)
_ENV_PREFIX: str = "SYLPHY_"

# Remote sources for descriptor tables used by encoders
BASE_URL_AAINDEX: str = "https://raw.githubusercontent.com/ProteinEngineering-PESB2/data_and_info/refs/heads/main/aaindex_processed/aaindex_encoders.csv"
BASE_URL_CLUSTERS_DESCRIPTORS: str = "https://raw.githubusercontent.com/ProteinEngineering-PESB2/data_and_info/refs/heads/main/cluster_based_encoder/cluster_encoders.csv"

# Canonical amino acids (20). Keep this as the default set to avoid ambiguity.
LIST_RESIDUES: tuple[str, ...] = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "N",
    "K",
    "L",
    "M",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)

# Extended alphabet (optionally enabled via function parameters)
# Includes: B (D/N), Z (E/Q), X (unknown), U (selenocysteine), O (pyrrolysine).
LIST_RESIDUES_EXTENDED: tuple[str, ...] = LIST_RESIDUES + ("B", "Z", "X", "U", "O")

# Standard sequence-level descriptors
LIST_DESCRIPTORS_SEQUENCE: tuple[str, ...] = (
    "length",
    "molecular_weight",
    "aliphatic_index",
    "aromaticity",
    "boman_index",
    "hydrophobic_ratio",
    "charge",
    "charge_density",
    "instability_index",
    "isoelectric_point",
    "freq_A",
    "freq_C",
    "freq_D",
    "freq_E",
    "freq_F",
    "freq_G",
    "freq_H",
    "freq_I",
    "freq_N",
    "freq_K",
    "freq_L",
    "freq_M",
    "freq_P",
    "freq_Q",
    "freq_R",
    "freq_S",
    "freq_T",
    "freq_V",
    "freq_W",
    "freq_Y",
)

# Non-numeric fields frequently carried in encoded tables
LIST_DESCRIPTORS_SEQUENCE_NON_NUMERIC: tuple[str, ...] = ("sequence", "is_canon")


def residues(extended: bool = False) -> tuple[str, ...]:
    """
    Return the residue alphabet.

    Parameters
    ----------
    extended : bool, optional
        If True, include B, Z, X, U, O. Default False (canonical 20).

    Returns
    -------
    tuple[str, ...]
        Alphabet tuple.
    """
    return LIST_RESIDUES_EXTENDED if extended else LIST_RESIDUES


def position_residues(extended: bool = False) -> dict[str, int]:
    """
    Return a residue → index mapping for the selected alphabet.

    Parameters
    ----------
    extended : bool, optional
        If True, mapping for the extended alphabet.

    Returns
    -------
    dict[str, int]
        Mapping of one-letter residue to 0-based index.
    """
    alpha = residues(extended=extended)
    return {res: i for i, res in enumerate(alpha)}


def get_index(residue: str, *, extended: bool = False, allow_unknown: bool = False) -> int:
    """
    Return the 0-based index for a one-letter amino acid.

    Parameters
    ----------
    residue : str
        One-letter amino acid code.
    extended : bool, optional
        If True, the extended alphabet is used (B, Z, X, U, O).
    allow_unknown : bool, optional
        If True and `extended=False`, allow 'X' by promoting to extended set.

    Returns
    -------
    int
        Index in the selected alphabet.

    Raises
    ------
    KeyError
        If the residue is not part of the selected alphabet.
    """
    r = residue.strip().upper()
    use_extended = extended or (allow_unknown and r == "X")
    mapping = position_residues(extended=use_extended)
    if r not in mapping:
        raise KeyError(
            f"Residue '{residue}' is not recognized in "
            f"{'extended' if use_extended else 'canonical'} alphabet."
        )
    return mapping[r]


def get_residue(index: int, *, extended: bool = False) -> str:
    """
    Return the one-letter amino acid at the given 0-based index.

    Parameters
    ----------
    index : int
        Position in the selected alphabet.
    extended : bool, optional
        If True, the extended alphabet is used.

    Returns
    -------
    str
        One-letter amino acid code.

    Raises
    ------
    IndexError
        If the index is out of bounds.
    """
    alpha = residues(extended=extended)
    if not 0 <= index < len(alpha):
        raise IndexError(
            f"Index '{index}' is out of bounds. "
            f"Valid range: [0, {len(alpha) - 1}] for "
            f"{'extended' if extended else 'canonical'} alphabet."
        )
    return alpha[index]
