from typing import List, Dict

_ENV_PREFIX : str = "PROTEIN_REPRESENTATION_"

# constant for download data for encoding
BASE_URL_AAINDEX: str = "https://raw.githubusercontent.com/ProteinEngineering-PESB2/data_and_info/refs/heads/main/aaindex_processed/aaindex_encoders.csv"
BASE_URL_CLUSTERS_DESCRIPTORS: str = "https://raw.githubusercontent.com/ProteinEngineering-PESB2/data_and_info/refs/heads/main/cluster_based_encoder/cluster_encoders.csv"

# residue constants
LIST_RESIDUES: List[str] = [
    "A",  # Alanine
    "C",  # Cysteine
    "D",  # Aspartic acid
    "E",  # Glutamic acid
    "F",  # Phenylalanine
    "G",  # Glycine
    "H",  # Histidine
    "I",  # Isoleucine
    "N",  # Asparagine
    "K",  # Lysine
    "L",  # Leucine
    "M",  # Methionine
    "P",  # Proline
    "Q",  # Glutamine
    "R",  # Arginine
    "S",  # Serine
    "T",  # Threonine
    "V",  # Valine
    "W",  # Tryptophan
    "Y",  # Tyrosine
]

POSITION_RESIDUES: Dict[str, int] = {
    residue: index for index, residue in enumerate(LIST_RESIDUES)
}

# === Sequence Descriptors ===
LIST_DESCRIPTORS_SEQUENCE: List[str] = [
    "length", "molecular_weight", "aliphatic_index", "aromaticity", "boman_index",
    "hydrophobic_ratio", "charge", "charge_density", "instability_index",
    "isoelectric_point",
    "freq_A", "freq_C", "freq_D", "freq_E", "freq_F", "freq_G", "freq_H",
    "freq_I", "freq_N", "freq_K", "freq_L", "freq_M", "freq_P", "freq_Q",
    "freq_R", "freq_S", "freq_T", "freq_V", "freq_W", "freq_Y"
]

LIST_DESCRIPTORS_SEQUENCE_NON_NUMERIC: List[str] = [
    "sequence", "is_canon"
]

def get_index(residue: str) -> int:
    """
    Get the index of a single-letter amino acid residue.

    Parameters
    ----------
    residue : str
        A single-letter code representing an amino acid (e.g., 'A', 'R', 'L').

    Returns
    -------
    int
        The index position of the residue in the standard amino acid list.

    Raises
    ------
    KeyError
        If the residue is not part of the defined amino acid list.
    """
    if residue not in POSITION_RESIDUES:
        raise KeyError(f"Residue '{residue}' is not a recognized amino acid.")
    return POSITION_RESIDUES[residue]

def get_residue(index: int) -> str:
    """
    Get the amino acid residue corresponding to a given index.

    Parameters
    ----------
    index : int
        The position index in the amino acid list.

    Returns
    -------
    str
        The one-letter amino acid code at the given index.

    Raises
    ------
    IndexError
        If the index is out of bounds of the amino acid list.
    """
    if not 0 <= index < len(LIST_RESIDUES):
        raise IndexError(f"Index '{index}' is out of bounds. Must be between 0 and {len(LIST_RESIDUES) - 1}.")
    return LIST_RESIDUES[index]
