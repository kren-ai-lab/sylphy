from typing import List, Dict


class Constant:
    """
    Container for constants and utility methods related to amino acid residues
    and protein sequence descriptors.

    This class includes URLs for accessing external databases (UniProt, AlphaFold, PDB),
    standard amino acid codes, and lists of descriptors used for protein feature extraction.
    """

    # === External Database URLs ===
    BASE_URL_UNIPROT: str = "https://rest.uniprot.org/uniprotkb/"
    BASE_URL_ALPHAFOLD: str = "https://alphafold.ebi.ac.uk/files"
    BASE_URL_PDB: str = "https://files.rcsb.org/download/"
    BASE_URL_AAINDEX: str = "https://raw.githubusercontent.com/ProteinEngineering-PESB2/data_and_info/refs/heads/main/aaindex_processed/aaindex_encoders.csv"
    BASE_URL_CLUSTERS_DESCRIPTORS: str = "https://raw.githubusercontent.com/ProteinEngineering-PESB2/data_and_info/refs/heads/main/cluster_based_encoder/cluster_encoders.csv"

    # === Residue Definitions ===
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

    @classmethod
    def get_index(cls, residue: str) -> int:
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
        if residue not in cls.POSITION_RESIDUES:
            raise KeyError(f"Residue '{residue}' is not a recognized amino acid.")
        return cls.POSITION_RESIDUES[residue]

    @classmethod
    def get_residue(cls, index: int) -> str:
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
        if not 0 <= index < len(cls.LIST_RESIDUES):
            raise IndexError(f"Index '{index}' is out of bounds. Must be between 0 and {len(cls.LIST_RESIDUES) - 1}.")
        return cls.LIST_RESIDUES[index]
