from enum import Enum

class DimReduction(str, Enum):
    pca = "pca"
    tsne = "tsne"
    umap = "umap"

class ExportFormat(str, Enum):
    csv = "csv"
    tsv = "tsv"
    fasta = "fasta"
