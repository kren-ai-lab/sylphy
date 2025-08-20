from enum import Enum

class AtomLevel(str, Enum):
    CA = "ca"            
    BACKBONE = "backbone"
    ALL = "all"

class ContactValue(str, Enum):
    BINARY = "binary"
    WEIGHTED = "weighted"
    BINNED = "binned"


class WeightScheme(str, Enum):
    INV_DIST = "inv_dist"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"


class GraphMode(str, Enum):
    THRESHOLD = "threshold"
    KNN = "knn"

class CoordMetric(str, Enum):
    RMSD = "rmsd"
    HAUSDORFF = "hausdorff"


class CMapMetric(str, Enum):
    FROBENIUS = "frobenius"
    COSINE = "cosine"
    CORR = "corr"  


class GraphMetric(str, Enum):
    EDGE_JACCARD = "edge_jaccard"            
    WEIGHTED_FROBENIUS = "weighted_frobenius"  
    SHORTESTPATH_FROBENIUS = "shortestpath_frobenius"  


SUPPORTED_EXTS = {".pdb", ".ent", ".cif", ".mmcif"}

PATTERNS = ["*.pdb", "*.ent", "*.cif", "*.mmcif"]

BACKBONE_SET = {"N", "CA", "C", "O"}
