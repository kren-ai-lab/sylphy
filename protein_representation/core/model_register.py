# core/model_registry.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple
import os

from .config import get_config

# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class ModelSpec:
    """
    Canonical specification for a model.

    Attributes
    ----------
    name: str
        Canonical short name in the registry (e.g., "prot_t5_xl_uniref50").
    provider: str
        "huggingface" or "other".
    ref: str
        For HF: "org/model". For other: URL or local path.
    subdir: Optional[str]
        If you want to put files under an extra subdir (rare).
    revision: Optional[str]
        For HF: branch/tag/commit to pin (e.g., "main" or a SHA).
    alias_of: Optional[str]
        If this entry is an alias, point to the canonical name.
    """

    name: str
    provider: str
    ref: str
    subdir: Optional[str] = None
    revision: Optional[str] = None
    alias_of: Optional[str] = None


# ----------------------------
# Registry storage
# ----------------------------

_REGISTRY: Dict[str, ModelSpec] = {}

# Shortcuts so users can override locations via env vars:
# e.g., BIOCLUST_MODEL_PROT_T5_XL_UNIREF50=/models/prot_t5
_ENV_PREFIX = "BIOCLUST_MODEL_"

def register_model(spec: ModelSpec) -> None:
    """Register or overwrite a model spec by name."""
    _REGISTRY[spec.name] = spec

def register_alias(alias: str, canonical: str) -> None:
    """Register an alias that resolves to an existing canonical name."""
    if canonical not in _REGISTRY:
        raise KeyError(f"Cannot alias unknown model '{canonical}'")
    _REGISTRY[alias] = ModelSpec(
        name=alias,
        provider=_REGISTRY[canonical].provider,
        ref=_REGISTRY[canonical].ref,
        subdir=_REGISTRY[canonical].subdir,
        revision=_REGISTRY[canonical].revision,
        alias_of=canonical,
    )

def get_model_spec(name: str) -> ModelSpec:
    """Get the spec for a model or alias; raises if missing."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]

# ----------------------------
# Resolution / download
# ----------------------------

def _env_override_path(name: str) -> Optional[Path]:
    # Allow BIOCLUST_MODEL_<UPPERCASE_NAME>=/path/to/model
    key = _ENV_PREFIX + name.upper().replace("-", "_")
    val = os.getenv(key)
    if val:
        p = Path(val).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Environment override {key} points to non-existent path: {p}"
            )
        return p
    return None

def resolve_model(name: str) -> Path:
    """
    Resolve a model name to a local directory path.
    If not present, download or copy it into the cache using provider-specific logic.

    Returns
    -------
    Path
        Local directory containing the model files (weights/config).
    """
    # 1) env override wins
    env_path = _env_override_path(name)
    if env_path is not None:
        return env_path

    cfg = get_config()
    spec = get_model_spec(name)

    if spec.provider.lower() == "huggingface":
        org, model = _split_org_model(spec.ref)
        local_dir = cfg.cache_paths.hf_model_dir(org, model)
        if spec.subdir:
            local_dir = local_dir / spec.subdir
        local_dir.mkdir(parents=True, exist_ok=True)

        # If already populated (has files), return
        if any(local_dir.iterdir()):
            return local_dir

        _download_huggingface(ref=spec.ref, revision=spec.revision, dst=local_dir)
        return local_dir

    elif spec.provider.lower() == "other":
        # 'other' can be a local path or an URL
        local_dir = cfg.cache_paths.other_model_dir("custom", spec.name)
        local_dir.mkdir(parents=True, exist_ok=True)
        if any(local_dir.iterdir()):
            return local_dir
        _download_other(spec.ref, local_dir)
        return local_dir

    else:
        raise ValueError(f"Unsupported provider '{spec.provider}' for model '{name}'")


def _split_org_model(ref: str) -> Tuple[str, str]:
    if "/" not in ref:
        raise ValueError(f"HF ref must be 'org/model', got '{ref}'")
    org, model = ref.split("/", 1)
    return org, model

def _download_huggingface(ref: str, revision: Optional[str], dst: Path) -> None:
    """
    Download a model snapshot into `dst` using huggingface_hub.
    """
    try:
        # Lazy import to avoid mandatory dependency if user never uses HF models.
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required to download models from HF. "
            "Install with: pip install 'huggingface_hub>=0.23'"
        ) from e

    # Respect your cache policy: put all files under dst (no global HF cache pollution).
    # We set use_symlinks=False to ensure files are physically present in dst.
    snapshot_download(
        ref,
        revision=revision,
        local_dir=str(dst),
        local_dir_use_symlinks=False,
        # You can pass token=os.getenv("HF_TOKEN") if you need gated models
    )

def _download_other(ref: str, dst: Path) -> None:
    """
    'ref' can be:
      - a local path to copy
      - an URL to download a tar/zip (you implement the logic you need)
    Here we leave a simple stub.
    """
    from urllib.parse import urlparse
    import shutil
    import requests
    import tarfile
    import zipfile
    p = Path(ref).expanduser()

    # Local directory: copy contents
    if p.exists():
        if p.is_dir():
            # copytree replacement (py<3.8 compat): copy recursive
            for item in p.iterdir():
                dest = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            return
        else:
            # single file - just copy
            shutil.copy2(p, dst / p.name)
            return

    # Otherwise assume URL
    url = ref
    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError(f"'other' provider ref is neither local path nor URL: {ref}")

    # naive download
    tmp = dst / "download.tmp"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    # If archive, extract; else leave as is
    if tarfile.is_tarfile(tmp):
        with tarfile.open(tmp) as tf:
            tf.extractall(dst)
        tmp.unlink(missing_ok=True)
    elif zipfile.is_zipfile(tmp):
        with zipfile.ZipFile(tmp) as zf:
            zf.extractall(dst)
        tmp.unlink(missing_ok=True)
    else:
        # leave the downloaded file in place
        pass


# ----------------------------
# Built-in registry entries
# ----------------------------

def _init_default_registry() -> None:

    # --- ESM2 family ---
    register_model(ModelSpec(
        name="esm2_t6_8M_UR50D",
        provider="huggingface",
        ref="facebook/esm2_t6_8M_UR50D",
    ))
    register_model(ModelSpec(
        name="esm2_t33_650M_UR50D",
        provider="huggingface",
        ref="facebook/esm2_t33_650M_UR50D",
    ))

    register_model(ModelSpec(
        name="esm2_t12_35M_UR50D",
        provider="huggingface",
        ref="facebook/esm2_t12_35M_UR50D",
    ))

    register_model(ModelSpec(
        name="esm2_t48_15B_UR50D",
        provider="huggingface",
        ref="facebook/esm2_t48_15B_UR50D",
    ))

    register_model(ModelSpec(
        name="esm2_t36_3B_UR50D",
        provider="huggingface",
        ref="facebook/esm2_t36_3B_UR50D",
    ))

    register_model(ModelSpec(
        name="esm2_t30_150M_UR50D",
        provider="huggingface",
        ref="facebook/esm2_t30_150M_UR50D",
    ))
    
    
    # --- ProtT5 family ---
    register_model(ModelSpec(
        name="prot_t5_xl_uniref50",
        provider="huggingface",
        ref="Rostlab/prot_t5_xl_uniref50",
    ))
    register_model(ModelSpec(
        name="prot_t5_xl_bfd",
        provider="huggingface",
        ref="Rostlab/prot_t5_xl_bfd",
    ))

    # --- ProtBERT ---
    register_model(ModelSpec(
        name="prot_bert",
        provider="huggingface",
        ref="Rostlab/prot_bert",
    ))

    # --- ANKH2 ---
    register_model(ModelSpec(
        name="ankh2_ext1",
        provider="huggingface",
        ref="ElnaggarLab/ankh2-ext1",
    ))
    register_model(ModelSpec(
        name="ankh2_large",
        provider="huggingface",
        ref="ElnaggarLab/ankh2-large",
    ))

    # --- Mistral-Prot ---
    register_model(ModelSpec(
        name="mistral_prot_15m",
        provider="huggingface",
        ref="RaphaelMourad/Mistral-Prot-v1-15M",
    ))


_init_default_registry()
