# core/model_registry.py
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from threading import RLock

from sylphy.constants.tool_constants import _ENV_PREFIX
from sylphy.logging import get_logger

from .config import get_config
from .model_spec import ModelSpec

logger = get_logger(__name__)
_LOCK = RLock()

# ----------------------------
# Exceptions
# ----------------------------


class ModelRegistryError(Exception):
    """Base error for the model registry."""


class ModelNotFoundError(ModelRegistryError):
    """Raised when a model name (or alias) is not found in the registry."""


class ModelDownloadError(ModelRegistryError):
    """Raised when a model cannot be downloaded or resolved."""


# ----------------------------
# Registry storage
# ----------------------------

_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    """
    Register (or overwrite) a model spec by name.

    Thread-safe and idempotent per `name`.
    """
    with _LOCK:
        _REGISTRY[spec.name] = spec
        logger.info("Registered model '%s' (provider=%s, ref=%s)", spec.name, spec.provider, spec.ref)


def register_alias(alias: str, canonical: str) -> None:
    """
    Register an alias that resolves to an existing canonical name.
    """
    with _LOCK:
        if canonical not in _REGISTRY:
            raise ModelNotFoundError(f"Cannot alias unknown model '{canonical}'")
        base = _REGISTRY[canonical]
        _REGISTRY[alias] = replace(base, name=alias, alias_of=canonical)
        logger.info("Registered alias '%s' -> '%s'", alias, canonical)


def unregister(name: str) -> None:
    """Remove a model or alias from the registry. Intended for tests."""
    with _LOCK:
        if _REGISTRY.pop(name, None) is not None:
            logger.info("Unregistered '%s'", name)


def clear_registry() -> None:
    """Clear all entries. Intended for tests."""
    with _LOCK:
        _REGISTRY.clear()
        logger.debug("Registry cleared")


def list_registered_models(include_aliases: bool = False) -> list[str]:
    """
    List registered model names.

    Parameters
    ----------
    include_aliases : bool, default=False
        If False, return only canonical names.
    """
    with _LOCK:
        names = [name for name, spec in _REGISTRY.items() if include_aliases or not spec.is_alias()]
    return sorted(names)


def get_model_spec(name: str) -> ModelSpec:
    """
    Get the spec for a model or alias.

    Raises
    ------
    ModelNotFoundError
        If `name` is missing.
    """
    with _LOCK:
        if name not in _REGISTRY:
            raise ModelNotFoundError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
        return _REGISTRY[name]


# ----------------------------
# Resolution / download
# ----------------------------


def _env_override_path(name: str) -> Path | None:
    """
    Environment variable override for local model path.

    The variable name is ``f"{_ENV_PREFIX}{name.upper().replace('-', '_')}"``.
    """
    key = _ENV_PREFIX + name.upper().replace("-", "_")
    val = os.getenv(key)
    if not val:
        return None

    p = Path(val).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Environment override {key} points to non-existent path: {p}")
    logger.debug("Using environment override %s -> %s", key, p)
    return p


def resolve_model(name: str) -> Path:
    """
    Resolve a model name to a local directory path, downloading if needed.

    Resolution priority:
    1) Environment override (see :func:`_env_override_path`).
    2) Provider-specific logic using the configured cache directory.
    """
    env_path = _env_override_path(name)
    if env_path is not None:
        return env_path

    cfg = get_config()
    spec = get_model_spec(name)

    provider = spec.provider.lower()
    try:
        if provider == "huggingface":
            org, model = _split_org_model(spec.ref)
            # Use revision-aware path layout to isolate snapshots
            local_dir = cfg.cache_paths.hf_model_dir(org, model, revision=spec.revision)
            if spec.subdir:
                local_dir = local_dir / spec.subdir
            local_dir.mkdir(parents=True, exist_ok=True)

            # If directory is non-empty, assume already present
            if any(local_dir.iterdir()):
                logger.info("Resolved HF model (cached): %s", local_dir)
                return local_dir

            _download_huggingface(ref=spec.ref, revision=spec.revision, dst=local_dir)
            return local_dir

        if provider == "other":
            local_dir = cfg.cache_paths.other_model_dir("custom", spec.name)
            if spec.subdir:
                local_dir = local_dir / spec.subdir
            local_dir.mkdir(parents=True, exist_ok=True)

            if any(local_dir.iterdir()):
                logger.info("Resolved OTHER model (cached): %s", local_dir)
                return local_dir

            _download_other(spec.ref, local_dir)
            return local_dir

        raise ValueError(f"Unsupported provider '{spec.provider}' for model '{name}'")

    except Exception as e:
        raise ModelDownloadError(f"Failed to resolve model '{name}': {e}") from e


def _split_org_model(ref: str) -> tuple[str, str]:
    if "/" not in ref:
        raise ValueError(f"Hugging Face ref must be 'org/model', got '{ref}'")
    org, model = ref.split("/", 1)
    return org, model


def _download_huggingface(ref: str, revision: str | None, dst: Path) -> None:
    """
    Download a model snapshot into `dst` using huggingface_hub with a local cache.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to download models from HF. "
            "Install with: pip install 'huggingface_hub>=0.23'"
        ) from e

    logger.info("Downloading HF model %s -> %s (rev=%s)", ref, dst, revision or "default")
    snapshot_download(
        ref,
        revision=revision,
        local_dir=str(dst),
        # token=os.getenv("HF_TOKEN")  # Uncomment if you want to force auth via env
    )


def _download_other(ref: str, dst: Path) -> None:
    """
    Download or copy a non-HF model into `dst`.

    - If `ref` is a local path, copy its contents into `dst`.
    - Else treat `ref` as a URL and download; if archive, extract.
    """
    import shutil
    import tarfile
    import zipfile
    from urllib.parse import urlparse

    import requests

    p = Path(ref).expanduser()
    if p.exists():
        logger.info("Copying local model from %s to %s", p, dst)
        if p.is_dir():
            for item in p.iterdir():
                dest = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
        else:
            shutil.copy2(p, dst / p.name)
        return

    parsed = urlparse(ref)
    if not parsed.scheme:
        raise ValueError(f"'other' provider ref is neither local path nor URL: {ref}")

    tmp = dst / "download.tmp"
    logger.info("Downloading model from %s", ref)
    with requests.get(ref, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    try:
        if tarfile.is_tarfile(tmp):
            with tarfile.open(tmp) as tf:
                tf.extractall(dst)
            tmp.unlink(missing_ok=True)
        elif zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as zf:
                zf.extractall(dst)
            tmp.unlink(missing_ok=True)
        else:
            logger.debug("Downloaded file is not an archive; leaving as-is: %s", tmp)
    except Exception:
        logger.exception("Failed to extract downloaded archive: %s", tmp)
        raise


def _init_default_registry() -> None:
    # --- ESM2 family ---
    register_model(
        ModelSpec(
            name="esm2_t6_8M_UR50D",
            provider="huggingface",
            ref="facebook/esm2_t6_8M_UR50D",
        )
    )
    register_model(
        ModelSpec(
            name="esm2_t12_35M_UR50D",
            provider="huggingface",
            ref="facebook/esm2_t12_35M_UR50D",
        )
    )
    register_model(
        ModelSpec(
            name="esm2_t30_150M_UR50D",
            provider="huggingface",
            ref="facebook/esm2_t30_150M_UR50D",
        )
    )
    register_model(
        ModelSpec(
            name="esm2_t33_650M_UR50D",
            provider="huggingface",
            ref="facebook/esm2_t33_650M_UR50D",
        )
    )
    register_model(
        ModelSpec(
            name="esm2_t36_3B_UR50D",
            provider="huggingface",
            ref="facebook/esm2_t36_3B_UR50D",
        )
    )
    register_model(
        ModelSpec(
            name="esm2_t48_15B_UR50D",
            provider="huggingface",
            ref="facebook/esm2_t48_15B_UR50D",
        )
    )

    # --- ProtT5 family ---
    register_model(
        ModelSpec(
            name="prot_t5_xl_uniref50",
            provider="huggingface",
            ref="Rostlab/prot_t5_xl_uniref50",
        )
    )
    register_model(
        ModelSpec(
            name="prot_t5_xl_bfd",
            provider="huggingface",
            ref="Rostlab/prot_t5_xl_bfd",
        )
    )

    # --- ProtBERT ---
    register_model(
        ModelSpec(
            name="prot_bert",
            provider="huggingface",
            ref="Rostlab/prot_bert",
        )
    )

    # --- ANKH2 ---
    register_model(
        ModelSpec(
            name="ankh2_ext1",
            provider="huggingface",
            ref="ElnaggarLab/ankh2-ext1",
        )
    )
    register_model(
        ModelSpec(
            name="ankh2_large",
            provider="huggingface",
            ref="ElnaggarLab/ankh2-large",
        )
    )

    # --- Mistral-Prot ---
    register_model(
        ModelSpec(
            name="mistral_prot_15m",
            provider="huggingface",
            ref="RaphaelMourad/Mistral-Prot-v1-15M",
        )
    )


# Initialize once on import
_init_default_registry()

__all__ = [
    "ModelRegistryError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "register_model",
    "register_alias",
    "unregister",
    "clear_registry",
    "list_registered_models",
    "get_model_spec",
    "resolve_model",
]
