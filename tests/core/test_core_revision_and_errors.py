from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from sylphy.core.config import get_config
from sylphy.core.model_registry import (
    ModelDownloadError,
    ModelNotFoundError,
    register_alias,
    register_model,
    resolve_model,
)
from sylphy.core.model_spec import ModelSpec


def test_hf_revision_path_is_used(monkeypatch, tmp_path):
    """
    If a revision is provided, the resolved cache path should include the revision
    (per CachePaths.hf_model_dir(org, model, revision)).
    """
    calls = {"n": 0}
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(ref, revision=None, local_dir=None, local_dir_use_symlinks=None):
        calls["n"] += 1
        if local_dir is None:
            raise AssertionError("local_dir should be provided in snapshot_download mock.")
        dest = Path(local_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "marker.txt").write_text("ok")

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    spec = ModelSpec(
        name="hf_with_rev",
        provider="huggingface",
        ref="org/name",
        revision="r123",
    )
    register_model(spec)
    p = resolve_model("hf_with_rev")
    # Expected path must end with the revision directory (layout enforced by CachePaths)
    assert str(p).endswith("/org/name/r123") or str(p).endswith("\\org\\name\\r123")
    assert (p / "marker.txt").exists()
    assert calls["n"] == 1


def test_alias_requires_existing_canonical():
    with pytest.raises(ModelNotFoundError):
        register_alias("alias_x", "not_registered")


def test_env_override_nonexistent_raises(monkeypatch, tmp_path):
    import sylphy.core.model_registry as regmod

    monkeypatch.setattr(regmod, "_ENV_PREFIX", "PR_MODEL_", raising=True)
    register_model(ModelSpec(name="x", provider="huggingface", ref="org/x"))
    monkeypatch.setenv("PR_MODEL_X", str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError):
        resolve_model("x")


def test_unsupported_provider_wraps_in_download_error():
    register_model(ModelSpec(name="bad", provider=cast(Any, "weird"), ref="x/y"))
    with pytest.raises(ModelDownloadError):
        resolve_model("bad")


def test_cache_layout_helpers_agree_with_resolve(monkeypatch):
    """
    Smoke check: resolve_model() returns a path equal to CachePaths.hf_model_dir
    for HF models (no revision case).
    """
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(*args, **kwargs):
        # simulate download by creating the destination directory
        local_dir = kwargs.get("local_dir")
        # Also support positional style: snapshot_download(ref, revision=None, local_dir=..., ...)
        if local_dir is None and len(args) >= 1:
            # args[0] is ref; we still need local_dir from kwargs; keep no-op if missing
            pass
        if local_dir:
            Path(str(local_dir)).mkdir(parents=True, exist_ok=True)

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    spec = ModelSpec(name="hf_simple", provider="huggingface", ref="org/name2")
    register_model(spec)
    p_resolved = resolve_model("hf_simple")

    cfg = get_config()
    expected = cfg.cache_paths.hf_model_dir("org", "name2")
    assert str(p_resolved) == str(expected)
