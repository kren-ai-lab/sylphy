from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from sylphy.core.model_registry import (
    ModelDownloadError,
    ModelNotFoundError,
    register_alias,
    register_model,
    resolve_model,
)
from sylphy.core.model_spec import ModelSpec


def test_hf_revision_is_passed_to_snapshot_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """snapshot_download receives the revision kwarg."""
    received: dict[str, object] = {}
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(ref: str, revision: str | None = None, **_kwargs: object) -> str:
        received["ref"] = ref
        received["revision"] = revision
        dest = tmp_path / "snap"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "marker.txt").write_text("ok")
        return str(dest)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download, raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    spec = ModelSpec(name="hf_with_rev", provider="huggingface", ref="org/name", revision="r123")
    register_model(spec)
    p = resolve_model("hf_with_rev")
    assert received["revision"] == "r123"
    assert (p / "marker.txt").exists()


def test_alias_requires_existing_canonical() -> None:
    with pytest.raises(ModelNotFoundError):
        register_alias("alias_x", "not_registered")


def test_env_override_nonexistent_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sylphy.core.model_registry as regmod

    monkeypatch.setattr(regmod, "_ENV_PREFIX", "PR_MODEL_", raising=True)
    register_model(ModelSpec(name="x", provider="huggingface", ref="org/x"))
    monkeypatch.setenv("PR_MODEL_X", str(tmp_path / "does_not_exist"))
    with pytest.raises(FileNotFoundError):
        resolve_model("x")


def test_unsupported_provider_wraps_in_download_error() -> None:
    register_model(ModelSpec(name="bad", provider=cast("Any", "weird"), ref="x/y"))
    with pytest.raises(ModelDownloadError):
        resolve_model("bad")


def test_resolve_hf_returns_path_from_snapshot_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """resolve_model() returns whatever path snapshot_download returns."""
    hub = types.ModuleType("huggingface_hub")
    expected = tmp_path / "snap"
    expected.mkdir()

    def snapshot_download(_ref: str, _revision: str | None = None, **_kwargs: object) -> str:
        return str(expected)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download, raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    spec = ModelSpec(name="hf_simple", provider="huggingface", ref="org/name2")
    register_model(spec)
    p_resolved = resolve_model("hf_simple")
    assert p_resolved == expected
