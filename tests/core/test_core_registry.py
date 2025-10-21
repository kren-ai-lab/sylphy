# tests/test_core_registry.py
from __future__ import annotations

import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from sylphy.core.config import get_config, temporary_cache_root
from sylphy.core.model_registry import (
    ModelDownloadError,
    ModelNotFoundError,
    clear_registry,
    get_model_spec,
    list_registered_models,
    register_alias,
    register_model,
    resolve_model,
    unregister,
)
from sylphy.core.model_spec import ModelSpec


def test_register_and_get_spec_roundtrip():
    spec = ModelSpec(name="dummy", provider="huggingface", ref="org/model")
    register_model(spec)
    got = get_model_spec("dummy")
    assert got.name == "dummy"
    assert got.provider == "huggingface"
    assert got.ref == "org/model"

    with pytest.raises(ModelNotFoundError):
        get_model_spec("nope")


def test_alias_registration_and_listing():
    register_model(ModelSpec(name="esm2_small", provider="huggingface", ref="facebook/esm2_t6_8M_UR50D"))
    register_alias("esm2", "esm2_small")

    spec_alias = get_model_spec("esm2")
    assert spec_alias.alias_of == "esm2_small"
    assert spec_alias.name == "esm2"

    names_no_alias = list_registered_models(include_aliases=False)
    names_with_alias = list_registered_models(include_aliases=True)
    assert "esm2_small" in names_no_alias and "esm2" not in names_no_alias
    assert set(names_with_alias) >= {"esm2_small", "esm2"}


def test_unregister_and_clear():
    register_model(ModelSpec(name="to_drop", provider="huggingface", ref="org/model"))
    unregister("to_drop")
    with pytest.raises(ModelNotFoundError):
        get_model_spec("to_drop")

    register_model(ModelSpec(name="a", provider="huggingface", ref="x/y"))
    register_model(ModelSpec(name="b", provider="huggingface", ref="x/z"))
    clear_registry()
    assert list_registered_models(True) == []


def test_env_override_path(monkeypatch, tmp_path):
    # Forzar el prefijo usado dentro del módulo
    import sylphy.core.model_registry as regmod

    monkeypatch.setattr(regmod, "_ENV_PREFIX", "PR_MODEL_", raising=True)

    # Carpeta fake de override
    override_dir = tmp_path / "pre_downloaded" / "esm2_small"
    override_dir.mkdir(parents=True)
    (override_dir / "weights.bin").write_text("ok")

    register_model(ModelSpec(name="esm2_small", provider="huggingface", ref="facebook/esm2_t6_8M_UR50D"))
    # Set env var PR_MODEL_ESM2_SMALL -> override_dir
    monkeypatch.setenv("PR_MODEL_ESM2_SMALL", str(override_dir))

    resolved = resolve_model("esm2_small")
    assert resolved == override_dir
    assert (resolved / "weights.bin").exists()


def test_resolve_huggingface_download_mocked(monkeypatch, tmp_path):
    # Mock huggingface_hub como módulo real con snapshot_download
    calls = {"n": 0}

    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(ref, revision=None, local_dir=None, local_dir_use_symlinks=None):
        calls["n"] += 1
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text(json.dumps({"ref": ref}))

    hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    register_model(ModelSpec(name="hf_model", provider="huggingface", ref="org/hf_model"))
    p1 = resolve_model("hf_model")
    assert (p1 / "config.json").exists()
    # Second resolve should reuse (no extra download)
    p2 = resolve_model("hf_model")
    assert p1 == p2
    assert calls["n"] == 1  # downloaded once


def test_resolve_other_local_copy(tmp_path):
    src = tmp_path / "src_model"
    src.mkdir()
    (src / "file.txt").write_text("payload")

    register_model(ModelSpec(name="other_local", provider="other", ref=str(src)))
    local_dir = resolve_model("other_local")
    assert (local_dir / "file.txt").exists()


def _make_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as z:
        z.writestr("inner/ok.txt", "ok")
    return buf.getvalue()


def test_resolve_other_url_download_and_extract(monkeypatch, tmp_path):
    # Mock requests.get → stream un zip pequeño
    class DummyResp:
        def __init__(self, data: bytes):
            self._data = data
            self.status_code = 200
            self.ok = True

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size=65536):
            yield self._data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_get(url, stream=True, timeout=60):
        return DummyResp(_make_zip_bytes())

    req = types.ModuleType("requests")
    req.get = fake_get  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", req)

    register_model(ModelSpec(name="other_url", provider="other", ref="https://example.com/model.zip"))
    dst = resolve_model("other_url")
    assert (dst / "inner" / "ok.txt").exists()


def test_download_error_is_wrapped(monkeypatch):
    register_model(ModelSpec(name="will_fail", provider="huggingface", ref="org/fail"))
    import sylphy.core.model_registry as regmod

    def boom(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(regmod, "_download_huggingface", boom, raising=True)

    with pytest.raises(ModelDownloadError) as ei:
        resolve_model("will_fail")
    assert "Failed to resolve model 'will_fail'" in str(ei.value)


def test_config_temporary_cache_root(tmp_path):
    cfg = get_config()
    original = cfg.cache_paths.cache_root
    with temporary_cache_root(tmp_path / "alt"):
        assert get_config().cache_paths.cache_root != original
    assert get_config().cache_paths.cache_root == original
