# tests/test_public_api.py
from __future__ import annotations

def test_public_surface_imports():
    import sylphy as pr

    # version string
    assert isinstance(pr.__version__, str)

    # core API symbols exist
    assert hasattr(pr, "ModelSpec")
    assert hasattr(pr, "register_model")
    assert hasattr(pr, "resolve_model")
    assert hasattr(pr, "get_config")
