from __future__ import annotations


def test_lazy_exports_and_public_symbols() -> None:
    """Verify public API exports are available and lazy-loaded correctly."""
    import sylphy.embedding_extractor as ee

    assert "EmbeddingBase" in dir(ee)
    assert "create_embedding" in dir(ee)
    assert callable(ee.create_embedding)
