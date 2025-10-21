from __future__ import annotations


def test_lazy_exports_and_public_symbols():
    """Verify public API exports are available and lazy-loaded correctly."""
    import sylphy.embedding_extractor as ee

    assert "EmbeddingBased" in dir(ee)
    assert "EmbeddingFactory" in dir(ee)
    assert callable(ee.EmbeddingFactory)
