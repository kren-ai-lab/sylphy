from __future__ import annotations

"""
Surface sanity for the public API of `sylphy.embedding_extractor`.
"""


def test_lazy_exports_and_public_symbols():
    import sylphy.embedding_extractor as ee

    # Public symbols exposed
    assert "EmbeddingBased" in dir(ee)
    assert "EmbeddingFactory" in dir(ee)

    # Lazy access should not raise
    _ = ee.EmbeddingBased
    factory = ee.EmbeddingFactory
    assert callable(factory)
