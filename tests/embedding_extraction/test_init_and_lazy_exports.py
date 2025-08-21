# tests/embedding_extraction/test_init_and_lazy_exports.py
from __future__ import annotations

def test_lazy_exports_and_alias_available():
    import protein_representation.embedding_extractor as ee

    # Symbols exposed
    assert "EmbeddingBased" in dir(ee)
    assert "EmbeddingFactory" in dir(ee)
    assert "create_embedding" in dir(ee)
    assert isinstance(ee.SUPPORTED_FAMILIES, tuple)

    # Access triggers lazy import but should not error
    base = ee.EmbeddingBased
    factory = ee.EmbeddingFactory
    alias = ee.create_embedding

    assert factory is alias  # alias is in-place reference to the factory
