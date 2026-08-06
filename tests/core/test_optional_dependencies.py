from __future__ import annotations

from sylphy.core.optional_dependencies import (
    build_optional_dependency_error,
    is_missing_optional_dependency,
    wrap_optional_dependency_error,
)


def test_build_optional_dependency_error_mentions_extra() -> None:
    err = build_optional_dependency_error(
        feature="ProtT5 embeddings",
        extra="embeddings",
        packages=("sentencepiece",),
    )

    assert "ProtT5 embeddings" in str(err)
    assert "sentencepiece" in str(err)
    assert "sylphy[embeddings]" in str(err)


def test_build_optional_dependency_error_uses_plural_install_target() -> None:
    err = build_optional_dependency_error(
        feature="ESM-C embeddings",
        extra="embeddings",
        packages=("esm", "torch"),
    )

    assert "Install them with" in str(err)


def test_is_missing_optional_dependency_matches_module_name() -> None:
    exc = ModuleNotFoundError("No module named 'sentencepiece'")
    exc.name = "sentencepiece"

    assert is_missing_optional_dependency(exc, ("sentencepiece",)) is True
    assert is_missing_optional_dependency(exc, ("transformers",)) is False


def test_wrap_optional_dependency_error_returns_none_for_unrelated_error() -> None:
    exc = RuntimeError("network timeout")

    assert (
        wrap_optional_dependency_error(
            exc,
            feature="ProtT5 embeddings",
            extra="embeddings",
            packages=("sentencepiece",),
        )
        is None
    )
