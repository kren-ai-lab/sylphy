"""Provenance sidecar (``provenance/1``) shared by the ecosystem CLIs.

Copied into each library rather than imported, so the libraries stay independent.
Keep copies in sync with the contract's ``provenance.schema.json``.
"""
# ruff: noqa: ANN401

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

SCHEMA_VERSION = "provenance/1"
_CHUNK = 1 << 20


def utc_now() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_path(path: Path) -> str:
    r"""Hash a file, or a directory as the sorted ``relative/path\tfile_sha256`` lines of its files."""
    if path.is_dir():
        lines = (
            f"{p.relative_to(path).as_posix()}\t{_sha256_file(p)}\n"
            for p in sorted(path.rglob("*"))
            if p.is_file()
        )
        return hashlib.sha256("".join(lines).encode()).hexdigest()
    return _sha256_file(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize deterministically; paths and other non-JSON scalars become strings."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def fingerprint(parameters: Mapping[str, Any]) -> str:
    """Return ``sha256:<hex>`` of the canonical JSON of ``parameters``."""
    return "sha256:" + hashlib.sha256(canonical_json(parameters).encode()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_provenance(
    output: Path,
    *,
    tool: str,
    version: str,
    command: str,
    created_at: str,
    parameters: Mapping[str, Any],
    inputs: Iterable[Path] = (),
    packages: Iterable[str] = (),
    output_info: Mapping[str, int] | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``<output>.provenance.json`` next to ``output`` (a file or a directory) and return its path.

    Call it after ``output`` is complete; ``created_at`` is the ``utc_now()`` taken when the run started.
    """
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": tool, "version": version},
        "command": command,
        "created_at": created_at,
        "finished_at": utc_now(),
        "parameters": json.loads(canonical_json(parameters)),
        "config_fingerprint": fingerprint(parameters),
        "inputs": [{"path": str(path), "sha256": sha256_path(path)} for path in inputs],
        "outputs": [{"path": str(output), "sha256": sha256_path(output), **(output_info or {})}],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: _package_version(name) for name in packages},
        },
        "extensions": dict(extensions or {}),
    }
    target = output.with_name(output.name + ".provenance.json")
    target.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    return target
