"""Public ``sylphy.validate_config``."""

from __future__ import annotations

import subprocess
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING

from sylphy import ConfigIssue, validate_config

if TYPE_CHECKING:
    from pathlib import Path


def test_valid_and_invalid_dict() -> None:
    assert validate_config({"max_length": 9, "method": "kmers"}, "encode") == []
    paths = {i.path for i in validate_config({"method": "nope", "bogus": 1}, "encode")}
    assert paths == {"method", "bogus"}


def test_non_dict_mapping() -> None:
    assert validate_config(MappingProxyType({"batch_size": 2}), "embed") == []
    assert validate_config(MappingProxyType({"batch_size": 0}), "embed")[0].path == "batch_size"


def test_path(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("batch_size: 2\n", encoding="utf-8")
    assert validate_config(cfg, "embed") == []
    assert validate_config(str(cfg), "embed") == []


def test_missing_file(tmp_path: Path) -> None:
    issues = validate_config(tmp_path / "nope.yaml", "encode")
    assert len(issues) == 1
    assert issues[0].path == ""
    assert isinstance(issues[0], ConfigIssue)


def test_stays_light() -> None:
    code = (
        "import sys, sylphy; sylphy.validate_config({'max_length': 5}, 'encode');"
        "sys.exit(any(m in sys.modules for m in ('torch', 'transformers')))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603
