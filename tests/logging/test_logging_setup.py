# tests/test_logging_setup.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from sylphy.logging import (
    setup_logger,
    get_logger,
    add_context,
    reset_logging,
    silence_external,
    set_global_level,
)

PKG_LOGGER_NAME = "protein_representation"  # default in our implementation


def _count_handlers(logger: logging.Logger) -> int:
    return len(logger.handlers)

def test_setup_logger_idempotent(monkeypatch, tmp_path):
    # Force a specific log file to avoid hitting user home
    log_path = tmp_path / "idempotent.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))

    reset_logging(PKG_LOGGER_NAME)
    logger = setup_logger(name=PKG_LOGGER_NAME, level="INFO")
    n1 = _count_handlers(logger)

    # Calling again should NOT add more handlers
    logger2 = setup_logger(name=PKG_LOGGER_NAME, level="DEBUG")
    n2 = _count_handlers(logger2)

    assert logger is logger2
    assert n1 == n2 >= 2  # console + file
    assert log_path.parent.exists()


def test_env_level_override(monkeypatch, tmp_path, caplog):
    log_path = tmp_path / "level.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))
    monkeypatch.setenv("PR_LOG_LEVEL", "ERROR")

    reset_logging(PKG_LOGGER_NAME)
    logger = setup_logger(name=PKG_LOGGER_NAME, level="DEBUG")

    with caplog.at_level(logging.DEBUG, logger=PKG_LOGGER_NAME):
        logger.debug("debug-hidden")
        logger.info("info-hidden")
        logger.error("error-visible")

    # Only ERROR should appear in console capture, file gets DEBUG but we don't rely on that here
    assert "error-visible" in caplog.text
    assert "info-hidden" not in caplog.text
    assert "debug-hidden" not in caplog.text


def test_json_console_output_and_stderr(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "jsonconsole.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))
    monkeypatch.setenv("PR_LOG_JSON", "true")
    monkeypatch.setenv("PR_LOG_STDERR", "true")

    reset_logging(PKG_LOGGER_NAME)
    logger = setup_logger(name=PKG_LOGGER_NAME, level="INFO")

    # Capture stderr (console handler should be writing there now)
    logger.info("hello-json", extra={"run_id": "abc123"})
    captured = capsys.readouterr()
    # stdout empty, stderr contains JSON line
    assert captured.out == ""
    assert captured.err.strip()

    # Validate JSON shape
    line = captured.err.strip().splitlines()[-1]
    payload = json.loads(line)
    # expected fields
    assert payload["level"] == "INFO"
    assert payload["name"] == PKG_LOGGER_NAME
    assert payload["msg"] == "hello-json"
    assert payload["run_id"] == "abc123"
    # UTC suffix 'Z' expected (formatter sets UTC)
    assert payload["ts"].endswith("Z")


def test_file_handler_writes_and_rotates(monkeypatch, tmp_path):
    log_path = tmp_path / "rotate.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))
    # make rotation tiny to force rollover
    monkeypatch.setenv("PR_LOG_MAX_BYTES", "256")
    monkeypatch.setenv("PR_LOG_BACKUPS", "2")

    reset_logging(PKG_LOGGER_NAME)
    logger = setup_logger(name=PKG_LOGGER_NAME, level="INFO")
    # Emit enough data to rotate a couple of times
    for i in range(200):
        logger.debug("x" * 40)  # file handler logs at DEBUG

    # Base file should exist
    assert log_path.exists()
    # Some backups should exist (at least .1)
    backups = list(log_path.parent.glob("rotate.log*"))
    assert any(p.name.endswith(".1") for p in backups)


def test_console_formatter_utc_suffix(monkeypatch, tmp_path, capsys):
    # Use human-readable console (json off)
    log_path = tmp_path / "utc.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))
    monkeypatch.setenv("PR_LOG_JSON", "false")
    monkeypatch.setenv("PR_LOG_STDERR", "false")  # to stdout

    reset_logging(PKG_LOGGER_NAME)
    logger = setup_logger(name=PKG_LOGGER_NAME, level="INFO")
    logger.info("utc-line")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    # Example: "2025-08-20 21:10:00Z | INFO | protein_representation | utc-line"
    assert line.endswith(" | utc-line")
    # Ensure we have the "Z" after time
    assert "Z | INFO |" in line


def test_add_context_injection_with_json(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "ctx.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))
    monkeypatch.setenv("PR_LOG_JSON", "true")

    reset_logging(PKG_LOGGER_NAME)
    logger = setup_logger(name=PKG_LOGGER_NAME, level="INFO")
    add_context(logger, project="pr", phase="train")
    logger.info("ctx-msg", extra={"epoch": 1})

    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    # context from filter + extra
    assert payload["project"] == "pr"
    assert payload["phase"] == "train"
    assert payload["epoch"] == 1


def test_reset_logging_removes_handlers(monkeypatch, tmp_path):
    log_path = tmp_path / "reset.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))

    reset_logging(PKG_LOGGER_NAME)
    lg = setup_logger(name=PKG_LOGGER_NAME, level="INFO")
    assert _count_handlers(lg) >= 2

    reset_logging(PKG_LOGGER_NAME)
    lg2 = logging.getLogger(PKG_LOGGER_NAME)
    assert _count_handlers(lg2) == 0
    # Ensure we can configure again after reset
    lg3 = setup_logger(name=PKG_LOGGER_NAME, level="DEBUG")
    assert _count_handlers(lg3) >= 2


def test_get_logger_lazy_config(monkeypatch, tmp_path):
    log_path = tmp_path / "lazy.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))

    # Make sure clean slate
    reset_logging(PKG_LOGGER_NAME)
    # get_logger should configure defaults if not yet configured
    lg = get_logger(PKG_LOGGER_NAME)
    assert _count_handlers(lg) >= 2
    lg.info("hello")
    assert log_path.exists()


def test_silence_external_changes_level(monkeypatch, tmp_path):
    log_path = tmp_path / "silence.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))

    reset_logging(PKG_LOGGER_NAME)
    setup_logger(name=PKG_LOGGER_NAME, level="DEBUG")

    noisy = logging.getLogger("urllib3")
    noisy.setLevel(logging.DEBUG)
    silence_external(["urllib3"], level=logging.ERROR)
    assert noisy.level == logging.ERROR


def test_global_disable(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "disable.log"
    monkeypatch.setenv("PR_LOG_FILE", str(log_path))
    monkeypatch.setenv("PR_LOG_DISABLE", "true")

    reset_logging(PKG_LOGGER_NAME)
    lg = setup_logger(name=PKG_LOGGER_NAME, level="DEBUG")
    assert lg.disabled is True

    lg.info("no-output")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    # file should not be created either (handler never emits)
    assert not log_path.exists()


def test_set_global_level_affects_root(capsys):
    # This doesn't depend on our package logger directly.
    root = logging.getLogger()
    set_global_level("ERROR")
    assert root.level == logging.ERROR
