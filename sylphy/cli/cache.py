"""sylphy/cli/cache.py

Fast-start CLI utilities to inspect and manage Sylphy's cache directory.

Keeps the same UX as your current version, but with lazy imports to speed up startup.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import typer

# ----------------------------
# Typer Application
# ----------------------------
app = typer.Typer(
    name="cache",
    help="Inspect and manage Sylphy's cache directory (list, stats, prune, rm).",
    no_args_is_help=True,
)

# ----------------------------
# Utilities (pure stdlib -> fast import)
# ----------------------------
_SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}
_SIZE_RE = re.compile(r"^\s*(\d+)\s*(B|KB|MB|GB|TB)?\s*$", re.IGNORECASE)
_DELTA_RE = re.compile(
    r"^\s*((?P<days>\d+)d)?((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s)?\s*$",
    re.IGNORECASE,
)


def _human_size(num_bytes: int) -> str:
    for unit, factor in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)):
        if num_bytes >= factor:
            return f"{num_bytes} B" if unit == "B" else f"{num_bytes / factor:.2f} {unit}"
    return "0 B"


def _parse_size(text: str) -> int:
    m = _SIZE_RE.match(text or "")
    if not m:
        raise typer.BadParameter(f"Invalid size: {text}")
    value, unit = m.group(1), (m.group(2) or "B").upper()
    return int(value) * _SIZE_UNITS[unit]


def _parse_timedelta(text: str) -> timedelta:
    m = _DELTA_RE.match(text or "")
    if not m or m.group(0).strip() == "":
        raise typer.BadParameter(f"Invalid timedelta: {text!r} (use forms like 30d, 12h, 15m, 7d12h)")
    return timedelta(
        days=int(m.group("days") or 0),
        hours=int(m.group("hours") or 0),
        minutes=int(m.group("minutes") or 0),
        seconds=int(m.group("seconds") or 0),
    )


# ----------------------------
# Lazy helpers (avoid heavy imports at import-time)
# ----------------------------
def _console():
    """Return a rich Console if available, else None (lazy import)."""
    try:
        from rich.console import Console  # type: ignore

        return Console(stderr=False)
    except Exception:
        return None


def _table():
    """Return (Table, box) lazily if rich is available, else (None, None)."""
    try:
        from rich import box  # type: ignore
        from rich.table import Table  # type: ignore

        return Table, box
    except Exception:
        return None, None


def _user_cache_dir(app: str, vendor: str) -> Path | None:
    """Lazy import appdirs if present."""
    try:
        from appdirs import user_cache_dir  # type: ignore
    except Exception:
        return None
    return Path(user_cache_dir(app, vendor)).expanduser().resolve()


# ----------------------------
# Cache Discovery
# ----------------------------
DEFAULT_VENDOR = "KREN AI LAB"
DEFAULT_APP = "sylphy"
ENV_CACHE = "SYLPHY_CACHE_DIR"


def _default_cache_dir() -> Path:
    # Priority: env var -> appdirs -> ~/.cache/sylphy
    env = os.getenv(ENV_CACHE)
    if env:
        return Path(env).expanduser().resolve()
    appdirs = _user_cache_dir(DEFAULT_APP, DEFAULT_VENDOR)
    if appdirs is not None:
        return appdirs
    base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / DEFAULT_APP).expanduser().resolve()


# ----------------------------
# Data Structures
# ----------------------------
@dataclass(frozen=True)
class CacheEntry:
    path: Path
    size: int
    mtime: float  # POSIX timestamp
    is_dir: bool

    @property
    def mtime_dt(self) -> datetime:
        return datetime.fromtimestamp(self.mtime, tz=timezone.utc).astimezone()


class CacheManager:
    """Helper to inspect and manipulate the cache directory (stdlib-only)."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = (cache_dir or _default_cache_dir()).resolve()

    # ---------- Inspect ----------
    def iter_entries(
        self,
        pattern: str | None = None,
        recursive: bool = False,
        include_dirs: bool = False,
    ) -> Iterator[CacheEntry]:
        base = self.cache_dir
        if not base.exists():
            return
        glob = pattern or ("**/*" if recursive else "*")
        for p in base.glob(glob):
            try:
                if not p.exists():
                    continue
                if p.is_dir():
                    if include_dirs:
                        st = p.stat()
                        yield CacheEntry(p, 0, st.st_mtime, True)
                    continue
                st = p.stat()
                yield CacheEntry(p, st.st_size, st.st_mtime, False)
            except OSError:
                continue  # skip unreadable entries

    def du(self) -> tuple[int, int]:
        files = 0
        total = 0
        for e in self.iter_entries(recursive=True):
            if not e.is_dir:
                files += 1
                total += e.size
        return files, total

    # ---------- Mutate ----------
    def rm(
        self,
        pattern: str | None = None,
        older_than: timedelta | None = None,
        dry_run: bool = False,
    ) -> tuple[int, int]:
        now = datetime.now(timezone.utc)
        deleted = freed = 0
        for e in list(self.iter_entries(pattern=pattern, recursive=True)):
            if e.is_dir:
                continue
            if older_than is not None and (now - e.mtime_dt) < older_than:
                continue
            if not dry_run:
                try:
                    e.path.unlink(missing_ok=True)
                except OSError:
                    continue
            deleted += 1
            freed += e.size
        return deleted, freed

    def prune_empty_dirs(self) -> int:
        count = 0
        base = self.cache_dir
        if not base.exists():
            return 0
        for p in sorted(base.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                    count += 1
                except OSError:
                    pass
        return count

    def prune_to_max_size(self, max_bytes: int, dry_run: bool = False) -> tuple[int, int]:
        entries = [e for e in self.iter_entries(recursive=True) if not e.is_dir]
        total = sum(e.size for e in entries)
        if total <= max_bytes:
            return 0, 0
        entries.sort(key=lambda e: e.mtime)  # oldest first
        deleted = freed = 0
        for e in entries:
            if total - freed <= max_bytes:
                break
            if not dry_run:
                try:
                    e.path.unlink(missing_ok=True)
                except OSError:
                    continue
            deleted += 1
            freed += e.size
        return deleted, freed


# ----------------------------
# Presentation helpers
# ----------------------------
SORT_CHOICES = {"name", "size", "mtime"}


def _sort_entries(entries: list[CacheEntry], sort: str, reverse: bool) -> list[CacheEntry]:
    key = {
        "name": lambda e: str(e.path).lower(),
        "size": lambda e: e.size,
        "mtime": lambda e: e.mtime,
    }[sort]
    return sorted(entries, key=key, reverse=reverse)


def _print_kv(con, key: str, value: str) -> None:
    if con:
        con.print(f"[bold]{key}[/bold] {value}")
    else:
        typer.echo(f"{key} {value}")


# ----------------------------
# Commands
# ----------------------------
@app.callback()
def _callback() -> None:
    con = _console()
    msg = f"Cache dir: {_default_cache_dir()}  (set {ENV_CACHE} to override)"
    if con:
        con.print(f"[dim]{msg}[/dim]")
    else:
        typer.echo(msg)


@app.command("ls")
def cmd_ls(
    pattern: str | None = typer.Option(None, "--pattern", "-p", help="Glob pattern (e.g., '**/*.pt')."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
    sort: str = typer.Option("name", "--sort", case_sensitive=False, help="Sort by: name | size | mtime"),
    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    human_readable: bool = typer.Option(True, "--human-readable/--bytes", help="Pretty sizes."),
    limit: int | None = typer.Option(None, "--limit", help="Show only first N entries."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON instead of a table."),
) -> None:
    sort = sort.lower()
    if sort not in SORT_CHOICES:
        raise typer.BadParameter(f"--sort must be one of: {', '.join(sorted(SORT_CHOICES))}")

    mgr = CacheManager()
    entries = [e for e in mgr.iter_entries(pattern=pattern, recursive=recursive) if not e.is_dir]
    entries = _sort_entries(entries, sort, reverse)
    if limit is not None:
        entries = entries[: max(0, limit)]

    if json_out:
        payload = [
            {"path": str(e.path.relative_to(mgr.cache_dir)), "size": e.size, "mtime": e.mtime_dt.isoformat()}
            for e in entries
        ]
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    Table, box = _table()
    con = _console()
    if con and Table and box:
        table = Table(title=None, box=box.SIMPLE_HEAVY)
        table.add_column("Path", overflow="fold")
        table.add_column("Size", justify="right")
        table.add_column("Modified", justify="left")
        for e in entries:
            size = _human_size(e.size) if human_readable else str(e.size)
            table.add_row(str(e.path.relative_to(mgr.cache_dir)), size, e.mtime_dt.strftime("%Y-%m-%d %H:%M"))
        con.print(table)
    else:
        for e in entries:
            size = _human_size(e.size) if human_readable else str(e.size)
            typer.echo(f"{e.path.relative_to(mgr.cache_dir)}\t{size}\t{e.mtime_dt:%Y-%m-%d %H:%M}")


@app.command("stats")
def cmd_stats() -> None:
    mgr = CacheManager()
    files, total = mgr.du()
    newest = oldest = None
    for e in mgr.iter_entries(recursive=True):
        if e.is_dir:
            continue
        newest = e if (newest is None or e.mtime > newest.mtime) else newest
        oldest = e if (oldest is None or e.mtime < oldest.mtime) else oldest

    con = _console()
    _print_kv(con, "Cache:", str(mgr.cache_dir))
    _print_kv(con, "Files:", str(files))
    _print_kv(con, "Total size:", f"{_human_size(total)} ({total} B)")
    if newest:
        _print_kv(con, "Newest:", f"{newest.path.name} @ {newest.mtime_dt:%Y-%m-%d %H:%M}")
    if oldest:
        _print_kv(con, "Oldest:", f"{oldest.path.name} @ {oldest.mtime_dt:%Y-%m-%d %H:%M}")


@app.command("rm")
def cmd_rm(
    pattern: str | None = typer.Option(None, "--pattern", "-p", help="Glob like '**/*.tmp'."),
    older_than: str | None = typer.Option(
        None, "--older-than", help="Delete files older than given age (e.g., '30d', '12h')."
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Show what would be removed."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation when applying."),
) -> None:
    mgr = CacheManager()
    td = _parse_timedelta(older_than) if older_than else None
    candidates = [
        e
        for e in mgr.iter_entries(pattern=pattern, recursive=True)
        if not e.is_dir and (td is None or (datetime.now(timezone.utc) - e.mtime_dt) >= td)
    ]
    total_bytes = sum(e.size for e in candidates)
    con = _console()
    _print_kv(con, "Candidates:", f"{len(candidates)} files, total {_human_size(total_bytes)}")

    if not dry_run and not force and not typer.confirm("Proceed with deletion?"):
        raise typer.Abort()

    deleted, freed = mgr.rm(pattern=pattern, older_than=td, dry_run=dry_run)
    _print_kv(con, "Result:", f"Deleted {deleted} files, freed {_human_size(freed)}")


@app.command("prune")
def cmd_prune(
    max_size: str | None = typer.Option(
        None, "--max-size", help="Ensure total cache size <= VALUE by deleting oldest files (e.g., 10GB)."
    ),
    remove_empty_dirs: bool = typer.Option(
        True, "--prune-empty/--keep-empty", help="Remove empty directories."
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Show what would be removed."),
) -> None:
    mgr = CacheManager()
    deleted = freed = 0

    if max_size:
        max_bytes = _parse_size(max_size)
        if dry_run:
            entries = [e for e in mgr.iter_entries(recursive=True) if not e.is_dir]
            total = sum(e.size for e in entries)
            entries.sort(key=lambda e: e.mtime)
            freed_sim = 0
            to_delete = 0
            for e in entries:
                if total - freed_sim <= max_bytes:
                    break
                freed_sim += e.size
                to_delete += 1
            con = _console()
            _print_kv(
                con,
                "Prune (simulated):",
                f"Would delete {to_delete} files to reach {max_size} (free {_human_size(freed_sim)}).",
            )
        else:
            deleted, freed = mgr.prune_to_max_size(max_bytes=max_bytes, dry_run=False)

    removed_dirs = 0
    if remove_empty_dirs:
        if dry_run:
            empties = 0
            for p in sorted(mgr.cache_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
                if p.is_dir() and not any(p.iterdir()):
                    empties += 1
            removed_dirs = len([None] * empties)
        else:
            removed_dirs = mgr.prune_empty_dirs()

    con = _console()
    if not dry_run and max_size:
        _print_kv(con, "Prune:", f"Deleted {deleted} files, freed {_human_size(freed)}")
    _print_kv(con, "Removed empty dirs:", str(removed_dirs))


@app.command("path")
def cmd_path() -> None:
    typer.echo(str(_default_cache_dir()))


@app.command("clear")
def cmd_clear(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    mgr = CacheManager()
    if not mgr.cache_dir.exists():
        typer.echo("Cache directory does not exist.")
        raise typer.Exit(code=0)
    if not force and not typer.confirm(f"This will permanently remove {mgr.cache_dir}. Continue?"):
        raise typer.Abort()
    try:
        shutil.rmtree(mgr.cache_dir)
        typer.echo("Cache cleared.")
    except Exception as exc:
        raise typer.Exit(code=1) from exc
