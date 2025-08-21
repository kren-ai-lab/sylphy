"""protein_representation/cli/cache.py

CLI utilities to inspect and manage the protein_representation cache directory.

Usage examples
--------------
$ protein_representation cache ls
$ protein_representation cache ls --recursive --sort size --human-readable
$ protein_representation cache ls --pattern "**/*.pt" --json
$ protein_representation cache stats
$ protein_representation cache rm --older-than 30d --dry-run
$ protein_representation cache prune --max-size 10GB

"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import typer

try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
except Exception:  # pragma: no cover - optional dependency
    Console = None  # type: ignore

try:
    from appdirs import user_cache_dir
except Exception:  # pragma: no cover - optional dependency
    user_cache_dir = None  # type: ignore

# ----------------------------
# Typer Application
# ----------------------------
app = typer.Typer(
    name="cache",
    help="Inspect and manage the protein_representation cache directory (list, stats, prune, rm).",
    no_args_is_help=True,
)


# ----------------------------
# Utilities
# ----------------------------
_SIZE_UNITS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024**2,
    "GB": 1024**3,
    "TB": 1024**4,
}


def _human_size(num_bytes: int) -> str:
    """Return human readable size (base 1024)."""
    for unit, factor in ("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1):
        if num_bytes >= factor:
            if unit == "B":
                return f"{num_bytes} B"
            return f"{num_bytes / factor:.2f} {unit}"
    return "0 B"


def _parse_size(text: str) -> int:
    """Parse human size like '10GB', '500 MB', '2048' (bytes)."""
    s = text.strip().upper().replace(" ", "")
    m = re.fullmatch(r"(\d+)(B|KB|MB|GB|TB)?", s)
    if not m:
        raise typer.BadParameter(f"Invalid size: {text}")
    value, unit = m.group(1), m.group(2) or "B"
    return int(value) * _SIZE_UNITS[unit]


def _parse_timedelta(text: str) -> timedelta:
    """Parse a compact timedelta like '30d', '12h', '15m', '7d12h'."""
    s = text.strip().lower().replace(" ", "")
    if not s:
        raise typer.BadParameter("Empty timedelta string")
    pattern = r"((?P<days>\d+)d)?((?P<hours>\d+)h)?((?P<minutes>\d+)m)?((?P<seconds>\d+)s)?"
    m = re.fullmatch(pattern, s)
    if not m or m.group(0) == "":
        raise typer.BadParameter(f"Invalid timedelta: {text}")
    days = int(m.group("days") or 0)
    hours = int(m.group("hours") or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


# ----------------------------
# Cache Discovery
# ----------------------------
DEFAULT_VENDOR = "KREN AI LAB"
DEFAULT_APP = "protein_representation"
ENV_CACHE = "protein_representation_CACHE_DIR"


def _default_cache_dir() -> Path:
    # Priority: env var -> appdirs -> ~/.cache/protein_representation
    env = os.getenv(ENV_CACHE)
    if env:
        return Path(env).expanduser().resolve()
    if user_cache_dir is not None:  # type: ignore
        return Path(user_cache_dir(DEFAULT_APP, DEFAULT_VENDOR)).expanduser().resolve()
    # Fallback: XDG-compliant-ish
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
    """Helper to inspect and manipulate the cache directory."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = (cache_dir or _default_cache_dir()).resolve()

    # ---------- Inspect ----------
    def iter_entries(
        self,
        pattern: Optional[str] = None,
        recursive: bool = False,
        include_dirs: bool = False,
    ) -> Iterator[CacheEntry]:
        if not self.cache_dir.exists():
            return
        base = self.cache_dir
        glob = "**/*" if recursive else "*"
        if pattern:
            glob = pattern
        for p in base.glob(glob):
            try:
                if not p.exists():
                    continue
                if p.is_dir():
                    if include_dirs:
                        stat = p.stat()
                        yield CacheEntry(p, 0, stat.st_mtime, True)
                    continue
                stat = p.stat()
                yield CacheEntry(p, stat.st_size, stat.st_mtime, False)
            except OSError:
                # Skip unreadable entries
                continue

    def du(self) -> Tuple[int, int]:
        """Return (num_files, total_bytes)."""
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
        pattern: Optional[str] = None,
        older_than: Optional[timedelta] = None,
        dry_run: bool = False,
    ) -> Tuple[int, int]:
        """Delete files by pattern and/or age. Returns (deleted_count, freed_bytes)."""
        now = datetime.now(timezone.utc)
        deleted = 0
        freed = 0
        for e in list(self.iter_entries(pattern=pattern, recursive=True)):
            if e.is_dir:
                continue
            if older_than is not None:
                age = now - e.mtime_dt
                if age < older_than:
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
        """Remove empty directories under cache. Returns number of dirs removed."""
        count = 0
        if not self.cache_dir.exists():
            return 0
        # Walk deepest-first
        for p in sorted(self.cache_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            if p.is_dir():
                try:
                    p.rmdir()
                    count += 1
                except OSError:
                    pass
        return count

    def prune_to_max_size(self, max_bytes: int, dry_run: bool = False) -> Tuple[int, int]:
        """Ensure cache total size <= max_bytes by deleting oldest files first.
        Returns (deleted_count, freed_bytes).
        """
        # Collect files sorted by mtime ascending (oldest first)
        entries = [e for e in self.iter_entries(recursive=True) if not e.is_dir]
        total = sum(e.size for e in entries)
        if total <= max_bytes:
            return 0, 0
        entries.sort(key=lambda e: e.mtime)  # oldest first
        deleted = 0
        freed = 0
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

def _console() -> Optional[Console]:  # type: ignore[name-defined]
    return Console(stderr=False) if Console else None  # type: ignore


SORT_CHOICES = {"name", "size", "mtime"}

def _sort_entries(entries: List[CacheEntry], sort: str, reverse: bool) -> List[CacheEntry]:
    key = {
        "name": lambda e: str(e.path).lower(),
        "size": lambda e: e.size,
        "mtime": lambda e: e.mtime,
    }[sort]
    return sorted(entries, key=key, reverse=reverse)


# ----------------------------
# Commands
# ----------------------------


@app.callback()
def _callback() -> None:
    """Additional help and environment notes."""
    if _console():
        _console().print(
            f"[dim]Cache dir:[/dim] {_default_cache_dir()}  [dim](set {ENV_CACHE} to override)[/dim]"
        )


@app.command("ls")
def cmd_ls(
    pattern: Optional[str] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Glob pattern relative to cache dir (e.g., '**/*.pt').",
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories."),
    sort: str = typer.Option("name", "--sort", case_sensitive=False, help="Sort by: name | size | mtime"),    reverse: bool = typer.Option(False, "--reverse", help="Reverse sort order."),
    human_readable: bool = typer.Option(True, "--human-readable/--bytes", help="Pretty sizes."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Show only the first N entries after sorting."),
    json_out: bool = typer.Option(False, "--json", help="Output JSON instead of a table."),
) -> None:
    
    if sort.lower() not in SORT_CHOICES:
        raise typer.BadParameter(f"--sort must be one of: {', '.join(sorted(SORT_CHOICES))}")

    """List cache assets with size and modification date."""
    mgr = CacheManager()
    entries = list(mgr.iter_entries(pattern=pattern, recursive=recursive))
    entries = [e for e in entries if not e.is_dir]
    entries = _sort_entries(entries, sort, reverse)
    if limit is not None:
        entries = entries[: max(0, limit)]

    if json_out:
        payload = [
            {
                "path": str(e.path.relative_to(mgr.cache_dir)),
                "size": e.size,
                "mtime": e.mtime_dt.isoformat(),
            }
            for e in entries
        ]
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    con = _console()
    if con:
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
    """Show basic cache statistics (file count and total size)."""
    mgr = CacheManager()
    files, total = mgr.du()
    newest = None
    oldest = None
    for e in mgr.iter_entries(recursive=True):
        if e.is_dir:
            continue
        newest = e if (newest is None or e.mtime > newest.mtime) else newest
        oldest = e if (oldest is None or e.mtime < oldest.mtime) else oldest

    con = _console()
    if con:
        con.print(f"[bold]Cache:[/bold] {mgr.cache_dir}")
        con.print(f"Files: {files}")
        con.print(f"Total size: {_human_size(total)} ({total} B)")
        if newest:
            con.print(f"Newest: {newest.path.name} @ {newest.mtime_dt:%Y-%m-%d %H:%M}")
        if oldest:
            con.print(f"Oldest: {oldest.path.name} @ {oldest.mtime_dt:%Y-%m-%d %H:%M}")
    else:
        typer.echo(f"Cache: {mgr.cache_dir}")
        typer.echo(f"Files: {files}")
        typer.echo(f"Total size: {_human_size(total)} ({total} B)")
        if newest:
            typer.echo(f"Newest: {newest.path.name} @ {newest.mtime_dt:%Y-%m-%d %H:%M}")
        if oldest:
            typer.echo(f"Oldest: {oldest.path.name} @ {oldest.mtime_dt:%Y-%m-%d %H:%M}")


@app.command("rm")
def cmd_rm(
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="Glob like '**/*.tmp'."),
    older_than: Optional[str] = typer.Option(
        None, "--older-than", help="Delete files older than given age (e.g., '30d', '12h')."
    ),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Show what would be removed."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt when applying."),
) -> None:
    """Remove files by pattern and/or age."""
    mgr = CacheManager()
    td = _parse_timedelta(older_than) if older_than else None
    candidates = list(mgr.iter_entries(pattern=pattern, recursive=True))
    candidates = [e for e in candidates if not e.is_dir and (td is None or (datetime.now(timezone.utc) - e.mtime_dt) >= td)]
    total_bytes = sum(e.size for e in candidates)

    con = _console()
    if con:
        con.print(f"Candidates: {len(candidates)} files, total {_human_size(total_bytes)}")
    else:
        typer.echo(f"Candidates: {len(candidates)} files, total {_human_size(total_bytes)}")

    if not dry_run and not force:
        if not typer.confirm("Proceed with deletion?"):
            raise typer.Abort()

    deleted, freed = mgr.rm(pattern=pattern, older_than=td, dry_run=dry_run)
    msg = f"Deleted {deleted} files, freed {_human_size(freed)}"
    _console().print(msg) if _console() else typer.echo(msg)


@app.command("prune")
def cmd_prune(
    max_size: Optional[str] = typer.Option(
        None,
        "--max-size",
        help="Ensure total cache size is <= this value by deleting oldest files (e.g., 10GB).",
    ),
    remove_empty_dirs: bool = typer.Option(True, "--prune-empty/--keep-empty", help="Remove empty directories."),
    dry_run: bool = typer.Option(True, "--dry-run/--apply", help="Show what would be removed."),
) -> None:
    """Prune cache: enforce max size and/or remove empty directories."""
    mgr = CacheManager()
    deleted = freed = 0

    if max_size:
        max_bytes = _parse_size(max_size)
        # Estimate what would be deleted in dry-run by simulating
        if dry_run:
            entries = [e for e in mgr.iter_entries(recursive=True) if not e.is_dir]
            total = sum(e.size for e in entries)
            entries.sort(key=lambda e: e.mtime)
            freed_sim = 0
            to_delete = []
            for e in entries:
                if total - freed_sim <= max_bytes:
                    break
                freed_sim += e.size
                to_delete.append(e)
            con = _console()
            if con:
                con.print(f"Would delete {len(to_delete)} files to reach {max_size} (free {_human_size(freed_sim)}).")
            else:
                typer.echo(f"Would delete {len(to_delete)} files to reach {max_size} (free {_human_size(freed_sim)}).")
        else:
            deleted, freed = mgr.prune_to_max_size(max_bytes=max_bytes, dry_run=False)

    removed_dirs = 0
    if remove_empty_dirs:
        if dry_run:
            # Count empties without removing
            empties = 0
            for p in sorted(mgr.cache_dir.rglob("*"), key=lambda x: len(x.parts), reverse=True):
                if p.is_dir() and not any(p.iterdir()):
                    empties += 1
            removed_dirs = empties
        else:
            removed_dirs = mgr.prune_empty_dirs()

    con = _console()
    if con:
        if not dry_run and max_size:
            con.print(f"Deleted {deleted} files, freed {_human_size(freed)}.")
        con.print(f"Removed empty dirs: {removed_dirs}")
    else:
        if not dry_run and max_size:
            typer.echo(f"Deleted {deleted} files, freed {_human_size(freed)}.")
        typer.echo(f"Removed empty dirs: {removed_dirs}")


@app.command("path")
def cmd_path() -> None:
    """Print the cache directory path in the filesystem."""
    typer.echo(str(_default_cache_dir()))


@app.command("clear")
def cmd_clear(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    """Delete the entire cache directory. USE WITH CARE."""
    mgr = CacheManager()
    if not mgr.cache_dir.exists():
        typer.echo("Cache directory does not exist.")
        raise typer.Exit(code=0)

    if not force and not typer.confirm(f"This will permanently remove {mgr.cache_dir}. Continue?"):
        raise typer.Abort()

    try:
        shutil.rmtree(mgr.cache_dir)
        typer.echo("Cache cleared.")
    except Exception as exc:  # pragma: no cover - defensive
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
