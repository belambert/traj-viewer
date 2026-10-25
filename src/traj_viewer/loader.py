"""Load trajectories from the Hugging Face Hub or local files.

Hub datasets are read from the Hub's auto-converted parquet branch
(`refs/convert/parquet`), which has a uniform `{config}/{split}/*.parquet`
layout regardless of the dataset's original format. Rows are read one parquet
row group at a time, and row groups before `start` are skipped using the file
footer, so peeking at a few rows of a large dataset downloads very little.

We read parquet directly rather than via `datasets` streaming, whose open file
handles hang the interpreter at exit.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from traj_viewer.adapters import Adapter, detect_adapter, get_adapter
from traj_viewer.models import Trajectory

_LOCAL_SUFFIXES = {".jsonl", ".json", ".parquet", ".db"}
OPENCODE_DB = Path("~/.local/share/opencode/opencode.db").expanduser()
_BLOCK_SIZE = 8 << 20  # bigger HTTP range reads; row groups are often tens of MB


def _fs() -> HfFileSystem:
    return HfFileSystem()


def _parquet_root(repo: str) -> str:
    return f"datasets/{repo}@~parquet"


def _is_local(path: str) -> bool:
    return Path(path).suffix.lower() in _LOCAL_SUFFIXES and Path(path).exists()


def list_configs(repo: str) -> list[str]:
    entries = _fs().ls(_parquet_root(repo), detail=True)
    return sorted(e["name"].rsplit("/", 1)[-1] for e in entries if e["type"] == "directory")


def list_splits(repo: str, config: str) -> list[str]:
    # Very large splits are converted only partially, into `partial-{split}`.
    paths = _fs().ls(f"{_parquet_root(repo)}/{config}", detail=False)
    return sorted(p.rsplit("/", 1)[-1].removeprefix("partial-") for p in paths)


@dataclass
class Source:
    """A resolved dataset location: path/repo plus config and split."""

    path: str
    config: str | None = None
    split: str | None = None

    @property
    def label(self) -> str:
        parts = [self.path] + [p for p in (self.config, self.split) if p]
        return ":".join(parts)

    def resolve(self) -> Source:
        """Fill in default config/split for Hub datasets; `opencode` means the local opencode DB."""
        if self.path == "opencode":
            return Source(str(OPENCODE_DB))
        if _is_local(self.path):
            return Source(self.path)
        config = self.config or list_configs(self.path)[0]
        split = self.split or list_splits(self.path, config)[0]
        return Source(self.path, config, split)


def _iter_parquet(open_file, start: int) -> Iterator[dict[str, Any]]:
    """Yield rows from one parquet file, skipping the first `start` rows."""
    with open_file() as f:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            if start >= n:
                start -= n
                continue
            yield from pf.read_row_group(rg).slice(start).to_pylist()
            start = 0


def _iter_opencode(path: Path, start: int) -> Iterator[dict[str, Any]]:
    """Yield one row per opencode session, with its messages and their parts inlined."""
    # read-only so a running opencode isn't disturbed
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        for s in con.execute("select * from session order by time_created limit -1 offset ?", (start,)):
            messages = []
            for m in con.execute("select id, data from message where session_id = ? order by time_created, id", (s["id"],)):
                parts = [json.loads(p["data"]) for p in con.execute("select data from part where message_id = ? order by id", (m["id"],))]
                messages.append({**json.loads(m["data"]), "parts": parts})
            yield {"instance_id": s["id"], "session": dict(s), "oc_messages": messages}


def _hub_files(source: Source) -> list[str]:
    base = f"{_parquet_root(source.path)}/{source.config}"
    fs = _fs()
    for split_dir in (source.split, f"partial-{source.split}"):
        files = sorted(fs.glob(f"{base}/{split_dir}/*.parquet"))
        if files:
            return files
    raise FileNotFoundError(f"No parquet files for {source.label}")


def _iter_rows(source: Source, start: int) -> Iterator[dict[str, Any]]:
    if _is_local(source.path):
        path = Path(source.path)
        if path.suffix.lower() == ".db":
            yield from _iter_opencode(path, start)
        elif path.suffix.lower() == ".parquet":
            yield from _iter_parquet(lambda: open(path, "rb"), start)
        else:
            with open(path) as f:
                if path.suffix.lower() == ".json":
                    rows = json.load(f)
                    yield from rows[start:]
                else:
                    for i, line in enumerate(f):
                        if i >= start and line.strip():
                            yield json.loads(line)
        return

    fs = _fs()
    for file in _hub_files(source):
        n = pq.ParquetFile(fs.open(file, "rb")).metadata.num_rows if start else 0
        if start >= n > 0:  # skip whole files using just the footer
            start -= n
            continue
        yield from _iter_parquet(lambda: fs.open(file, "rb", block_size=_BLOCK_SIZE), start)
        start = 0


def iter_trajectories(
    source: Source,
    *,
    start: int = 0,
    limit: int | None = None,
    adapter: str | None = None,
    instance_id: str | None = None,
) -> Iterator[Trajectory]:
    """Yield normalized trajectories from `source`, starting at row `start`."""
    source = source.resolve()
    chosen: Adapter | None = get_adapter(adapter) if adapter else None
    yielded = 0
    for offset, row in enumerate(_iter_rows(source, start)):
        if limit is not None and yielded >= limit:
            return
        if chosen is None:
            chosen = detect_adapter(set(row))
        if instance_id and row.get("instance_id") != instance_id:
            continue
        yield chosen.convert(row, start + offset, source.label)
        yielded += 1


def load_one(source: Source, index: int, adapter: str | None = None) -> Trajectory:
    for traj in iter_trajectories(source, start=index, limit=1, adapter=adapter):
        return traj
    raise IndexError(f"No row at index {index} in {source.label}")
