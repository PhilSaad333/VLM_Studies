"""Utilities for caching interesting NLVR2 sample UIDs.

The cache is persisted as JSON and keeps insertion order while preventing
duplicates. Intended for notebook use so Lord Krang can quickly revisit
specific examples without re-querying the dataset.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class UIDCache:
    """Simple persistent cache for NLVR2 example UIDs."""

    path: Path
    uids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.uids = [str(uid) for uid in data]
            except json.JSONDecodeError:
                # Start fresh if the file is corrupted.
                self.uids = []

    def add(self, uid: str) -> None:
        if uid not in self.uids:
            self.uids.append(uid)
            self._save()

    def extend(self, new_uids: Iterable[str]) -> None:
        updated = False
        for uid in new_uids:
            if uid not in self.uids:
                self.uids.append(uid)
                updated = True
        if updated:
            self._save()

    def remove(self, uid: str) -> None:
        if uid in self.uids:
            self.uids.remove(uid)
            self._save()

    def clear(self) -> None:
        if self.uids:
            self.uids.clear()
            self._save()

    def get(self, index: int) -> str:
        if not self.uids:
            raise IndexError("UID cache is empty")
        return self.uids[index % len(self.uids)]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.uids)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self.uids)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.uids, indent=2), encoding="utf-8")


def cycle_uids(cache: UIDCache, start: Optional[str] = None):
    """Yield UIDs cyclically, starting after ``start`` if provided."""

    if not cache.uids:
        return

    idx = 0
    if start is not None and start in cache.uids:
        idx = (cache.uids.index(start) + 1) % len(cache.uids)

    while True:
        yield cache.uids[idx]
        idx = (idx + 1) % len(cache.uids)
