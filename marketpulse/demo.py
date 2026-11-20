"""Bundled snapshot watchlists and saved AI analyses, so the app works without a key or network."""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

from .ingest import Dataset
from .watchlist import Watchlist

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
SNAPSHOTS_DIR = DEMO_DIR / "snapshots"
ANALYSES_DIR = DEMO_DIR / "analyses"


def dataset_to_dict(ds: Dataset, order: int = 99) -> dict:
    return {
        "id": ds.id,
        "order": order,
        "watchlist": ds.watchlist.to_dict(),
        "fetched_at": ds.fetched_at.isoformat(timespec="seconds"),
        "days": ds.days,
        "source": ds.source,
        "synthetic": ds.synthetic,
        "fetch_log": [dict(row) for row in ds.fetch_log],
        "articles": list(ds.articles),
    }


def dataset_from_dict(data: dict) -> Dataset:
    return Dataset(
        id=data["id"],
        watchlist=Watchlist.from_dict(data["watchlist"]),
        articles=tuple(data["articles"]),
        fetched_at=datetime.fromisoformat(data["fetched_at"].replace("Z", "+00:00")),
        days=int(data.get("days", 30)),
        source=data.get("source", "Google News RSS"),
        synthetic=bool(data.get("synthetic", False)),
        fetch_log=tuple(data.get("fetch_log", [])),
    )


def load_snapshots() -> list[Dataset]:
    """Every bundled snapshot, in display order."""
    raw = [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(SNAPSHOTS_DIR.glob("*.json.gz"))]
    raw.sort(key=lambda d: (d.get("order", 99), d["id"]))
    return [dataset_from_dict(d) for d in raw]


def save_snapshot(ds: Dataset, order: int = 99) -> Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS_DIR / f"{ds.id}.json.gz"
    payload = json.dumps(dataset_to_dict(ds, order), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(gzip.compress(payload, mtime=0))
    return path


def load_saved_analysis(fingerprint: str) -> dict | None:
    path = ANALYSES_DIR / f"{fingerprint}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_analysis(analysis: dict) -> Path:
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSES_DIR / f"{analysis['fingerprint']}.json"
    path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
