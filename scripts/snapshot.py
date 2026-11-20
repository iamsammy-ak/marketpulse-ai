"""Build the bundled demo snapshots from live Google News RSS search feeds.

Fetches the last 30 days of headlines for each sample watchlist once (2-day query windows, two
requests at a time with a pause between them) and stores only title, outlet, link, publication
time and the feed's short summary under demo/snapshots/<id>.json.gz. Raw feed responses are cached
in .cache/feeds, so a re-run within a day costs no requests.

If a watchlist cannot be fetched (no network, blocked, too few headlines), a seeded synthetic
snapshot with fictional companies is written instead and clearly labelled as synthetic.

    python scripts/snapshot.py                    # fetch the real watchlists
    python scripts/snapshot.py --as-of ISO-TIME   # re-run with the same windows (cached ones are reused)
    python scripts/snapshot.py --synthetic        # write only the synthetic sample
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketpulse.demo import save_snapshot  # noqa: E402
from marketpulse.ingest import fetch_url, fetch_watchlist  # noqa: E402
from marketpulse.watchlist import Watchlist  # noqa: E402

DAYS = 30
STEP_DAYS = 2
MIN_HEADLINES = 150

WATCHLISTS = [
    {
        "id": "food-delivery",
        "watchlist": {
            "name": "Food delivery",
            "competitors": [
                {"name": "DoorDash", "aliases": []},
                {"name": "Uber Eats", "aliases": []},
                {"name": "Instacart", "aliases": ["Maplebear"]},
                {"name": "Deliveroo", "aliases": []},
                {"name": "Grubhub", "aliases": []},
            ],
            "topics": ["food delivery", "gig workers"],
        },
    },
    {
        "id": "ai-chips",
        "watchlist": {
            "name": "AI chips",
            "competitors": [
                {"name": "Nvidia", "aliases": ["NVDA"]},
                {"name": "AMD", "aliases": ["Advanced Micro Devices"]},
                {"name": "Intel", "aliases": ["INTC"]},
                {"name": "Qualcomm", "aliases": []},
                {"name": "Broadcom", "aliases": ["AVGO"]},
            ],
            "topics": ["AI chips", "chip export controls"],
        },
    },
    {
        "id": "electric-vehicles",
        "watchlist": {
            "name": "Electric vehicles",
            "competitors": [
                {"name": "Tesla", "aliases": ["TSLA"]},
                {"name": "BYD", "aliases": []},
                {"name": "Rivian", "aliases": ["RIVN"]},
                {"name": "Lucid", "aliases": ["Lucid Motors", "Lucid Group"],
                 "exclude": ["Lucid Capital", "Lucid Computing"]},
                {"name": "Polestar", "aliases": []},
            ],
            "topics": ["electric vehicles", "EV charging"],
        },
    },
]


def _fetch(url: str) -> tuple[bytes, bool]:
    return fetch_url(url, ttl=24 * 3600)


def write_synthetic(order: int = 9) -> None:
    from marketpulse.synthetic import generate_dataset

    path = save_snapshot(generate_dataset(), order=order)
    print(f"saved  synthetic sample -> {path.name}")


def main(argv: list[str]) -> int:
    if "--synthetic" in argv:
        write_synthetic()
        return 0
    now = datetime.now(timezone.utc).replace(microsecond=0)
    if "--as-of" in argv:
        now = datetime.fromisoformat(argv[argv.index("--as-of") + 1])
    failed = 0
    for order, spec in enumerate(WATCHLISTS, start=1):
        wl = Watchlist.from_dict(spec["watchlist"])
        print(f"fetch  {wl.name}: {len(wl.competitors)} competitors, {len(wl.topics)} topics")
        ds = fetch_watchlist(
            wl, days=DAYS, step=STEP_DAYS, now=now, fetch=_fetch, max_workers=2, delay=1.0,
            on_progress=lambda done, total: print(f"       {done}/{total}", end="\r"),
        )
        errors = sum(1 for row in ds.fetch_log if row["status"] == "error")
        capped = sum(1 for row in ds.fetch_log if row["capped"])
        print(f"       {len(ds.articles)} unique links, {errors} failed requests, {capped} capped windows")
        if len(ds.articles) < MIN_HEADLINES or errors > len(ds.fetch_log) // 2:
            failed += 1
            print(f"FAILED {wl.name}: too few headlines; not saved")
            continue
        path = save_snapshot(replace(ds, id=spec["id"]), order=order)
        print(f"saved  {wl.name} -> {path.name}")
    if failed == len(WATCHLISTS):
        print("No live watchlist could be fetched: writing the synthetic fallback sample instead.")
        write_synthetic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
