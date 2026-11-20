"""Normalise raw headline records and merge near-duplicates.

Syndicated stories reappear under many outlets with slightly different titles. Titles are
compared with TF-IDF cosine similarity (word unigrams and bigrams, English stop words removed so
"the" and "to" do not link unrelated headlines); pairs above the threshold that were published
within a few days of each other and mention the same competitors are joined with union-find, and
each group keeps its earliest headline. The competitor check matters: templated headlines such as
"Why Nvidia stock is falling today" and "Why AMD stock is falling today" are textually close but
are different stories. The group size ("outlets carrying the story") is kept as a reach signal.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd

MAX_GAP_DAYS = 3.0  # syndicated copies appear within days; a recurring headline weeks later is a new story
_NON_WORD = re.compile(r"[^a-z0-9$%& ]+")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})

COLUMNS = ["id", "title", "source", "link", "published", "summary", "queries"]


def normalise_title(title: str) -> str:
    """Lowercase, unify quotes and dashes, drop punctuation and repeated spaces."""
    text = unicodedata.normalize("NFKC", title or "").translate(_QUOTES).lower().replace("'", "")
    return " ".join(_NON_WORD.sub(" ", text).split())


def to_frame(records: Iterable[dict]) -> pd.DataFrame:
    """Raw records → typed frame with a stable id and UTC timestamps, newest first."""
    rows = []
    for rec in records:
        title = " ".join(str(rec.get("title", "")).split())
        if not title:
            continue
        link = str(rec.get("link", ""))
        rows.append(
            {
                "id": hashlib.sha1((link or title).encode("utf-8")).hexdigest()[:12],
                "title": title,
                "source": str(rec.get("source") or "Unknown outlet"),
                "link": link,
                "published": rec.get("published"),
                "summary": str(rec.get("summary") or ""),
                "queries": list(rec.get("queries") or []),
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["published"]).drop_duplicates("id")
    return df.sort_values("published", ascending=False, kind="stable").reset_index(drop=True)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def near_duplicate_groups(
    titles: list[str],
    threshold: float = 0.75,
    times: Iterable[float] | None = None,
    max_gap_days: float = MAX_GAP_DAYS,
    keys: list | None = None,
    chunk: int = 512,
) -> list[int]:
    """Group index per title. Titles join a group when cosine similarity ≥ threshold and, if given,
    they were published within `max_gap_days` (`times` in epoch seconds) and have equal `keys`."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    n = len(titles)
    if n == 0:
        return []
    norm = [normalise_title(t) for t in titles]
    try:
        x = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english",
                            token_pattern=r"(?u)\b\w+\b").fit_transform(norm)
    except ValueError:  # empty vocabulary: nothing to compare
        return list(range(n))
    t = np.asarray(list(times), dtype=float) if times is not None else None
    uf = _UnionFind(n)
    for start in range(0, n, chunk):
        sims = (x[start : start + chunk] @ x.T).toarray()
        rows, cols = np.nonzero(sims >= threshold - 1e-9)
        for i, j in zip(rows + start, cols, strict=False):
            if i >= j:
                continue
            if t is not None and abs(t[i] - t[j]) > max_gap_days * 86400:
                continue
            if keys is not None and keys[i] != keys[j]:
                continue
            uf.union(int(i), int(j))
    # identical normalised titles are always duplicates, even if TF-IDF had nothing to work with
    first: dict[str, int] = {}
    for i, key in enumerate(norm):
        if key and key in first and (t is None or abs(t[i] - t[first[key]]) <= max_gap_days * 86400):
            if keys is None or keys[i] == keys[first[key]]:
                uf.union(i, first[key])
        first.setdefault(key, i)
    return [uf.find(i) for i in range(n)]


def deduplicate(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Keep the earliest headline of each near-duplicate group.

    Adds `copies` (headlines in the group), `outlets` (distinct outlets carrying it) and merges
    the `queries` that surfaced any copy. When the frame has a `competitors` column (from tagging),
    only headlines that mention the same competitors can merge.
    """
    if df.empty:
        return df.assign(copies=pd.Series(dtype=int), outlets=pd.Series(dtype=object))
    times = [ts.timestamp() for ts in df["published"]]  # resolution-independent epoch seconds
    keys = [tuple(c) for c in df["competitors"]] if "competitors" in df else None
    groups = near_duplicate_groups(df["title"].tolist(), threshold, times, keys=keys)
    work = df.assign(_group=groups).sort_values("published", kind="stable")
    sizes = work.groupby("_group").size()
    out = work.drop_duplicates("_group", keep="first").set_index("_group")
    out["copies"] = sizes.reindex(out.index).astype(int)
    multi = sizes[sizes > 1].index
    outlets: dict = {}
    queries: dict = {}
    if len(multi):  # only groups with copies need merging, which keeps this fast on large feeds
        dup = work[work["_group"].isin(multi)].groupby("_group")
        outlets = dup["source"].agg(lambda s: list(dict.fromkeys(s))).to_dict()
        queries = dup["queries"].agg(lambda s: list(dict.fromkeys(q for qs in s for q in qs))).to_dict()
    out["outlets"] = [outlets.get(g, [s]) for g, s in zip(out.index, out["source"], strict=True)]
    out["queries"] = [queries.get(g, q) for g, q in zip(out.index, out["queries"], strict=True)]
    out = out.reset_index(drop=True)
    return out.sort_values("published", ascending=False, kind="stable").reset_index(drop=True)
