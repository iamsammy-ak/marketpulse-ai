"""Theme discovery: group headlines into themes and name each theme by its distinguishing terms.

Pipeline: TF-IDF on headline text (unigrams + bigrams; competitor names, market-data boilerplate
and terms in more than 20% of headlines removed, so themes cut across companies) → latent semantic
analysis (truncated SVD to 8 dimensions, unit-length rows) → k-means for every k in 4-12 → keep the
k with the best cosine silhouette score. A low-dimensional space gives k-means compact clusters on
short headlines and a silhouette curve with a clear peak; in the full TF-IDF space the score
simply keeps rising with k.

A theme's label is the three terms whose average TF-IDF weight inside the theme exceeds their
corpus-wide average by the most, preferring a bigram ("export controls") over its single words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_DOCS = 12
K_RANGE = (4, 12)
LSA_COMPONENTS = 8
SEED = 42

NEWS_STOPWORDS = {
    # reporting boilerplate
    "says", "said", "say", "new", "news", "report", "reports", "reported", "update", "updates", "today", "latest",
    "know", "need", "things", "here", "heres", "just", "now", "could", "would", "will", "may", "might", "get", "gets",
    "got", "make", "makes", "big", "via", "amid", "after", "ahead", "despite", "over", "vs", "us", "one", "two",
    "first", "top", "best", "look", "looks", "set", "sets", "see", "sees", "time", "people", "way", "ways", "back",
    "like", "still", "much", "many", "really", "going", "want", "watch", "live", "video", "photos", "exclusive",
    "opinion", "analysis", "review", "breaking", "week", "weeks", "year", "years", "month", "months", "day", "days",
    # company suffixes
    "inc", "incorporated", "corp", "corporation", "co", "ltd", "plc", "llc", "company", "companies", "firm", "firms",
    # market-data page boilerplate (quote pages, tickers and price templates)
    "stock", "stocks", "shares", "share", "nasdaq", "nyse", "price", "prices", "quote", "quotes", "history",
    "chart", "charts", "ticker", "premarket", "trading", "traded", "trade", "buy", "sell", "hold",
    # calendar words
    "january", "february", "march", "april", "june", "july", "august", "september", "october", "november",
    "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "2025", "2026", "2027",
}
_TOKEN = re.compile(r"[a-z][a-z0-9]+")


@dataclass(frozen=True)
class ThemeModel:
    labels: np.ndarray  # theme per headline; -1 = no usable terms
    terms: dict[int, list[str]]  # distinguishing terms per theme, best first
    silhouette: pd.DataFrame  # columns: k, score
    k: int
    auto: bool


def stopwords_for(names: list[str]) -> set[str]:
    """Generic English and news stop words plus every token of the watchlist names and aliases."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    extra = {tok for name in names for tok in _TOKEN.findall(name.lower())}
    return set(ENGLISH_STOP_WORDS) | NEWS_STOPWORDS | extra


def _distinct_terms(scores: np.ndarray, vocab: np.ndarray, n: int = 3) -> list[str]:
    """Top terms by score. A unigram is replaced by a strong bigram containing it, and a term is
    skipped when it shares a word with one already chosen."""
    order = [i for i in np.argsort(-scores)[:60] if scores[i] > 0]
    bigrams = [(str(vocab[i]), scores[i]) for i in order if " " in str(vocab[i])]
    chosen: list[str] = []
    for idx in order:
        if len(chosen) >= n:
            break
        term = str(vocab[idx])
        if " " not in term:
            better = [b for b, sc in bigrams if term in b.split() and sc >= 0.5 * scores[idx]]
            term = better[0] if better else term
        words = set(term.split())
        if any(words & set(c.split()) for c in chosen):
            continue
        chosen.append(term)
    return chosen


def fit_themes(texts: list[str], stop_words: set[str], n_themes: int = 0, k_range: tuple[int, int] = K_RANGE,
               seed: int = SEED) -> ThemeModel:
    """Cluster texts into themes. `n_themes` = 0 picks k by silhouette within `k_range`."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import Normalizer

    n = len(texts)
    empty = pd.DataFrame(columns=["k", "score"])
    if n < MIN_DOCS:
        return ThemeModel(np.zeros(n, dtype=int), {0: []}, empty, 1, False)
    vec = TfidfVectorizer(stop_words=sorted(stop_words), ngram_range=(1, 2), min_df=3, max_df=0.2,
                          sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b")
    try:
        x = vec.fit_transform(texts)
    except ValueError:  # every term was filtered out
        return ThemeModel(np.zeros(n, dtype=int), {0: []}, empty, 1, False)
    usable = np.asarray(x.getnnz(axis=1) > 0)
    idx = np.flatnonzero(usable)
    if len(idx) < MIN_DOCS or x.shape[1] < 3:
        return ThemeModel(np.where(usable, 0, -1), {0: []}, empty, 1, False)

    xu = x[idx]
    components = max(2, min(LSA_COMPONENTS, xu.shape[1] - 1, len(idx) - 1))
    z = TruncatedSVD(components, random_state=seed).fit_transform(xu)
    z = Normalizer(copy=False).fit_transform(z)

    k_min, k_max = k_range
    k_max = max(2, min(k_max, len(idx) // 5))
    k_min = min(k_min, k_max)
    scores, fits = [], {}
    for k in range(max(2, k_min), k_max + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(z)
        if len(set(km.labels_)) < 2:
            continue
        score = silhouette_score(z, km.labels_, metric="cosine", sample_size=min(3000, len(idx)), random_state=seed)
        scores.append({"k": k, "score": float(score)})
        fits[k] = km.labels_
    sil = pd.DataFrame(scores, columns=["k", "score"])
    auto = not n_themes
    if auto:
        k = int(sil.loc[sil["score"].idxmax(), "k"]) if not sil.empty else 1
    else:
        k = int(max(2, min(n_themes, len(idx) // 2)))
    raw = fits.get(k)
    if raw is None:
        raw = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(z).labels_ if k > 1 else np.zeros(len(idx), int)

    # renumber themes by size (0 = largest) so ids are stable and meaningful
    sizes = pd.Series(raw).value_counts()
    order = {old: new for new, old in enumerate(sorted(sizes.index, key=lambda c: (-sizes[c], c)))}
    labels_u = np.array([order[c] for c in raw])
    labels = np.full(n, -1, dtype=int)
    labels[idx] = labels_u

    vocab = vec.get_feature_names_out()
    overall = np.asarray(xu.mean(axis=0)).ravel()
    terms = {}
    for theme in range(len(order)):
        inside = np.asarray(xu[labels_u == theme].mean(axis=0)).ravel()
        terms[theme] = _distinct_terms(inside - overall, vocab)
    return ThemeModel(labels, terms, sil, len(order), auto)


def theme_label(terms: list[str]) -> str:
    return " · ".join(terms) if terms else "Mixed headlines"
