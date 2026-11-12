"""Analytics: activity aggregation, negative-spike detection, and issue clustering."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


SENTIMENT_LABELS = {
    "positive": "Положительная",
    "negative": "Отрицательная",
    "neutral": "Нейтральная",
}


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    result["created_at"] = pd.to_datetime(result["created_at"], utc=True, errors="coerce")
    result = result.dropna(subset=["created_at"])
    result["sentiment_ru"] = result["sentiment"].map(SENTIMENT_LABELS).fillna("Не определена")
    result["message_type_ru"] = result["kind"].map({"issue": "Issue", "comment": "Комментарий"}).fillna(result["kind"])
    result["text_preview"] = result["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 240)
    return result


def activity_by_time(frame: pd.DataFrame, frequency: str = "1h") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["time", "total", "negative", "positive", "neutral"])
    indexed = frame.set_index("created_at")
    grouped = indexed.groupby(pd.Grouper(freq=frequency)).agg(
        total=("message_id", "count"),
        negative=("sentiment", lambda values: int((values == "negative").sum())),
        positive=("sentiment", lambda values: int((values == "positive").sum())),
        neutral=("sentiment", lambda values: int((values == "neutral").sum())),
    )
    return grouped.reset_index(names="time").fillna(0)


def detect_negative_spikes(
    frame: pd.DataFrame,
    frequency: str = "15min",
    baseline_periods: int = 24,
    min_negative_count: int = 3,
    z_threshold: float = 2.5,
    min_negative_share: float = 0.60,
) -> pd.DataFrame:
    """Flag time buckets with unusually many negative messages and a high negative share.

    Baseline statistics are calculated only from preceding buckets, avoiding leakage
    from the bucket currently being evaluated.
    """
    columns = ["time", "total", "negative", "negative_share", "baseline_mean", "threshold", "is_spike"]
    if frame.empty or not (frame["sentiment"] == "negative").any():
        return pd.DataFrame(columns=columns)

    source = frame.set_index("created_at").sort_index()
    counts = source.resample(frequency).agg(
        total=("message_id", "count"),
        negative=("sentiment", lambda values: int((values == "negative").sum())),
    )
    counts["negative_share"] = (counts["negative"] / counts["total"].where(counts["total"] > 0, 1)).where(counts["total"] > 0, 0.0)
    prior = counts["negative"].shift(1)
    rolling = prior.rolling(window=baseline_periods, min_periods=min(4, baseline_periods))
    counts["baseline_mean"] = rolling.mean().fillna(0.0)
    baseline_std = rolling.std(ddof=0).fillna(0.0)
    counts["threshold"] = (counts["baseline_mean"] + z_threshold * baseline_std).clip(lower=min_negative_count)
    counts["is_spike"] = (
        (counts["negative"] >= counts["threshold"])
        & (counts["negative"] >= min_negative_count)
        & (counts["negative_share"] >= min_negative_share)
    )
    return counts.reset_index(names="time")[columns].sort_values("time", ascending=False)


def cluster_messages(frame: pd.DataFrame, max_clusters: int = 5) -> pd.DataFrame:
    """Group messages using TF-IDF + KMeans; return cluster summaries and examples."""
    columns = ["cluster", "keywords", "messages", "negative_share", "examples"]
    if len(frame) < 4:
        return pd.DataFrame(columns=columns)

    try:
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    except ImportError:  # pragma: no cover
        return pd.DataFrame(columns=columns)

    frame = frame.reset_index(drop=True)
    texts = frame["text"].fillna("").astype(str).tolist()
    stop_words = set(ENGLISH_STOP_WORDS) | {
        "issue", "issues", "please", "thanks", "thank", "user", "users", "same", "using", "use",
        "also", "still", "need", "like", "work", "works", "working", "going", "new", "problem",
        "что", "это", "как", "для", "при", "или", "если", "когда", "уже", "мне", "они",
        "быть", "также", "после", "очень", "просто", "весь", "этот", "эта", "эти", "его", "ее",
        "вам", "ваш", "ваша", "ваши", "чтобы", "может",
    }
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=sorted(stop_words),
            ngram_range=(1, 2),
            min_df=2,
            max_features=1500,
            token_pattern=r"(?u)\b[^\W\d_][\w-]+\b",
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=columns)

    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=columns)

    unique_rows = matrix.shape[0]
    cluster_count = min(max_clusters, max(2, int(math.sqrt(unique_rows / 2))))
    if unique_rows < cluster_count:
        cluster_count = unique_rows
    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)
    terms = vectorizer.get_feature_names_out()

    rows: list[dict[str, Any]] = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        subset = frame.loc[mask]
        center = model.cluster_centers_[cluster_id]
        top_indexes = center.argsort()[::-1][:5]
        keywords = ", ".join(terms[index] for index in top_indexes if center[index] > 0)
        examples = subset["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).head(2).tolist()
        negative_share = float((subset["sentiment"] == "negative").mean()) if len(subset) else 0.0
        rows.append(
            {
                "cluster": int(cluster_id) + 1,
                "keywords": keywords or "Ключевые слова не выделены",
                "messages": int(mask.sum()),
                "negative_share": negative_share,
                "examples": " | ".join(example[:180] for example in examples),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("messages", ascending=False)
