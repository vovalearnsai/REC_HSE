"""Метрики ранжирования — все батчевые, по top-K рекомендациям.

Контракт: `recs` — `dict[user_id -> list[item_id]]` (упорядочены по убыванию score),
          `truth` — `dict[user_id -> set[item_id]]`.
Юзеры из `truth`, отсутствующие в `recs`, считаются как пустые рекомендации.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _hits_array(rec: list[int], pos: set[int], k: int) -> np.ndarray:
    """Маска размера k: 1 если item ∈ relevant, иначе 0."""
    top = rec[:k]
    hits = np.fromiter((1 if i in pos else 0 for i in top), dtype=np.int8, count=len(top))
    if len(hits) < k:
        hits = np.concatenate([hits, np.zeros(k - len(hits), dtype=np.int8)])
    return hits


def ndcg_at_k(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int = 20) -> float:
    """Binary-relevance NDCG@k. IDCG считается по min(|pos|, k)."""
    if not truth:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    total = 0.0
    for uid, pos in truth.items():
        rec = recs.get(uid, [])
        hits = _hits_array(rec, pos, k)
        dcg = float((hits * discounts).sum())
        ideal = min(len(pos), k)
        idcg = float(discounts[:ideal].sum()) if ideal > 0 else 1.0
        total += dcg / idcg
    return total / len(truth)


def recall_at_k(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int = 20) -> float:
    if not truth:
        return 0.0
    total = 0.0
    for uid, pos in truth.items():
        top = recs.get(uid, [])[:k]
        if not pos:
            continue
        total += sum(1 for i in top if i in pos) / len(pos)
    return total / len(truth)


def hitrate_at_k(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int = 20) -> float:
    if not truth:
        return 0.0
    total = 0
    for uid, pos in truth.items():
        top = recs.get(uid, [])[:k]
        if any(i in pos for i in top):
            total += 1
    return total / len(truth)


def map_at_k(recs: dict[int, list[int]], truth: dict[int, set[int]], k: int = 20) -> float:
    """Mean Average Precision@k."""
    if not truth:
        return 0.0
    total = 0.0
    for uid, pos in truth.items():
        top = recs.get(uid, [])[:k]
        if not pos:
            continue
        hits = 0
        precision_sum = 0.0
        for i, item in enumerate(top, start=1):
            if item in pos:
                hits += 1
                precision_sum += hits / i
        total += precision_sum / min(len(pos), k)
    return total / len(truth)


def coverage_at_k(
    recs: dict[int, list[int]],
    catalog_size: int,
    k: int = 20,
) -> float:
    """Доля каталога, появляющаяся хотя бы в одной топ-K рекомендации."""
    seen: set[int] = set()
    for rec in recs.values():
        seen.update(rec[:k])
    return len(seen) / max(catalog_size, 1)


def novelty_at_k(
    recs: dict[int, list[int]],
    item_popularity: dict[int, int],
    n_users: int,
    k: int = 20,
) -> float:
    """Mean self-information: -log2(p(item)). Чем выше — тем менее популярные товары."""
    if not recs:
        return 0.0
    n = max(n_users, 1)
    total = 0.0
    cnt = 0
    for rec in recs.values():
        for item in rec[:k]:
            p = (item_popularity.get(item, 0) + 1) / (n + 1)
            total += -np.log2(p)
            cnt += 1
    return total / max(cnt, 1)


def compute_all(
    recs: dict[int, list[int]],
    truth: dict[int, set[int]],
    *,
    k: int = 20,
    catalog_size: int | None = None,
    item_popularity: dict[int, int] | None = None,
    n_users: int | None = None,
) -> dict[str, float]:
    """Удобная сводка для отчёта/results.md."""
    out = {
        f"ndcg@{k}": ndcg_at_k(recs, truth, k),
        f"recall@{k}": recall_at_k(recs, truth, k),
        f"map@{k}": map_at_k(recs, truth, k),
        f"hitrate@{k}": hitrate_at_k(recs, truth, k),
    }
    if catalog_size is not None:
        out[f"coverage@{k}"] = coverage_at_k(recs, catalog_size, k)
    if item_popularity is not None and n_users is not None:
        out[f"novelty@{k}"] = novelty_at_k(recs, item_popularity, n_users, k)
    return out
