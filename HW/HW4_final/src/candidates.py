"""Сборка пула кандидатов из нескольких retriever'ов.

Каждый retriever возвращает `dict[uid -> list[item_id]]` (упорядоченный top-K).
Здесь объединяем, дедуплицируем и складываем ранги/скоры в плоский DataFrame.

Выходной формат (`cand_features.parquet` основа):
    user_id | item_id | rank_<retriever> (int) | score_<retriever> (float)
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# тип "retriever" — здесь просто dict (после `model.recommend`)


def build_candidates(
    per_retriever: dict[str, dict[int, list[int]]],
    *,
    score_dicts: dict[str, dict[int, dict[int, float]]] | None = None,
) -> pd.DataFrame:
    """Собрать единый DataFrame кандидатов.

    Args:
        per_retriever: name → {uid → [item_id, ...]} (топ-K в порядке убывания)
        score_dicts:   опционально, name → {uid → {item_id → score}}
                       если None — score выводится из ранга (1 / rank).
    """
    score_dicts = score_dicts or {}

    # Соберём union (user, item) и для каждого retriever-а сохраним ранг.
    rows: dict[tuple[int, int], dict[str, float | int]] = {}

    for name, recs in per_retriever.items():
        for uid, items in recs.items():
            scores_map = score_dicts.get(name, {}).get(uid, {})
            for r, item in enumerate(items, start=1):
                key = (uid, item)
                bucket = rows.setdefault(key, {})
                bucket[f"rank_{name}"] = r
                bucket[f"score_{name}"] = float(scores_map.get(item, 1.0 / r))

    if not rows:
        return pd.DataFrame(columns=["user_id", "item_id"])

    df = pd.DataFrame(
        [
            {"user_id": uid, "item_id": item, **v}
            for (uid, item), v in rows.items()
        ]
    )
    # Numeric NaN — там, где retriever не предложил пары; tree-based нормально съест.
    score_cols = [c for c in df.columns if c.startswith("score_")]
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    df[score_cols] = df[score_cols].astype("float32")
    df[rank_cols] = df[rank_cols].astype("float32")  # с NaN — float
    df["user_id"] = df["user_id"].astype("int32")
    df["item_id"] = df["item_id"].astype("int32")

    # Доп.фичи смешивания: n_retrievers_hit (в скольких retriever'ах появилась пара) — сильный сигнал.
    df["n_retrievers_hit"] = df[rank_cols].notna().sum(axis=1).astype("int8")
    df["mean_rank"] = df[rank_cols].mean(axis=1).astype("float32")
    df["min_rank"] = df[rank_cols].min(axis=1).astype("float32")

    return df


def attach_label(
    candidates: pd.DataFrame,
    truth: dict[int, set[int]],
) -> pd.DataFrame:
    """Добавить колонку `label` (1 если item в ground-truth юзера, иначе 0)."""
    out = candidates.copy()
    out["label"] = out.apply(
        lambda r: 1 if r["item_id"] in truth.get(int(r["user_id"]), set()) else 0,
        axis=1,
    ).astype("int8")
    return out


def recall_of_candidates(
    candidates: pd.DataFrame,
    truth: dict[int, set[int]],
) -> float:
    """Какая доля релевантных товаров попала в пул кандидатов."""
    if not truth:
        return 0.0
    grouped = candidates.groupby("user_id")["item_id"].agg(set)
    total = 0.0
    for uid, pos in truth.items():
        cand = grouped.get(uid, set())
        if pos:
            total += len(pos & cand) / len(pos)
    return total / len(truth)
