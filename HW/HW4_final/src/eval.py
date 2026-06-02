"""Единый eval-интерфейс для всех моделей.

Любая модель = объект с методом `recommend(user_ids, k, exclude) -> dict[uid, list[item_id]]`.
Здесь же протокол `Recommender` и удобный обёрток-цикл.
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np
import pandas as pd

from . import metrics as M


class Recommender(Protocol):
    """Минимальный контракт модели."""

    name: str

    def recommend(
        self,
        user_ids: list[int],
        k: int,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        ...


def evaluate(
    model: Recommender,
    truth: dict[int, set[int]],
    *,
    k: int = 20,
    exclude: dict[int, set[int]] | None = None,
    catalog_size: int | None = None,
    item_popularity: dict[int, int] | None = None,
    n_users_for_novelty: int | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Получить top-K и посчитать все метрики."""
    users = list(truth.keys())
    t0 = time.time()
    recs = model.recommend(users, k=k, exclude=exclude)
    dt_rec = time.time() - t0

    out = M.compute_all(
        recs,
        truth,
        k=k,
        catalog_size=catalog_size,
        item_popularity=item_popularity,
        n_users=n_users_for_novelty,
    )
    out["recommend_sec"] = dt_rec
    if verbose:
        head = ", ".join(f"{n}={v:.4f}" for n, v in out.items() if not n.startswith("recommend"))
        print(f"[{model.name}] {head}  ({dt_rec:.1f}s for {len(users):,} users)")
    return out


def to_submission(
    recs: dict[int, list[int]],
    user_order: list[int],
) -> pd.DataFrame:
    """Конвертирует словарь рекомендаций в submission-формат (user_id, item_id),
    20 строк на пользователя в порядке убывания приоритета."""
    rows_u: list[int] = []
    rows_i: list[int] = []
    for uid in user_order:
        items = recs.get(uid, [])
        rows_u.extend([uid] * len(items))
        rows_i.extend(items)
    return pd.DataFrame({"user_id": rows_u, "item_id": rows_i})


class ResultsLog:
    """Маленький накопитель для results.md."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, model_name: str, split: str, metrics: dict[str, float], notes: str = "") -> None:
        self.rows.append({"model": model_name, "split": split, **metrics, "notes": notes})

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_markdown(self) -> str:
        return self.to_df().to_markdown(index=False, floatfmt=".4f")
