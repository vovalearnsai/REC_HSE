"""Negative sampling из `impressions`.

Идея: для каждой строки train у нас есть слейт из 20 товаров. Item_id всегда
лежит в impressions, остальные 19 — это **показанные, но не выбранные** товары.
Это качественные hard-negatives — гораздо лучше random sampling.

Здесь два формата:
- `explode_impressions` — длинная таблица для построения impression-фичей
  и LGBM-ranker'а: одна строка = один (user, item, impression_id, label).
- `sample_negatives_for_user` — быстрый сэмплер для DSSM training loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def explode_impressions(
    df: pd.DataFrame,
    *,
    with_purchase_label: bool = True,
) -> pd.DataFrame:
    """Развернуть слейт impressions в длинную таблицу.

    На входе: train с колонками (user_id, item_id, is_purchased, rating, timestamp, impressions).
    На выходе: (impression_id, user_id, item_id, was_clicked, was_purchased, rating, timestamp).

    `impression_id` уникально идентифицирует строку исходного train (= показ слейта).
    `was_clicked` = 1 если item_id равен выбранному в этой строке (т.е. строка клика).
    """
    if "impression_id" not in df.columns:
        df = df.reset_index(drop=True).copy()
        df["impression_id"] = np.arange(len(df), dtype=np.int64)

    long = df[["impression_id", "user_id", "timestamp", "item_id", "is_purchased", "rating", "impressions"]].rename(
        columns={"item_id": "clicked_item", "is_purchased": "click_was_purchased", "rating": "click_rating"}
    )
    long = long.explode("impressions").rename(columns={"impressions": "item_id"})
    long["item_id"] = long["item_id"].astype("int32")
    long["was_clicked"] = (long["item_id"] == long["clicked_item"]).astype("int8")
    if with_purchase_label:
        long["was_purchased"] = (long["was_clicked"].astype(bool) & long["click_was_purchased"].astype(bool)).astype("int8")
        long["rating"] = np.where(long["was_clicked"].astype(bool), long["click_rating"], 0).astype("int8")
    long = long.drop(columns=["clicked_item", "click_was_purchased", "click_rating"])
    return long


def build_impression_negatives(
    df: pd.DataFrame,
    *,
    keep_positives: bool = True,
) -> pd.DataFrame:
    """Длинная таблица для LGBM-ranker'а.

    Каждая «группа» = один impression (20 строк). Из них 0 или 1 положительная
    (если в этой строке была покупка). Остальные — жёсткие негативы.
    """
    long = explode_impressions(df, with_purchase_label=True)
    if not keep_positives:
        long = long[long["was_purchased"] == 0]
    return long


def sample_negatives_for_user(
    user_impressions: np.ndarray,
    user_positives: set[int],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Из всех impressions данного юзера сэмплируем n негативов
    (товары, которые были показаны, но не куплены)."""
    cand = user_impressions[~np.isin(user_impressions, list(user_positives))]
    if len(cand) == 0:
        return np.array([], dtype=np.int64)
    if len(cand) <= n:
        return cand
    return rng.choice(cand, size=n, replace=False)
