"""Datamart: RAW и AGG слои.

Все агрегаты считаются строго по данным `< cutoff` чтобы избежать утечки времени.
Файлы складируются в `HW/datamart/{raw,agg,feat}/`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATAMART_DIR

RAW_DIR = DATAMART_DIR / "raw"
AGG_DIR = DATAMART_DIR / "agg"
FEAT_DIR = DATAMART_DIR / "feat"
for _d in (RAW_DIR, AGG_DIR, FEAT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# RAW
# ──────────────────────────────────────────────────────────────────────────────
def build_raw(train: pd.DataFrame, items: pd.DataFrame) -> None:
    """Складываем нормализованные исходники в parquet (без эксплода — экономия места).

    Импрешен-эксплод дорогой (×20 рядов) — собираем по требованию, не складируем.
    """
    out_train = train.copy()
    out_train["impression_id"] = np.arange(len(out_train), dtype=np.int64)
    # impressions храним как python list[int32] — pyarrow ок
    out_train.to_parquet(RAW_DIR / "interactions.parquet", index=False)
    items.to_parquet(RAW_DIR / "items.parquet", index=False)
    print(f"[raw] interactions: {len(out_train):,} rows → {RAW_DIR/'interactions.parquet'}")
    print(f"[raw] items       : {len(items):,} rows → {RAW_DIR/'items.parquet'}")


# ──────────────────────────────────────────────────────────────────────────────
# AGG: user / item / pair / content
# ──────────────────────────────────────────────────────────────────────────────
def build_user_stats(train_fit: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    """Поведенческие фичи пользователя — на момент `cutoff_ts`."""
    g = train_fit.groupby("user_id", sort=False)
    df = pd.DataFrame(
        {
            "user_n_interactions": g.size(),
            "user_n_purchases": g["is_purchased"].sum().astype("int32"),
            "user_n_unique_items": g["item_id"].nunique().astype("int32"),
            "user_mean_rating": g["rating"].mean().astype("float32"),
            "user_n_rated": (g["rating"].apply(lambda s: int((s > 0).sum()))).astype("int32"),
            "user_last_ts": g["timestamp"].max(),
            "user_first_ts": g["timestamp"].min(),
        }
    ).reset_index()
    df["user_purchase_rate"] = (df["user_n_purchases"] / df["user_n_interactions"]).astype("float32")
    df["user_recency_days"] = ((cutoff_ts - df["user_last_ts"]).dt.total_seconds() / 86400).astype("float32")
    df["user_tenure_days"] = ((df["user_last_ts"] - df["user_first_ts"]).dt.total_seconds() / 86400).astype("float32")

    # активность по hour / dow — самый частый
    train_fit = train_fit.assign(
        _hour=train_fit["timestamp"].dt.hour.astype("int8"),
        _dow=train_fit["timestamp"].dt.dayofweek.astype("int8"),
    )
    fav_hour = train_fit.groupby("user_id")["_hour"].agg(lambda s: int(s.mode().iat[0]))
    fav_dow = train_fit.groupby("user_id")["_dow"].agg(lambda s: int(s.mode().iat[0]))
    df = df.merge(fav_hour.rename("user_fav_hour").reset_index(), on="user_id", how="left")
    df = df.merge(fav_dow.rename("user_fav_dow").reset_index(), on="user_id", how="left")

    df["user_n_interactions"] = df["user_n_interactions"].astype("int32")
    return df


def build_item_stats(train_fit: pd.DataFrame, cutoff_ts: pd.Timestamp) -> pd.DataFrame:
    g = train_fit.groupby("item_id", sort=False)
    df = pd.DataFrame(
        {
            "item_n_interactions": g.size().astype("int32"),
            "item_n_purchases": g["is_purchased"].sum().astype("int32"),
            "item_n_unique_users": g["user_id"].nunique().astype("int32"),
            "item_mean_rating": g["rating"].mean().astype("float32"),
            "item_last_ts": g["timestamp"].max(),
            "item_first_ts": g["timestamp"].min(),
        }
    ).reset_index()
    df["item_purchase_rate"] = (df["item_n_purchases"] / df["item_n_interactions"]).astype("float32")
    df["item_recency_days"] = ((cutoff_ts - df["item_last_ts"]).dt.total_seconds() / 86400).astype("float32")
    df["item_age_days"] = ((cutoff_ts - df["item_first_ts"]).dt.total_seconds() / 86400).astype("float32")
    df["item_popularity_rank"] = df["item_n_interactions"].rank(method="dense", ascending=False).astype("int32")
    df["item_purchase_rank"] = df["item_n_purchases"].rank(method="dense", ascending=False).astype("int32")
    return df


def build_user_item_pair(train_fit: pd.DataFrame) -> pd.DataFrame:
    """Пары (user, item): n_interactions, n_purchases, last_ts."""
    g = train_fit.groupby(["user_id", "item_id"], sort=False)
    df = pd.DataFrame(
        {
            "ui_n_interactions": g.size().astype("int32"),
            "ui_n_purchases": g["is_purchased"].sum().astype("int32"),
            "ui_max_rating": g["rating"].max().astype("int8"),
            "ui_last_ts": g["timestamp"].max(),
        }
    ).reset_index()
    return df


def build_item_content(items: pd.DataFrame, top_tags_k: int = 200, top_authors_k: int = 100) -> pd.DataFrame:
    """Контент-фичи для каталога — приземлённый формат для reranker'а.

    Сохраняем сырые списки + флаги принадлежности к top-K тегам/авторам.
    """
    from collections import Counter
    from itertools import chain

    df = items[["item_id", "category_tags", "series_id", "author_ids"]].copy()

    def _safe_len(x) -> int:
        return 0 if x is None else len(x)

    def _safe_iter(x):
        return [] if x is None else list(x)

    df["n_tags"] = df["category_tags"].apply(_safe_len).astype("int32")
    df["n_series"] = df["series_id"].apply(_safe_len).astype("int32")
    df["n_authors"] = df["author_ids"].apply(_safe_len).astype("int32")

    # топ-K самых частых тегов / авторов
    top_tags = [t for t, _ in Counter(chain.from_iterable(_safe_iter(x) for x in df["category_tags"])).most_common(top_tags_k)]
    top_authors = [a for a, _ in Counter(chain.from_iterable(_safe_iter(x) for x in df["author_ids"])).most_common(top_authors_k)]

    top_tags_set = set(top_tags)
    top_authors_set = set(top_authors)
    df["n_top_tags"] = df["category_tags"].apply(
        lambda x: sum(1 for t in _safe_iter(x) if t in top_tags_set)
    ).astype("int16")
    df["n_top_authors"] = df["author_ids"].apply(
        lambda x: sum(1 for a in _safe_iter(x) if a in top_authors_set)
    ).astype("int16")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: запись/чтение
# ──────────────────────────────────────────────────────────────────────────────
def save_agg(df: pd.DataFrame, name: str) -> Path:
    path = AGG_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"[agg] {name:>18}: {len(df):>10,} rows → {path}")
    return path


def load_agg(name: str) -> pd.DataFrame:
    return pd.read_parquet(AGG_DIR / f"{name}.parquet")


def build_all_agg(train_fit: pd.DataFrame, items: pd.DataFrame, cutoff_ts: pd.Timestamp) -> None:
    """Построить все AGG-таблицы. Cutoff = граница train_fit/val."""
    save_agg(build_user_stats(train_fit, cutoff_ts), "user_stats")
    save_agg(build_item_stats(train_fit, cutoff_ts), "item_stats")
    save_agg(build_user_item_pair(train_fit), "user_item_pair")
    save_agg(build_item_content(items), "item_content")
