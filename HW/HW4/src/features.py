"""Сборка фичей для reranker'а: джойн user/item/pair/content + content-match."""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import datamart as DM


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка готовых AGG-таблиц
# ──────────────────────────────────────────────────────────────────────────────
def load_feature_tables() -> dict[str, pd.DataFrame]:
    return {
        "user_stats": DM.load_agg("user_stats"),
        "item_stats": DM.load_agg("item_stats"),
        "item_content": DM.load_agg("item_content"),
        "user_item_pair": DM.load_agg("user_item_pair"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Build features
# ──────────────────────────────────────────────────────────────────────────────
def build_features(
    candidates: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    *,
    user_fav_tags: dict[int, set[int]] | None = None,
) -> pd.DataFrame:
    """Джойнит к candidates все user/item/pair фичи + content match.

    candidates: (user_id, item_id, ...retrieval scores/ranks...).
    """
    df = candidates.copy()
    df = df.merge(tables["user_stats"], on="user_id", how="left")
    df = df.merge(tables["item_stats"], on="item_id", how="left")

    # pair-фичи (large таблица, оставим только то, что есть в кандидатах)
    pair = tables["user_item_pair"]
    df = df.merge(pair, on=["user_id", "item_id"], how="left")
    for col in ["ui_n_interactions", "ui_n_purchases", "ui_max_rating"]:
        df[col] = df[col].fillna(0)
    df["ui_last_seen_days"] = (
        pd.Timestamp.now(tz=None) - df["ui_last_ts"]
    ).dt.total_seconds() / 86400.0
    df = df.drop(columns=["ui_last_ts", "user_last_ts", "user_first_ts", "item_last_ts", "item_first_ts"], errors="ignore")

    # content фичи (количество тегов/серий/авторов)
    item_content = tables["item_content"][
        ["item_id", "n_tags", "n_series", "n_authors", "n_top_tags", "n_top_authors"]
    ]
    df = df.merge(item_content, on="item_id", how="left")

    # content match: пересечение любимых категорий юзера с тегами товара.
    # ВАЖНО: не используем df.apply (медленно). Векторизуем через словари и
    # tight numpy-loop по уникальным item_id (мемоизация item-set).
    if user_fav_tags is not None:
        raw_cat = (
            tables["item_content"]
            .set_index("item_id")["category_tags"]
            .to_dict()
        )
        # frozenset по item — считаем один раз
        cat_set_lookup: dict[int, frozenset] = {}
        for k, v in raw_cat.items():
            if v is None:
                continue
            try:
                if len(v) == 0:
                    continue
            except TypeError:
                continue
            cat_set_lookup[int(k)] = frozenset(int(t) for t in v)

        uids_arr = df["user_id"].to_numpy()
        iids_arr = df["item_id"].to_numpy()
        n = len(df)
        out = np.zeros(n, dtype=np.float32)
        # один Python-loop по numpy-массивам — никаких pandas-overhead
        for k in tqdm(range(n), desc="content_jaccard", mininterval=1.0):
            u = user_fav_tags.get(int(uids_arr[k]))
            if not u:
                continue
            it_set = cat_set_lookup.get(int(iids_arr[k]))
            if not it_set:
                continue
            inter = len(u & it_set)
            if inter == 0:
                continue
            union = len(u) + len(it_set) - inter
            if union:
                out[k] = inter / union
        df["content_jaccard"] = out

    return df


def make_user_fav_tags(
    train_fit: pd.DataFrame,
    items: pd.DataFrame,
    top_n: int = 10,
) -> dict[int, set[int]]:
    """Top-N category_tags по покупкам каждого юзера."""
    from collections import Counter

    purchases = train_fit[train_fit["is_purchased"]][["user_id", "item_id"]]
    cats_by_item = items.set_index("item_id")["category_tags"].to_dict()
    out: dict[int, set[int]] = {}
    groups = purchases.groupby("user_id")
    for uid, sub in tqdm(groups, total=len(groups), desc="user_fav_tags", mininterval=1.0):
        cnt: Counter = Counter()
        for item in sub["item_id"]:
            cats = cats_by_item.get(int(item))
            if cats is not None:
                cnt.update(int(c) for c in cats)
        out[int(uid)] = set(c for c, _ in cnt.most_common(top_n))
    return out
