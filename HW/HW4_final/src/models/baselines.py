"""Baseline-модели: GlobalTopPop, RecentTopPop, UserHistoryTopPop."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._persist import Persistable


class GlobalTopPop(Persistable):
    """Топ-K самых покупаемых товаров. Один и тот же ответ всем."""
    name = "GlobalTopPop"

    def __init__(self) -> None:
        self.popular: list[int] = []
        self.popularity: dict[int, int] = {}

    def fit(self, df: pd.DataFrame) -> "GlobalTopPop":
        cnt = (
            df[df["is_purchased"]]
            .groupby("item_id")
            .size()
            .sort_values(ascending=False)
        )
        self.popular = cnt.index.astype(np.int64).tolist()
        self.popularity = cnt.to_dict()
        return self

    def recommend(
        self,
        user_ids: list[int],
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        exclude = exclude or {}
        pool_size = k + 200  # запас, чтобы хватило после exclude
        pool = self.popular[:pool_size]
        out: dict[int, list[int]] = {}
        for uid in user_ids:
            seen = exclude.get(uid, set())
            rec = [i for i in pool if i not in seen][:k]
            if len(rec) < k:  # добор из хвоста
                for i in self.popular[pool_size:]:
                    if i not in seen and i not in rec:
                        rec.append(i)
                        if len(rec) == k:
                            break
            out[uid] = rec
        return out


class RecentTopPop(Persistable):
    """Топ-K с экспоненциальным decay по recency.
    score(item) = Σ exp(-Δt_days / τ) по покупкам.
    """
    name = "RecentTopPop"

    def __init__(self, tau_days: float = 30.0) -> None:
        self.tau_days = tau_days
        self.popular: list[int] = []
        self.popularity: dict[int, float] = {}

    def fit(self, df: pd.DataFrame, ref_ts: pd.Timestamp | None = None) -> "RecentTopPop":
        purchases = df[df["is_purchased"]][["item_id", "timestamp"]]
        if ref_ts is None:
            ref_ts = purchases["timestamp"].max()
        delta_days = (ref_ts - purchases["timestamp"]).dt.total_seconds() / 86400.0
        weight = np.exp(-delta_days / self.tau_days)
        score = pd.Series(weight.values, index=purchases["item_id"].values).groupby(level=0).sum()
        score = score.sort_values(ascending=False)
        self.popular = score.index.astype(np.int64).tolist()
        self.popularity = score.to_dict()
        return self

    def recommend(
        self,
        user_ids: list[int],
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        exclude = exclude or {}
        out: dict[int, list[int]] = {}
        for uid in user_ids:
            seen = exclude.get(uid, set())
            rec: list[int] = []
            for i in self.popular:
                if i not in seen:
                    rec.append(i)
                    if len(rec) == k:
                        break
            out[uid] = rec
        return out


class UserCategoryTopPop(Persistable):
    """Top-K популярных товаров в любимых категориях пользователя.

    «Любимая категория» = top-N category_tags, по которым у юзера были покупки.
    Score(item) = популярность(item) * число пересечений категорий с любимыми.
    Для пользователей без истории fallback к GlobalTopPop.
    """
    name = "UserCategoryTopPop"

    def __init__(self, n_fav_cats: int = 5) -> None:
        self.n_fav_cats = n_fav_cats
        self.global_pop_list: list[int] = []
        self.item_popularity: dict[int, int] = {}
        self.item_categories: dict[int, list[int]] = {}
        self.user_fav_categories: dict[int, list[int]] = {}
        # обратный индекс категория → топ-популярные item_ids в ней
        self.cat_top_items: dict[int, list[int]] = {}

    def fit(self, df: pd.DataFrame, items: pd.DataFrame, cat_top: int = 200) -> "UserCategoryTopPop":
        purchases = df[df["is_purchased"]]
        pop = purchases.groupby("item_id").size().sort_values(ascending=False)
        self.item_popularity = pop.to_dict()
        self.global_pop_list = pop.index.astype(np.int64).tolist()

        # item → categories (только наблюдаемые в train, чтобы не плодить мусор)
        items_index = items.set_index("item_id")
        self.item_categories = {
            int(i): list(items_index.loc[i, "category_tags"])
            for i in pop.index
            if i in items_index.index
        }

        # для каждой категории — самые покупаемые item-ы в ней
        from collections import defaultdict
        cat_buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for item, cats in self.item_categories.items():
            p = self.item_popularity.get(item, 0)
            for c in cats:
                cat_buckets[int(c)].append((p, item))
        self.cat_top_items = {
            c: [it for _, it in sorted(lst, reverse=True)[:cat_top]]
            for c, lst in cat_buckets.items()
        }

        # для каждого юзера — top-N категорий по числу покупок
        merged = purchases.merge(items[["item_id", "category_tags"]], on="item_id", how="left")
        from collections import Counter
        user_cats: dict[int, list[int]] = {}
        for uid, sub in merged.groupby("user_id"):
            cnt: Counter = Counter()
            for cats in sub["category_tags"]:
                if cats is not None:
                    cnt.update(cats)
            user_cats[int(uid)] = [c for c, _ in cnt.most_common(self.n_fav_cats)]
        self.user_fav_categories = user_cats
        return self

    def recommend(
        self,
        user_ids: list[int],
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        exclude = exclude or {}
        out: dict[int, list[int]] = {}
        for uid in user_ids:
            seen = exclude.get(uid, set())
            fav_cats = self.user_fav_categories.get(uid, [])
            # объединяем top-items из любимых категорий, сохраняя «голос» каждой категории
            scores: dict[int, float] = {}
            for c in fav_cats:
                for rank, item in enumerate(self.cat_top_items.get(c, [])[:200]):
                    if item in seen:
                        continue
                    scores[item] = scores.get(item, 0.0) + 1.0 / (rank + 1)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rec = [i for i, _ in ranked[:k]]
            # доберём глобальной популярностью, если мало (или совсем нет любимых категорий)
            if len(rec) < k:
                for i in self.global_pop_list:
                    if i not in seen and i not in rec:
                        rec.append(i)
                        if len(rec) == k:
                            break
            out[uid] = rec
        return out
