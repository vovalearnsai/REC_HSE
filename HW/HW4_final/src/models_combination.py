import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random
from HW.HW4_final.src import metrics as Mtx


def build_blend_candidates(recs_by_model: dict[str, dict[int, list[int]]]) -> pd.DataFrame:
    """
    recs_by_model:
        {
            'pop': {user_id: [item1, item2, ...]},
            'als': {user_id: [item1, item2, ...]},
            ...
        }

    Возвращает long-table:
        user_id, item_id, rank_{model}, from_{model}
    """
    rows = []

    for model_name, recs in recs_by_model.items():
        for uid, items in recs.items():
            for rank, iid in enumerate(items, start=1):
                rows.append((int(uid), int(iid), int(rank), model_name))

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "source"])

    # Если один item пришёл от нескольких моделей — собираем в одну строку
    rank_pivot = (
        df.pivot_table(
            index=["user_id", "item_id"],
            columns="source",
            values="rank",
            aggfunc="min",
        )
        .reset_index()
    )

    # flatten columns
    rank_pivot.columns.name = None

    # rank columns
    source_names = list(recs_by_model.keys())
    for src in source_names:
        if src in rank_pivot.columns:
            rank_pivot = rank_pivot.rename(columns={src: f"rank_{src}"})
        else:
            rank_pivot[f"rank_{src}"] = np.nan

    # binary source indicators
    for src in source_names:
        rcol = f"rank_{src}"
        rank_pivot[f"from_{src}"] = rank_pivot[rcol].notna().astype("int8")

    return rank_pivot


def add_blend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rank_cols = [c for c in df.columns if c.startswith("rank_")]

    for c in rank_cols:
        src = c.replace("rank_", "")
        # если модели item не рекомендовала, rank = NaN
        # inverse rank для отсутствующих делаем 0
        df[f"inv_{c}"] = 1.0 / df[c]
        df[f"inv_{c}"] = df[f"inv_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df

def add_blend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rank_cols = [c for c in df.columns if c.startswith("rank_")]

    for c in rank_cols:
        src = c.replace("rank_", "")
        # если модели item не рекомендовала, rank = NaN
        # inverse rank для отсутствующих делаем 0
        df[f"inv_{c}"] = 1.0 / df[c]
        df[f"inv_{c}"] = df[f"inv_{c}"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df

def add_userwise_normalized_scores(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in score_cols:
        x = df[col].fillna(0.0).astype("float32")

        g = df.groupby("user_id")[col]
        mn = g.transform("min")
        mx = g.transform("max")
        denom = (mx - mn).replace(0, np.nan)

        df[f"{col}_u_norm"] = ((df[col] - mn) / denom).fillna(0.0).astype("float32")

    return df

def recommend_by_weights(
    cand: pd.DataFrame,
    weights: dict[str, float],
    *,
    user_ids: list[int] | None = None,
    k: int = 20,
    exclude: dict[int, set[int]] | None = None,
) -> dict[int, list[int]]:
    """
    cand содержит user_id, item_id и feature columns.
    weights: {'inv_rank_als': 1.0, 'score_als_u_norm': 1.0, ...}
    """
    exclude = exclude or {}

    if user_ids is not None:
        user_set = set(user_ids)
        df = cand[cand["user_id"].isin(user_set)].copy()
    else:
        df = cand.copy()

    df["blend_score"] = 0.0

    for col, w in weights.items():
        if col not in df.columns:
            continue
        df["blend_score"] += float(w) * df[col].fillna(0.0).astype("float32")

    df = df.sort_values(
        ["user_id", "blend_score"],
        ascending=[True, False],
        kind="mergesort",
    )

    out = {}
    for uid, sub in df.groupby("user_id", sort=False):
        uid = int(uid)
        seen = exclude.get(uid, set())

        rec = []
        used = set()

        for iid in sub["item_id"].values:
            iid = int(iid)
            if iid in seen:
                continue
            if iid in used:
                continue

            rec.append(iid)
            used.add(iid)

            if len(rec) == k:
                break

        out[uid] = rec

    if user_ids is not None:
        for uid in user_ids:
            out.setdefault(int(uid), [])

    return out

###########

def sample_weights(rng: np.random.Generator) -> dict[str, float]:
    """
    Рандомно сэмплируем веса.
    ALS/EASE даём больше веса, popularity/category — меньше.
    """
    return {
        "inv_rank_pop": float(rng.uniform(0.0, 0.5)),
        "inv_rank_recent": float(rng.uniform(0.0, 0.8)),
        "inv_rank_cat": float(rng.uniform(0.0, 1.0)),

        "inv_rank_als": float(rng.uniform(0.0, 2.0)),
        "inv_rank_ease": float(rng.uniform(0.0, 2.0)),

        "score_als_u_norm": float(rng.uniform(0.0, 3.0)),
        "score_ease_u_norm": float(rng.uniform(0.0, 3.0)),

        "from_pop": float(rng.uniform(0.0, 0.2)),
        "from_recent": float(rng.uniform(0.0, 0.2)),
        "from_cat": float(rng.uniform(0.0, 0.2)),
        "from_als": float(rng.uniform(0.0, 0.3)),
        "from_ease": float(rng.uniform(0.0, 0.3)),
    }


def evaluate_weights(
    cand: pd.DataFrame,
    weights: dict[str, float],
    *,
    val_users: list[int],
    truth_val: dict[int, set[int]],
    history_seen: dict[int, set[int]],
    k: int = 20,
) -> float:
    recs = recommend_by_weights(
        cand,
        weights,
        user_ids=val_users,
        k=k,
        exclude=history_seen,
    )
    return Mtx.ndcg_at_k(recs, truth_val, k=k)



def recommend_by_rrf(
    cand: pd.DataFrame,
    *,
    sources: list[str],
    c: float = 60.0,
    user_ids: list[int] | None = None,
    k: int = 20,
    exclude: dict[int, set[int]] | None = None,
    source_weights: dict[str, float] | None = None,
) -> dict[int, list[int]]:
    exclude = exclude or {}
    source_weights = source_weights or {}

    if user_ids is not None:
        user_set = set(user_ids)
        df = cand[cand["user_id"].isin(user_set)].copy()
    else:
        df = cand.copy()

    df["rrf_score"] = 0.0

    for src in sources:
        rcol = f"rank_{src}"
        if rcol not in df.columns:
            continue

        w = float(source_weights.get(src, 1.0))
        rank = df[rcol].astype("float32")

        contrib = 1.0 / (c + rank)
        contrib = contrib.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        df["rrf_score"] += w * contrib

    df = df.sort_values(
        ["user_id", "rrf_score"],
        ascending=[True, False],
        kind="mergesort",
    )

    out = {}
    for uid, sub in df.groupby("user_id", sort=False):
        uid = int(uid)
        seen = exclude.get(uid, set())

        rec = []
        used = set()

        for iid in sub["item_id"].values:
            iid = int(iid)
            if iid in seen:
                continue
            if iid in used:
                continue

            rec.append(iid)
            used.add(iid)

            if len(rec) == k:
                break

        out[uid] = rec

    if user_ids is not None:
        for uid in user_ids:
            out.setdefault(int(uid), [])

    return out


def sample_rrf_weights(rng):
    return {
        "pop": float(rng.uniform(0.0, 1.0)),
        "recent": float(rng.uniform(0.0, 1.0)),
        "cat": float(rng.uniform(0.0, 1.5)),
        "als": float(rng.uniform(0.0, 3.0)),
        "ease": float(rng.uniform(0.0, 3.0)),
    }



class BlendWrapper:
    def __init__(
        self,
        cand: pd.DataFrame,
        mode: str = "weights",
        weights: dict[str, float] | None = None,
        rrf_sources: list[str] | None = None,
        rrf_c: float = 60.0,
        rrf_source_weights: dict[str, float] | None = None,
        name: str = "BlendWrapper",
    ):
        self.cand = cand
        self.mode = mode
        self.weights = weights or {}
        self.rrf_sources = rrf_sources or []
        self.rrf_c = rrf_c
        self.rrf_source_weights = rrf_source_weights or {}
        self.name = name

    def recommend(self, user_ids, k=20, exclude=None):
        if self.mode == "rrf":
            return recommend_by_rrf(
                self.cand,
                sources=self.rrf_sources,
                c=self.rrf_c,
                source_weights=self.rrf_source_weights,
                user_ids=user_ids,
                k=k,
                exclude=exclude,
            )

        return recommend_by_weights(
            self.cand,
            self.weights,
            user_ids=user_ids,
            k=k,
            exclude=exclude,
        )
