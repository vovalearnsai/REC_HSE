"""Хелперы для матричных моделей (ALS / EASE / DSSM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import IdEncoder


def build_interaction_matrix(
    df: pd.DataFrame,
    user_enc: IdEncoder,
    item_enc: IdEncoder,
    *,
    weight: str = "binary",
    alpha: float = 1.0,
    beta: float = 0.5,
    drop_unknown: bool = True,
) -> sp.csr_matrix:
    """Построить разреженную матрицу взаимодействий (n_users, n_items).

    weight:
      - 'binary'       — 1 для каждого взаимодействия, агрегация max → {0,1}
      - 'count'        — число взаимодействий
      - 'confidence'   — `1 + α·is_purchased + β·log(1+rating)` (для ALS)
      - 'purchases'    — 1 только если был purchase (для EASE)
    """
    u = user_enc.transform(df["user_id"])
    i = item_enc.transform(df["item_id"])
    mask = (u >= 0) & (i >= 0)
    if mask.mean() < 1 and drop_unknown:
        df = df.loc[mask].copy()
        u = u[mask]
        i = i[mask]

    if weight == "binary":
        v = np.ones(len(df), dtype=np.float32)
    elif weight == "count":
        v = np.ones(len(df), dtype=np.float32)
    elif weight == "confidence":
        v = (
            1.0
            + alpha * df["is_purchased"].astype("float32").values
            + beta * np.log1p(df["rating"].astype("float32").values)
        ).astype(np.float32)
    elif weight == "purchases":
        v = df["is_purchased"].astype("float32").values
    else:
        raise ValueError(f"Unknown weight: {weight}")

    n_u, n_i = user_enc.n, item_enc.n
    mat = sp.coo_matrix((v, (u, i)), shape=(n_u, n_i), dtype=np.float32)
    mat = mat.tocsr()
    mat.sum_duplicates()
    if weight in ("binary", "purchases"):
        # бинаризация после агрегации
        mat.data = (mat.data > 0).astype(np.float32)
    return mat


def topk_from_scores(
    scores: np.ndarray,
    k: int,
    *,
    exclude_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K по scores (1D или 2D). Возвращает (indices, values)."""
    if scores.ndim == 1:
        if exclude_idx is not None and len(exclude_idx):
            scores = scores.copy()
            scores[exclude_idx] = -np.inf
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return order, scores[order]
    raise NotImplementedError("Use per-row in caller for batched.")
