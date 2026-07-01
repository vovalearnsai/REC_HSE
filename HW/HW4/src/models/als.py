"""ALS (implicit feedback) wrapper.

Использует пакет `implicit`. Confidence-матрица:
    1 + α · is_purchased + β · log(1 + rating)
Этим мы кодируем «силу» сигнала: покупка > клик; высокий рейтинг > низкий.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import IdEncoder
from ._matrix import build_interaction_matrix
from ._persist import Persistable


class ALSRecommender(Persistable):
    name = "ALS"

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.05,
        iterations: int = 20,
        alpha_conf: float = 40.0,  # масштаб для confidence (как в Hu, Koren, Volinsky)
        purchase_weight: float = 1.0,
        rating_weight: float = 0.5,
        use_gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.reg = regularization
        self.iters = iterations
        self.alpha_conf = alpha_conf
        self.purchase_weight = purchase_weight
        self.rating_weight = rating_weight
        self.use_gpu = use_gpu
        self.random_state = random_state

        self.user_enc: IdEncoder | None = None
        self.item_enc: IdEncoder | None = None
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_items: sp.csr_matrix | None = None  # для filter_already_liked
        self._model = None

    def fit(self, df: pd.DataFrame, user_enc: IdEncoder, item_enc: IdEncoder) -> "ALSRecommender":
        from implicit.als import AlternatingLeastSquares

        self.user_enc = user_enc
        self.item_enc = item_enc

        # confidence-матрица. Затем умножаем на alpha_conf, как принято в implicit.
        ui = build_interaction_matrix(
            df,
            user_enc,
            item_enc,
            weight="confidence",
            alpha=self.purchase_weight,
            beta=self.rating_weight,
        )
        ui_conf = (ui * self.alpha_conf).tocsr()
        self.user_items = ui  # без alpha — для filter

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iters,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
        )
        self._model.fit(ui_conf, show_progress=True)
        self.user_factors = np.asarray(self._model.user_factors)
        self.item_factors = np.asarray(self._model.item_factors)
        return self

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_ids: list[int],
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        assert self._model is not None and self.user_enc is not None and self.item_enc is not None
        exclude = exclude or {}

        # маппинг → внутренние idx
        u_idx_arr = self.user_enc.transform(user_ids)
        out: dict[int, list[int]] = {uid: [] for uid in user_ids}

        # бьём батчами через recommend_all для скорости
        known_mask = u_idx_arr >= 0
        unknown_users = [uid for uid, k_ in zip(user_ids, known_mask) if not k_]
        known_users = [uid for uid, k_ in zip(user_ids, known_mask) if k_]
        known_idx = u_idx_arr[known_mask]
        if len(known_idx) == 0:
            return out

        # implicit умеет recommend(userid, user_items, N, filter_already_liked_items=True)
        # Делаем батч-вариант: model.recommend на каждого, или через .recommend для array.
        ids, scores = self._model.recommend(
            known_idx,
            self.user_items[known_idx],
            N=k + 200,  # запас для exclude
            filter_already_liked_items=True,
        )
        # ids: shape (B, k+200), item indices in encoder space; -1 при недостатке кандидатов
        for row, uid in enumerate(known_users):
            seen = exclude.get(uid, set())
            rec: list[int] = []
            for idx in ids[row]:
                if idx < 0:
                    break
                item_id = int(self.item_enc.inverse_transform(np.array([idx]))[0])
                if item_id in seen:
                    continue
                rec.append(item_id)
                if len(rec) == k:
                    break
            out[uid] = rec
        return out

    # ------------------------------------------------------------------
    # Persist hooks
    # ------------------------------------------------------------------
    def _pickle_skip(self) -> set[str]:
        # implicit AlternatingLeastSquares handle и тяжёлые массивы выгружаем отдельно
        return {"_model", "user_factors", "item_factors", "user_items"}

    def _save_extra(self, dir: Path) -> None:
        if self.user_factors is not None:
            np.save(dir / "user_factors.npy", self.user_factors)
        if self.item_factors is not None:
            np.save(dir / "item_factors.npy", self.item_factors)
        if self.user_items is not None:
            sp.save_npz(dir / "user_items.npz", self.user_items.tocsr())

    def _load_extra(self, dir: Path) -> None:
        self.user_factors = np.load(dir / "user_factors.npy")
        self.item_factors = np.load(dir / "item_factors.npy")
        self.user_items = sp.load_npz(dir / "user_items.npz").tocsr()
        # реконструируем implicit-handle и инъектим факторы (нужно для recommend())
        from implicit.als import AlternatingLeastSquares
        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iters,
            use_gpu=self.use_gpu,
            random_state=self.random_state,
        )
        self._model.user_factors = self.user_factors
        self._model.item_factors = self.item_factors

    # ------------------------------------------------------------------
    def score(self, user_ids: list[int], item_ids: list[int]) -> np.ndarray:
        """Скор по парам (user_ids[i], item_ids[i]) — для reranker'a.

        Чанкуем по 500к пар, чтобы показать tqdm-прогресс и держать память умеренной.
        """
        from tqdm.auto import tqdm
        assert self.user_factors is not None and self.item_factors is not None
        u_idx = self.user_enc.transform(user_ids)
        i_idx = self.item_enc.transform(item_ids)
        n = len(user_ids)
        out = np.zeros(n, dtype=np.float32)
        chunk = 500_000
        for s in tqdm(range(0, n, chunk), desc="ALS.score", mininterval=1.0):
            e = min(n, s + chunk)
            u_chunk = u_idx[s:e]
            i_chunk = i_idx[s:e]
            mask = (u_chunk >= 0) & (i_chunk >= 0)
            if not mask.any():
                continue
            uf = self.user_factors[u_chunk[mask]]
            if_ = self.item_factors[i_chunk[mask]]
            local = np.zeros(e - s, dtype=np.float32)
            local[mask] = np.einsum("ij,ij->i", uf, if_).astype(np.float32)
            out[s:e] = local
        return out
