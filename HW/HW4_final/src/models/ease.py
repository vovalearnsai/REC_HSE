"""EASE — Embarrassingly Shallow Autoencoder (Steck, 2019).

Закрытая форма: B = -(G + λI)^{-1} G with diag(B) = 0, где G = X^T X.
Score(u, i) = (X B)_{u,i}.

Особенности:
- Не использует item features → не предложит cold-items, нет в train.
- При |items| ≈ 31k матрица 31k×31k float32 ≈ 3.7 GB → решаемо. Если памяти мало —
  предусмотрен `n_top_items` для редуцирования каталога до самых популярных.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import IdEncoder
from ._matrix import build_interaction_matrix
from ._persist import Persistable


class EASE(Persistable):
    name = "EASE"

    def __init__(
        self,
        lam: float = 500.0,
        n_top_items: int | None = None,  # ограничить каталог самыми популярными
    ) -> None:
        self.lam = lam
        self.n_top_items = n_top_items

        self.user_enc: IdEncoder | None = None
        self.item_enc: IdEncoder | None = None
        self.B: np.ndarray | None = None
        self.X: sp.csr_matrix | None = None  # бинарная user×item для исключения уже купленного
        self._kept_items_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, user_enc: IdEncoder, item_enc: IdEncoder) -> "EASE":
        self.user_enc = user_enc
        self.item_enc = item_enc

        # Бинарная матрица покупок (EASE классически на implicit binary).
        X = build_interaction_matrix(df, user_enc, item_enc, weight="purchases").astype(np.float32)

        if self.n_top_items is not None and self.n_top_items < X.shape[1]:
            pop = np.asarray(X.sum(axis=0)).ravel()
            keep = np.argsort(-pop)[: self.n_top_items]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[keep] = True
            self._kept_items_mask = mask
            X = X[:, mask]

        print(f"[EASE] X: {X.shape}, nnz={X.nnz:,}")
        G = (X.T @ X).toarray().astype(np.float32)  # n_items × n_items
        diag_idx = np.diag_indices_from(G)
        G[diag_idx] += self.lam
        print(f"[EASE] solving inverse {G.shape}…")
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[diag_idx] = 0.0
        self.B = B.astype(np.float32)
        self.X = X
        print(f"[EASE] B ready: {self.B.shape}")
        return self

    # ------------------------------------------------------------------
    def _user_scores(self, u_idx: int) -> np.ndarray:
        """Скоры по всем items для одного юзера."""
        assert self.X is not None and self.B is not None
        row = self.X.getrow(u_idx)  # 1 × n_items
        return np.asarray(row @ self.B).ravel()

    def _expand_to_full_items(self, scores_reduced: np.ndarray) -> np.ndarray:
        """Если редуцировали каталог, вернём скоры в исходное пространство item_enc."""
        if self._kept_items_mask is None:
            return scores_reduced
        full = np.full(self._kept_items_mask.shape[0], -np.inf, dtype=np.float32)
        full[self._kept_items_mask] = scores_reduced
        return full

    def recommend(
        self,
        user_ids: list[int],
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        assert self.user_enc is not None and self.item_enc is not None and self.X is not None and self.B is not None
        exclude = exclude or {}

        u_idx_arr = self.user_enc.transform(user_ids)
        out: dict[int, list[int]] = {}

        # Батчевый matmul по группам — экономия памяти.
        BATCH = 1024
        for start in range(0, len(user_ids), BATCH):
            batch_uids = user_ids[start : start + BATCH]
            batch_idx = u_idx_arr[start : start + BATCH]
            for uid, ui in zip(batch_uids, batch_idx):
                if ui < 0:
                    out[uid] = []
                    continue
            valid_pairs = [(uid, ui) for uid, ui in zip(batch_uids, batch_idx) if ui >= 0]
            if not valid_pairs:
                continue
            valid_uids = [p[0] for p in valid_pairs]
            valid_idx = np.array([p[1] for p in valid_pairs])
            sub_X = self.X[valid_idx]                                  # (B, n_items_red)
            scores = np.asarray(sub_X @ self.B)                        # (B, n_items_red)
            scores[sub_X.astype(bool).toarray()] = -np.inf             # уже куплено → не рекомендуем

            for j, uid in enumerate(valid_uids):
                row_scores = scores[j]
                full_scores = self._expand_to_full_items(row_scores)
                seen = exclude.get(uid, set())
                if seen:
                    seen_idx = self.item_enc.transform(list(seen))
                    seen_idx = seen_idx[seen_idx >= 0]
                    if len(seen_idx):
                        full_scores[seen_idx] = -np.inf
                top_k_idx = np.argpartition(-full_scores, kth=min(k, len(full_scores) - 1))[:k]
                top_k_idx = top_k_idx[np.argsort(-full_scores[top_k_idx])]
                items = self.item_enc.inverse_transform(top_k_idx)
                items = [int(it) for it, s in zip(items, full_scores[top_k_idx]) if np.isfinite(s)]
                out[uid] = items
        return out

    # ------------------------------------------------------------------
    def score(self, user_ids: list[int], item_ids: list[int]) -> np.ndarray:
        """Скоры по парам — для reranker'а.

        Реализация — мемори-аккуратная: для каждого уникального юзера достаём ОДНУ
        sparse-строку из X и считаем `row @ B[:, items_for_this_user]`. Память:
        ~ O(nnz(row) + k_items_per_user); никаких dense (N_users × N_items) матриц.
        """
        assert self.user_enc is not None and self.item_enc is not None and self.X is not None and self.B is not None
        u_idx = self.user_enc.transform(user_ids)
        i_idx = self.item_enc.transform(item_ids)
        # Если редуцирован каталог — перевести в reduced пространство B
        if self._kept_items_mask is not None:
            reduced_idx = np.full(self._kept_items_mask.shape[0], -1, dtype=np.int64)
            reduced_idx[self._kept_items_mask] = np.arange(int(self._kept_items_mask.sum()))
            i_idx_red = np.where(i_idx >= 0, reduced_idx[i_idx], -1)
        else:
            i_idx_red = i_idx
        out = np.zeros(len(user_ids), dtype=np.float32)
        valid = (u_idx >= 0) & (i_idx_red >= 0)
        if not valid.any():
            return out

        valid_pos = np.where(valid)[0]
        sub_users = u_idx[valid_pos]
        sub_items = i_idx_red[valid_pos]

        # Группируем по уникальному юзеру (сортированно), чтобы потом нарезать
        # на батчи юзеров и делать один большой sparse @ dense matmul на батч.
        order = np.argsort(sub_users, kind="stable")
        sorted_u = sub_users[order]
        sorted_i = sub_items[order]
        sorted_p = valid_pos[order]
        uniq_u, starts = np.unique(sorted_u, return_index=True)
        ends = np.append(starts[1:], len(sorted_u))

        # Параметр батча: 4096 уникальных юзеров → ~ batch_u × n_items_red × 4B памяти.
        # Для 30k items это ~480MB — приемлемо и в разы быстрее, чем per-user fancy index.
        from tqdm.auto import tqdm
        BATCH_U = 4096
        n_uniq = len(uniq_u)
        for bu_s in tqdm(range(0, n_uniq, BATCH_U), desc="EASE.score", mininterval=1.0):
            bu_e = min(n_uniq, bu_s + BATCH_U)
            users_batch = uniq_u[bu_s:bu_e]
            # X[users_batch] : sparse (k × n_items_red)
            X_batch = self.X[users_batch]
            # Z : dense (k × n_items_red). Это самая дорогая операция —
            # один sparse-dense matmul вместо тысяч мелких.
            Z = np.asarray(X_batch @ self.B)
            # Разносим скоры по позициям, не материализуя fancy-копии столбцов.
            for local_idx in range(bu_e - bu_s):
                global_idx = bu_s + local_idx
                s = int(starts[global_idx]); e = int(ends[global_idx])
                items_chunk = sorted_i[s:e]
                out[sorted_p[s:e]] = Z[local_idx, items_chunk].astype(np.float32, copy=False)
        return out

    # ------------------------------------------------------------------
    # Persist hooks
    # ------------------------------------------------------------------
    def _pickle_skip(self) -> set[str]:
        return {"B", "X"}

    def _save_extra(self, dir: Path) -> None:
        if self.B is not None:
            np.save(dir / "B.npy", self.B)
        if self.X is not None:
            sp.save_npz(dir / "X.npz", self.X.tocsr())

    def _load_extra(self, dir: Path) -> None:
        self.B = np.load(dir / "B.npy").astype(np.float32)
        self.X = sp.load_npz(dir / "X.npz").tocsr().astype(np.float32)
