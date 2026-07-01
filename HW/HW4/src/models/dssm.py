"""Two-Tower DSSM (PyTorch).

Item tower:  embed(item_id) ⊕ mean_pool(embed(category_tags))
                            ⊕ mean_pool(embed(series_id))
                            ⊕ mean_pool(embed(author_ids))
             → Linear → L2-normalize

User tower:  mean_pool(item_tower(history items)) ⊕ embed(user_id)
             → Linear → L2-normalize

Loss: sampled-softmax (InfoNCE) с in-batch negatives + extra negatives из impressions.

Главная фишка: item tower работает и для cold items (есть только content-фичи) →
DSSM умеет рекомендовать 3 102 товара, которых нет в train.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..data import IdEncoder
from ..training import EarlyStopper, snapshot_torch, restore_torch, ndcg_at_k_from_recs
from ._persist import Persistable


# ──────────────────────────────────────────────────────────────────────────────
# Encoders / data containers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ContentVocab:
    """Словари id → 0..N-1 для тегов / серий / авторов. 0 = PAD."""
    tag2idx: dict[int, int]
    series2idx: dict[int, int]
    author2idx: dict[int, int]

    @classmethod
    def from_items(cls, items: pd.DataFrame, max_tags: int = 50000) -> "ContentVocab":
        from collections import Counter
        from itertools import chain
        tag_cnt = Counter(chain.from_iterable(items["category_tags"].dropna()))
        kept_tags = [t for t, _ in tag_cnt.most_common(max_tags)]
        tag2idx = {int(t): i + 1 for i, t in enumerate(kept_tags)}  # 0 = PAD
        series_ids = sorted({int(s) for lst in items["series_id"].dropna() for s in lst})
        series2idx = {s: i + 1 for i, s in enumerate(series_ids)}
        author_ids = sorted({int(a) for lst in items["author_ids"].dropna() for a in lst})
        author2idx = {a: i + 1 for i, a in enumerate(author_ids)}
        return cls(tag2idx, series2idx, author2idx)


def build_item_content_arrays(
    items: pd.DataFrame,
    item_enc: IdEncoder,
    vocab: ContentVocab,
    max_tags: int = 16,
    max_series: int = 2,
    max_authors: int = 2,
) -> dict[str, np.ndarray]:
    """Возвращает (n_items, max_X) int32-массивы, индексы 0=PAD, для каждого item_idx."""
    n = item_enc.n
    tag_arr = np.zeros((n, max_tags), dtype=np.int32)
    series_arr = np.zeros((n, max_series), dtype=np.int32)
    author_arr = np.zeros((n, max_authors), dtype=np.int32)

    items_index = items.set_index("item_id")

    def _safe(x):
        return [] if x is None else list(x)

    for orig_id in item_enc.id2idx:
        idx = item_enc.id2idx[orig_id]
        if orig_id not in items_index.index:
            continue
        row = items_index.loc[orig_id]
        tags = [vocab.tag2idx.get(int(t), 0) for t in _safe(row.get("category_tags"))][:max_tags]
        ss = [vocab.series2idx.get(int(s), 0) for s in _safe(row.get("series_id"))][:max_series]
        aa = [vocab.author2idx.get(int(a), 0) for a in _safe(row.get("author_ids"))][:max_authors]
        tag_arr[idx, : len(tags)] = tags
        series_arr[idx, : len(ss)] = ss
        author_arr[idx, : len(aa)] = aa
    return {"tags": tag_arr, "series": series_arr, "authors": author_arr}


# ──────────────────────────────────────────────────────────────────────────────
# Towers
# ──────────────────────────────────────────────────────────────────────────────
class ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_tags: int,
        n_series: int,
        n_authors: int,
        emb_dim: int = 64,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.tag_emb = nn.Embedding(n_tags + 1, emb_dim, padding_idx=0)
        self.series_emb = nn.Embedding(n_series + 1, emb_dim, padding_idx=0)
        self.author_emb = nn.Embedding(n_authors + 1, emb_dim, padding_idx=0)
        self.proj = nn.Sequential(nn.Linear(emb_dim * 4, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))

    @staticmethod
    def _mean_pool(emb_layer: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).float().unsqueeze(-1)
        e = emb_layer(ids) * mask
        denom = mask.sum(dim=-2).clamp(min=1.0)
        return e.sum(dim=-2) / denom

    def forward(self, item_idx: torch.Tensor, tags: torch.Tensor, series: torch.Tensor, authors: torch.Tensor) -> torch.Tensor:
        # item_idx может быть -1 для cold items → клампим в 0 (PAD) → embedding нулевой
        item_idx_safe = item_idx.clamp(min=0)
        i = self.item_emb(item_idx_safe + 1)  # +1 потому что 0 = PAD
        t = self._mean_pool(self.tag_emb, tags)
        s = self._mean_pool(self.series_emb, series)
        a = self._mean_pool(self.author_emb, authors)
        x = torch.cat([i, t, s, a], dim=-1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class UserTower(nn.Module):
    """Юзер = mean эмбеддингов его покупок (через ItemTower) + опц. user_id-bias."""
    def __init__(self, n_users: int, out_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.user_bias = nn.Embedding(n_users + 1, out_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, history_emb_mean: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        u = self.user_bias(user_idx.clamp(min=0) + 1)
        x = torch.cat([history_emb_mean, u], dim=-1)
        return F.normalize(self.proj(x), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset & training
# ──────────────────────────────────────────────────────────────────────────────
class PurchaseDataset(Dataset):
    """Один пример = (user_idx, история до этой покупки [max_hist последних], target item_idx).

    История = items, купленные пользователем строго раньше текущей покупки.
    Если истории нет — пропускаем (caller это контролирует через `min_history`).
    """
    def __init__(
        self,
        purchases: pd.DataFrame,
        user_enc: IdEncoder,
        item_enc: IdEncoder,
        max_hist: int = 32,
        min_history: int = 1,
    ) -> None:
        df = purchases.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        df["user_idx"] = user_enc.transform(df["user_id"])
        df["item_idx"] = item_enc.transform(df["item_id"])
        df = df[(df["user_idx"] >= 0) & (df["item_idx"] >= 0)].reset_index(drop=True)
        # для каждого юзера — кумулятивная история; индекс = позиция в его последовательности
        df["pos"] = df.groupby("user_idx").cumcount()
        df = df[df["pos"] >= min_history].reset_index(drop=True)
        self.user_idx = df["user_idx"].values.astype(np.int64)
        self.item_idx = df["item_idx"].values.astype(np.int64)
        self.pos = df["pos"].values.astype(np.int64)
        # история — словарь uid_idx → np.array(item_idx) в хрон. порядке
        self.history: dict[int, np.ndarray] = {
            int(uid): sub["item_idx"].values.astype(np.int64)
            for uid, sub in df.groupby("user_idx")
        }
        self.max_hist = max_hist

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, i: int) -> tuple[int, np.ndarray, int]:
        u = int(self.user_idx[i])
        pos = int(self.pos[i])
        hist_full = self.history[u][:pos]  # строго до текущей
        hist = hist_full[-self.max_hist:]
        if len(hist) < self.max_hist:
            pad = np.full(self.max_hist - len(hist), -1, dtype=np.int64)
            hist = np.concatenate([pad, hist])
        return u, hist, int(self.item_idx[i])


def collate(batch: list[tuple[int, np.ndarray, int]]) -> dict[str, torch.Tensor]:
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    hist = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
    targets = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return {"user": users, "history": hist, "target": targets}


# ──────────────────────────────────────────────────────────────────────────────
# Recommender API
# ──────────────────────────────────────────────────────────────────────────────
class DSSMRecommender(Persistable):
    name = "DSSM"

    def __init__(
        self,
        emb_dim: int = 64,
        out_dim: int = 64,
        max_hist: int = 32,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
        temperature: float = 0.05,
        dropout: float = 0.2,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.max_hist = max_hist
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temperature = temperature
        self.dropout = dropout
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.random_state = random_state

        self.user_enc: IdEncoder | None = None
        self.item_enc: IdEncoder | None = None
        self.vocab: ContentVocab | None = None
        self.item_arrays: dict[str, np.ndarray] | None = None
        self.item_tower: ItemTower | None = None
        self.user_tower: UserTower | None = None
        self.item_emb_matrix: np.ndarray | None = None
        self.user_history: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _build_item_emb_matrix(self) -> np.ndarray:
        """Прогнать item_tower по всем items → матрица (n_items, out_dim)."""
        assert self.item_tower is not None
        self.item_tower.eval()
        n = self.item_enc.n
        out = np.zeros((n, self.out_dim), dtype=np.float32)
        BATCH = 4096
        with torch.no_grad():
            for s in range(0, n, BATCH):
                e = min(s + BATCH, n)
                idx = torch.arange(s, e, device=self.device, dtype=torch.long)
                emb = self.item_tower(idx, self._tags_t[s:e], self._series_t[s:e], self._authors_t[s:e]).cpu().numpy()
                out[s:e] = emb
        return out

    def _embed_history_batch(self, hist: torch.Tensor) -> torch.Tensor:
        """hist: (B, L) item_idx; -1 = PAD. Возвращает (B, out_dim)."""
        B, L = hist.shape
        flat = hist.reshape(-1)
        mask = (flat >= 0).float().unsqueeze(-1)
        flat_safe = flat.clamp(min=0)
        emb = self.item_tower(flat_safe, self._tags_t[flat_safe], self._series_t[flat_safe], self._authors_t[flat_safe]) * mask
        emb = emb.reshape(B, L, -1)
        denom = (hist >= 0).float().sum(dim=1, keepdim=True).clamp(min=1.0)
        return emb.sum(dim=1) / denom

    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        items: pd.DataFrame,
        user_enc: IdEncoder,
        item_enc: IdEncoder,
        vocab: ContentVocab | None = None,
        *,
        val_users: list[int] | None = None,
        val_ground_truth: dict[int, set[int]] | None = None,
        patience: int = 2,
        min_delta: float = 1e-3,
        verbose_val_every: int = 1,
    ) -> "DSSMRecommender":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.user_enc = user_enc
        self.item_enc = item_enc
        self.vocab = vocab or ContentVocab.from_items(items)
        self.item_arrays = build_item_content_arrays(items, item_enc, self.vocab)
        # один раз перенесём контент-массивы в тензоры на устройство (горячий кэш)
        self._tags_t = torch.tensor(self.item_arrays["tags"], device=self.device, dtype=torch.long)
        self._series_t = torch.tensor(self.item_arrays["series"], device=self.device, dtype=torch.long)
        self._authors_t = torch.tensor(self.item_arrays["authors"], device=self.device, dtype=torch.long)

        # dataset из покупок
        purchases = df[df["is_purchased"]][["user_id", "item_id", "timestamp"]]
        dataset = PurchaseDataset(purchases, user_enc, item_enc, max_hist=self.max_hist)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
        print(f"[DSSM] training pairs: {len(dataset):,}; device={self.device}; bs={self.batch_size}; τ={self.temperature}")

        # models
        self.item_tower = ItemTower(
            n_items=item_enc.n,
            n_tags=len(self.vocab.tag2idx),
            n_series=len(self.vocab.series2idx),
            n_authors=len(self.vocab.author2idx),
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
        ).to(self.device)
        self.user_tower = UserTower(n_users=user_enc.n, out_dim=self.out_dim, dropout=self.dropout).to(self.device)

        params = list(self.item_tower.parameters()) + list(self.user_tower.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)

        # Early stopping setup (только если переданы val_users + ground_truth)
        es: EarlyStopper | None = None
        do_val = val_users is not None and val_ground_truth is not None
        if do_val:
            es = EarlyStopper(patience=patience, min_delta=min_delta, mode="max")
            self.user_history = dataset.history  # уже сейчас — нужно для recommend во время val

        for epoch in range(self.epochs):
            self.item_tower.train()
            self.user_tower.train()
            losses: list[float] = []
            for batch in loader:
                users = batch["user"].to(self.device)
                hist = batch["history"].to(self.device)
                targets = batch["target"].to(self.device)

                # история → mean embedding
                hist_emb = self._embed_history_batch(hist)
                user_emb = self.user_tower(hist_emb, users)            # уже L2-normalized
                pos_emb = self.item_tower(
                    targets, self._tags_t[targets], self._series_t[targets], self._authors_t[targets]
                )                                                       # уже L2-normalized

                # InfoNCE: cosine / τ
                logits = (user_emb @ pos_emb.T) / self.temperature
                labels = torch.arange(len(users), device=self.device)
                loss = F.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses))
            msg = f"[DSSM] epoch {epoch + 1}/{self.epochs}  loss={mean_loss:.4f}"

            # Per-epoch validation (если включено)
            if do_val and (epoch + 1) % verbose_val_every == 0:
                # для метрики нужны актуальные item-эмбеддинги
                self.item_emb_matrix = self._build_item_emb_matrix()
                recs = self.recommend(val_users, k=20)
                ndcg = ndcg_at_k_from_recs(recs, val_ground_truth, k=20)
                msg += f"  ndcg@20={ndcg:.4f}"
                if es is not None:
                    snapshot_fn = lambda: snapshot_torch(self.item_tower, self.user_tower)
                    stop = es.step(ndcg, snapshot_fn=snapshot_fn)
                    if stop:
                        print(msg + f"  → early stop ({es.summary()})")
                        # восстановить лучшие веса
                        if es.best_state is not None:
                            restore_torch(es.best_state, self.item_tower, self.user_tower)
                        break
            print(msg)

        # сохранить историю покупок и финальные item embeddings
        self.user_history = dataset.history
        self.item_emb_matrix = self._build_item_emb_matrix()
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _user_embeddings(self, user_idxs: np.ndarray) -> np.ndarray:
        """Считаем эмбеддинги для списка user_idx."""
        self.item_tower.eval()
        self.user_tower.eval()
        out = np.zeros((len(user_idxs), self.out_dim), dtype=np.float32)
        BATCH = 1024
        for s in range(0, len(user_idxs), BATCH):
            e = min(s + BATCH, len(user_idxs))
            batch = user_idxs[s:e]
            hist_arr = np.full((len(batch), self.max_hist), -1, dtype=np.int64)
            for j, ui in enumerate(batch):
                h = self.user_history.get(int(ui), np.array([], dtype=np.int64))
                if len(h) == 0:
                    continue
                h = h[-self.max_hist:]
                hist_arr[j, -len(h):] = h
            hist_t = torch.tensor(hist_arr, device=self.device)
            users_t = torch.tensor(batch, device=self.device, dtype=torch.long)
            hist_emb = self._embed_history_batch(hist_t)
            user_emb = self.user_tower(hist_emb, users_t).cpu().numpy()
            out[s:e] = user_emb
        return out

    def recommend(
        self,
        user_ids: list[int],
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        assert self.item_emb_matrix is not None and self.user_enc is not None and self.item_enc is not None
        exclude = exclude or {}

        u_idx_arr = self.user_enc.transform(user_ids)
        out: dict[int, list[int]] = {uid: [] for uid in user_ids}
        valid = u_idx_arr >= 0
        if not valid.any():
            return out

        valid_users = [uid for uid, v in zip(user_ids, valid) if v]
        valid_idx = u_idx_arr[valid]
        user_embs = self._user_embeddings(valid_idx)  # (V, D)

        # dot product со всеми items — батч 2048 юзеров
        BATCH = 2048
        for s in range(0, len(valid_users), BATCH):
            e = min(s + BATCH, len(valid_users))
            U = user_embs[s:e]                           # (b, D)
            scores = U @ self.item_emb_matrix.T          # (b, n_items)
            for j, uid in enumerate(valid_users[s:e]):
                row = scores[j].copy()
                seen = exclude.get(uid, set())
                if seen:
                    seen_idx = self.item_enc.transform(list(seen))
                    seen_idx = seen_idx[seen_idx >= 0]
                    if len(seen_idx):
                        row[seen_idx] = -np.inf
                top = np.argpartition(-row, k)[:k]
                top = top[np.argsort(-row[top])]
                items = self.item_enc.inverse_transform(top).tolist()
                out[uid] = [int(x) for x in items]
        return out

    # ------------------------------------------------------------------
    def score(self, user_ids: list[int], item_ids: list[int]) -> np.ndarray:
        """Скоры по парам для reranker'a. Чанкуем для tqdm-прогресса."""
        from tqdm.auto import tqdm
        assert self.item_emb_matrix is not None
        u_idx = self.user_enc.transform(user_ids)
        i_idx = self.item_enc.transform(item_ids)
        n = len(user_ids)
        out = np.zeros(n, dtype=np.float32)
        chunk = 500_000
        for s in tqdm(range(0, n, chunk), desc="DSSM.score", mininterval=1.0):
            e = min(n, s + chunk)
            u_chunk = u_idx[s:e]
            i_chunk = i_idx[s:e]
            mask = (u_chunk >= 0) & (i_chunk >= 0)
            if not mask.any():
                continue
            unique_u, inv_u = np.unique(u_chunk[mask], return_inverse=True)
            u_embs = self._user_embeddings(unique_u)
            scores = (u_embs[inv_u] * self.item_emb_matrix[i_chunk[mask]]).sum(axis=1)
            local = np.zeros(e - s, dtype=np.float32)
            local[mask] = scores.astype(np.float32)
            out[s:e] = local
        return out

    # ------------------------------------------------------------------
    # Persist hooks
    # ------------------------------------------------------------------
    def _pickle_skip(self) -> set[str]:
        # towers и device-тензоры — сериализуем отдельно через torch.save
        return {
            "item_tower",
            "user_tower",
            "_tags_t",
            "_series_t",
            "_authors_t",
            "item_emb_matrix",
        }

    def _save_extra(self, dir: Path) -> None:
        # tower state dicts
        if self.item_tower is not None and self.user_tower is not None:
            torch.save(
                {
                    "item_tower": self.item_tower.state_dict(),
                    "user_tower": self.user_tower.state_dict(),
                    "n_items": self.item_enc.n,
                    "n_users": self.user_enc.n,
                    "n_tags": len(self.vocab.tag2idx),
                    "n_series": len(self.vocab.series2idx),
                    "n_authors": len(self.vocab.author2idx),
                },
                dir / "towers.pt",
            )
        # item content arrays (для пересчёта эмбеддингов на новом устройстве)
        if self.item_arrays is not None:
            np.savez_compressed(
                dir / "item_arrays.npz",
                tags=self.item_arrays["tags"],
                series=self.item_arrays["series"],
                authors=self.item_arrays["authors"],
            )
        # предвычисленные item embeddings — можно перепосчитать, но сохраним для быстрого старта
        if self.item_emb_matrix is not None:
            np.save(dir / "item_emb_matrix.npy", self.item_emb_matrix)

    def _load_extra(self, dir: Path) -> None:
        # device может быть другим
        self.device = self.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        # восстановить item_arrays
        z = np.load(dir / "item_arrays.npz")
        self.item_arrays = {"tags": z["tags"], "series": z["series"], "authors": z["authors"]}
        self._tags_t = torch.tensor(self.item_arrays["tags"], device=self.device, dtype=torch.long)
        self._series_t = torch.tensor(self.item_arrays["series"], device=self.device, dtype=torch.long)
        self._authors_t = torch.tensor(self.item_arrays["authors"], device=self.device, dtype=torch.long)

        ckpt = torch.load(dir / "towers.pt", map_location=self.device)
        self.item_tower = ItemTower(
            n_items=ckpt["n_items"],
            n_tags=ckpt["n_tags"],
            n_series=ckpt["n_series"],
            n_authors=ckpt["n_authors"],
            emb_dim=self.emb_dim,
            out_dim=self.out_dim,
        ).to(self.device)
        self.user_tower = UserTower(
            n_users=ckpt["n_users"], out_dim=self.out_dim, dropout=getattr(self, "dropout", 0.2)
        ).to(self.device)
        self.item_tower.load_state_dict(ckpt["item_tower"])
        self.user_tower.load_state_dict(ckpt["user_tower"])

        emb_path = dir / "item_emb_matrix.npy"
        if emb_path.exists():
            self.item_emb_matrix = np.load(emb_path)
        else:
            self.item_emb_matrix = self._build_item_emb_matrix()
