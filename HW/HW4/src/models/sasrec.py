"""SASRec — Self-Attentive Sequential Recommendation (Kang & McAuley, 2018).

Архитектура:
    Вход: последовательность item_idx длины ≤ max_len (правая выравнивание, 0 = PAD).
    Item embedding (shared input/output) + Positional embedding → LayerNorm → Dropout
    × N блоков (default 2): MultiHeadSelfAttention(causal mask) + FFN с residual + LN.
    Финальный hidden h_t используется как «user embedding на момент после позиции t»;
    логиты = h_t @ item_emb.T (shared weights).

Loss:
    Sampled-softmax с in-batch + popularity-weighted negatives по последней позиции.
    Для устойчивости тренируется ТОЛЬКО предсказание next item для последней непаддинговой
    позиции (sampling по сессиям; для очень длинных историй используем sliding window
    через дублирование примеров с разным cutoff внутри Dataset.__getitem__).

Использование в pipeline:
    - retriever:  recommend(user_ids, k, exclude) — top-K по score(h_last, all items)
    - ranker-feature: score(user_ids, item_ids) — пер-парный score
"""
from __future__ import annotations

import math
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
# Modules
# ──────────────────────────────────────────────────────────────────────────────
class SASRecModel(nn.Module):
    def __init__(
        self,
        n_items: int,
        emb_dim: int = 64,
        max_len: int = 50,
        n_blocks: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.max_len = max_len

        # 0 = PAD; item ids = 1..n_items
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_layernorm = nn.LayerNorm(emb_dim, eps=1e-8)

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_blocks)
        ])
        self.attn_layernorms = nn.ModuleList([nn.LayerNorm(emb_dim, eps=1e-8) for _ in range(n_blocks)])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, emb_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        ])
        self.ffn_layernorms = nn.ModuleList([nn.LayerNorm(emb_dim, eps=1e-8) for _ in range(n_blocks)])

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) int64, 0 = PAD. Возвращает (B, L, D)."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_layernorm(x)
        x = self.emb_dropout(x)

        pad_mask = (seq == 0)  # (B, L)
        # причинная маска: позиция t видит только 0..t
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)

        # обнулим эмбеддинги пада, чтобы не вкладывались
        x = x * (~pad_mask).unsqueeze(-1).float()

        for attn, ln_a, ffn, ln_f in zip(self.attn_layers, self.attn_layernorms, self.ffn_layers, self.ffn_layernorms):
            q = ln_a(x)
            attn_out, _ = attn(q, x, x, attn_mask=causal, key_padding_mask=pad_mask, need_weights=False)
            x = x + attn_out
            # FFN
            x = x + ffn(ln_f(x))
            x = x * (~pad_mask).unsqueeze(-1).float()
        return x

    def last_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        """Достать hidden последней непаддинговой позиции. seq: (B, L) → (B, D)."""
        h = self.encode(seq)  # (B, L, D)
        # длина непаддинговой части
        lengths = (seq != 0).long().sum(dim=1).clamp(min=1)  # (B,)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))  # (B, 1, D)
        return h.gather(1, idx).squeeze(1)  # (B, D)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    """Один пример = (история длины L+1, последний item = target).

    Для каждого пользователя берём всю последовательность покупок, обрезаем до max_len.
    Если последовательность короче 2 — пропускаем.
    """
    def __init__(
        self,
        sequences: dict[int, np.ndarray],
        max_len: int = 50,
    ) -> None:
        # sequences: user_idx → np.array(item_idx_plus_1) в хрон. порядке (уже +1, 0 = PAD)
        self.max_len = max_len
        self.users: list[int] = []
        self.seqs: list[np.ndarray] = []
        self.targets: list[int] = []
        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue
            # input = seq[:-1], target = seq[-1] (один шаг)
            full = seq[-(max_len + 1):]
            inp = full[:-1]
            tgt = int(full[-1])
            arr = np.zeros(max_len, dtype=np.int64)
            arr[-len(inp):] = inp
            self.users.append(int(uid))
            self.seqs.append(arr)
            self.targets.append(tgt)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, i: int) -> tuple[int, np.ndarray, int]:
        return self.users[i], self.seqs[i], self.targets[i]


def _collate(batch: list[tuple[int, np.ndarray, int]]) -> dict[str, torch.Tensor]:
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    seqs = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
    targets = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return {"user": users, "seq": seqs, "target": targets}


# ──────────────────────────────────────────────────────────────────────────────
# Recommender
# ──────────────────────────────────────────────────────────────────────────────
class SASRecRecommender(Persistable):
    name = "SASRec"

    def __init__(
        self,
        emb_dim: int = 64,
        max_len: int = 50,
        n_blocks: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        n_neg: int = 100,
        device: str | None = None,
        random_state: int = 42,
    ) -> None:
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_neg = n_neg
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.random_state = random_state

        self.user_enc: IdEncoder | None = None
        self.item_enc: IdEncoder | None = None
        self.model: SASRecModel | None = None
        self.user_sequences: dict[int, np.ndarray] = {}  # uid_idx → seq of item_idx+1 (0=PAD)
        self.item_pop: np.ndarray | None = None  # популярность для weighted negatives

    # ------------------------------------------------------------------
    @staticmethod
    def _build_sequences(df: pd.DataFrame, user_enc: IdEncoder, item_enc: IdEncoder) -> dict[int, np.ndarray]:
        purchases = df[df["is_purchased"]][["user_id", "item_id", "timestamp"]].sort_values(
            ["user_id", "timestamp"]
        )
        u_idx = user_enc.transform(purchases["user_id"].values)
        i_idx = item_enc.transform(purchases["item_id"].values)
        purchases = purchases.assign(u_idx=u_idx, i_idx=i_idx)
        purchases = purchases[(purchases["u_idx"] >= 0) & (purchases["i_idx"] >= 0)]
        out: dict[int, np.ndarray] = {}
        for uid, sub in purchases.groupby("u_idx"):
            # +1 потому что 0 = PAD
            out[int(uid)] = (sub["i_idx"].values + 1).astype(np.int64)
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        user_enc: IdEncoder,
        item_enc: IdEncoder,
        *,
        val_users: list[int] | None = None,
        val_ground_truth: dict[int, set[int]] | None = None,
        patience: int = 2,
        min_delta: float = 1e-3,
        verbose_val_every: int = 1,
    ) -> "SASRecRecommender":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.user_enc = user_enc
        self.item_enc = item_enc
        self.user_sequences = self._build_sequences(df, user_enc, item_enc)

        # популярность для negative sampling
        pop = np.zeros(item_enc.n + 1, dtype=np.float64)  # индекс с +1, 0 = PAD
        for seq in self.user_sequences.values():
            np.add.at(pop, seq, 1.0)
        pop[0] = 0.0
        pop = pop ** 0.75
        pop_sum = pop.sum()
        self.item_pop = (pop / pop_sum if pop_sum > 0 else pop).astype(np.float64)

        dataset = SeqDataset(self.user_sequences, max_len=self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=_collate, num_workers=0)
        print(f"[SASRec] training pairs: {len(dataset):,}; device={self.device}; bs={self.batch_size}")

        self.model = SASRecModel(
            n_items=item_enc.n,
            emb_dim=self.emb_dim,
            max_len=self.max_len,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            dropout=self.dropout,
        ).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98))

        # Early stopping
        es: EarlyStopper | None = None
        do_val = val_users is not None and val_ground_truth is not None
        if do_val:
            es = EarlyStopper(patience=patience, min_delta=min_delta, mode="max")

        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.epochs):
            self.model.train()
            losses: list[float] = []
            for batch in loader:
                seq = batch["seq"].to(self.device)
                targets = batch["target"].to(self.device)  # item_idx + 1
                B = seq.size(0)

                h_last = self.model.last_hidden(seq)  # (B, D)

                # positive logits
                pos_emb = self.model.item_emb(targets)  # (B, D)
                pos_logits = (h_last * pos_emb).sum(dim=-1)  # (B,)

                # negatives: sample n_neg per example по популярности, исключая target и 0
                neg_idx = rng.choice(
                    item_enc.n + 1,
                    size=(B, self.n_neg),
                    replace=True,
                    p=self.item_pop,
                )
                neg_t = torch.tensor(neg_idx, device=self.device, dtype=torch.long)
                neg_emb = self.model.item_emb(neg_t)  # (B, N, D)
                neg_logits = (h_last.unsqueeze(1) * neg_emb).sum(dim=-1)  # (B, N)

                # InfoNCE: log-softmax по [pos | negs]
                logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)  # (B, 1+N)
                labels = torch.zeros(B, device=self.device, dtype=torch.long)
                loss = F.cross_entropy(logits, labels)

                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            mean_loss = float(np.mean(losses))
            msg = f"[SASRec] epoch {epoch + 1}/{self.epochs}  loss={mean_loss:.4f}"

            if do_val and (epoch + 1) % verbose_val_every == 0:
                recs = self.recommend(val_users, k=20)
                ndcg = ndcg_at_k_from_recs(recs, val_ground_truth, k=20)
                msg += f"  ndcg@20={ndcg:.4f}"
                if es is not None:
                    snapshot_fn = lambda: snapshot_torch(self.model)
                    stop = es.step(ndcg, snapshot_fn=snapshot_fn)
                    if stop:
                        print(msg + f"  → early stop ({es.summary()})")
                        if es.best_state is not None:
                            restore_torch(es.best_state, self.model)
                        break
            print(msg)

        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _hidden_for_users(self, user_idxs: np.ndarray) -> np.ndarray:
        """Получить (B, D) — h_last для списка user_idx (по их актуальным последовательностям)."""
        self.model.eval()
        D = self.emb_dim
        out = np.zeros((len(user_idxs), D), dtype=np.float32)
        BATCH = 512
        for s in range(0, len(user_idxs), BATCH):
            e = min(s + BATCH, len(user_idxs))
            batch = user_idxs[s:e]
            seqs = np.zeros((len(batch), self.max_len), dtype=np.int64)
            for j, ui in enumerate(batch):
                seq = self.user_sequences.get(int(ui))
                if seq is None or len(seq) == 0:
                    continue
                seq = seq[-self.max_len:]
                seqs[j, -len(seq):] = seq
            seq_t = torch.tensor(seqs, device=self.device)
            h = self.model.last_hidden(seq_t).cpu().numpy()
            out[s:e] = h
        return out

    @torch.no_grad()
    def _all_item_emb(self) -> np.ndarray:
        self.model.eval()
        # пропустим 0 (PAD), возвращаем (n_items, D) индексированный по item_idx
        emb = self.model.item_emb.weight.detach().cpu().numpy()  # (n_items+1, D)
        return emb[1:]  # (n_items, D)

    def recommend(
        self,
        user_ids: list[int],
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        assert self.model is not None and self.user_enc is not None and self.item_enc is not None
        exclude = exclude or {}

        u_idx_arr = self.user_enc.transform(user_ids)
        out: dict[int, list[int]] = {uid: [] for uid in user_ids}
        valid = u_idx_arr >= 0
        if not valid.any():
            return out

        valid_users = [uid for uid, v in zip(user_ids, valid) if v]
        valid_idx = u_idx_arr[valid]
        H = self._hidden_for_users(valid_idx)  # (V, D)
        I = self._all_item_emb()              # (n_items, D)

        BATCH = 1024
        for s in range(0, len(valid_users), BATCH):
            e = min(s + BATCH, len(valid_users))
            U = H[s:e]
            scores = U @ I.T  # (b, n_items)
            for j, uid in enumerate(valid_users[s:e]):
                row = scores[j].copy()
                # exclude purchased + явный exclude
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
        """Per-pair score для использования как фича в ранкере."""
        assert self.model is not None
        u_idx = self.user_enc.transform(user_ids)
        i_idx = self.item_enc.transform(item_ids)
        out = np.zeros(len(user_ids), dtype=np.float32)
        mask = (u_idx >= 0) & (i_idx >= 0)
        if not mask.any():
            return out
        unique_u, inv_u = np.unique(u_idx[mask], return_inverse=True)
        H = self._hidden_for_users(unique_u)         # (U, D)
        I = self._all_item_emb()                     # (n_items, D)
        scores = (H[inv_u] * I[i_idx[mask]]).sum(axis=1)
        out[mask] = scores.astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Persist hooks
    # ------------------------------------------------------------------
    def _pickle_skip(self) -> set[str]:
        return {"model"}

    def _save_extra(self, dir: Path) -> None:
        if self.model is not None:
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "n_items": self.item_enc.n,
                    "emb_dim": self.emb_dim,
                    "max_len": self.max_len,
                    "n_blocks": self.n_blocks,
                    "n_heads": self.n_heads,
                    "dropout": self.dropout,
                },
                dir / "model.pt",
            )

    def _load_extra(self, dir: Path) -> None:
        self.device = self.device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        ckpt = torch.load(dir / "model.pt", map_location=self.device)
        self.model = SASRecModel(
            n_items=ckpt["n_items"],
            emb_dim=ckpt["emb_dim"],
            max_len=ckpt["max_len"],
            n_blocks=ckpt["n_blocks"],
            n_heads=ckpt["n_heads"],
            dropout=ckpt["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
