"""Загрузка данных и time-based split.

Все таблицы используют исходные `user_id` / `item_id` (как в условии).
Для матричных моделей предоставлены отдельные кодеры в `IdEncoder` —
они мэппят исходные id в плотные 0..N-1 индексы и обратно.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
HW_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = HW_DIR / "dataset"
DATAMART_DIR = HW_DIR / "datamart"

TRAIN_PATH = DATA_DIR / "train.pq"
ITEMS_PATH = DATA_DIR / "items.pq"
TEST_USERS_PATH = DATA_DIR / "test_users.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────
def load_train(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Train interactions. Гарантируем datetime64[ns] для timestamp."""
    df = pd.read_parquet(path)
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        sample = df["timestamp"].iloc[0]
        unit = "s" if isinstance(sample, (int, np.integer, float, np.floating)) and sample < 1e11 else "ms"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit)
    # downcast: экономия памяти
    df["user_id"] = df["user_id"].astype("int32")
    df["item_id"] = df["item_id"].astype("int32")
    df["rating"] = df["rating"].astype("int8")
    df["is_purchased"] = df["is_purchased"].astype(bool)
    return df


def load_items(path: Path = ITEMS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["item_id"] = df["item_id"].astype("int32")
    return df


def load_test_users(path: Path = TEST_USERS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["user_id"] = df["user_id"].astype("int32")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Time-based split
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TimeSplit:
    """Контейнер с подмножествами train_fit / val / holdout и их границами."""
    train_fit: pd.DataFrame
    val: pd.DataFrame
    holdout: pd.DataFrame
    cutoff_val: pd.Timestamp
    cutoff_holdout: pd.Timestamp

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rows": [len(self.train_fit), len(self.val), len(self.holdout)],
                "users": [
                    self.train_fit["user_id"].nunique(),
                    self.val["user_id"].nunique(),
                    self.holdout["user_id"].nunique(),
                ],
                "purchases": [
                    int(self.train_fit["is_purchased"].sum()),
                    int(self.val["is_purchased"].sum()),
                    int(self.holdout["is_purchased"].sum()),
                ],
                "t_min": [s["timestamp"].min() for s in (self.train_fit, self.val, self.holdout)],
                "t_max": [s["timestamp"].max() for s in (self.train_fit, self.val, self.holdout)],
            },
            index=["train_fit", "val", "holdout"],
        )


def make_time_split(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    holdout_frac: float = 0.05,
) -> TimeSplit:
    """Глобальный time-based split по квантилям `timestamp`.

    train_fit: до Q(1 - val_frac - holdout_frac)
    val      : между Q(1 - val_frac - holdout_frac) и Q(1 - holdout_frac)
    holdout  : после Q(1 - holdout_frac)
    """
    assert 0 < val_frac + holdout_frac < 1
    q_val = 1.0 - val_frac - holdout_frac
    q_hold = 1.0 - holdout_frac
    cutoff_val = df["timestamp"].quantile(q_val)
    cutoff_holdout = df["timestamp"].quantile(q_hold)

    train_fit = df[df["timestamp"] <= cutoff_val].copy()
    val = df[(df["timestamp"] > cutoff_val) & (df["timestamp"] <= cutoff_holdout)].copy()
    holdout = df[df["timestamp"] > cutoff_holdout].copy()
    return TimeSplit(train_fit, val, holdout, cutoff_val, cutoff_holdout)


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth для оценки
# ──────────────────────────────────────────────────────────────────────────────
def build_ground_truth(
    eval_df: pd.DataFrame,
    train_history: pd.DataFrame,
    only_purchases: bool = True,
) -> dict[int, set[int]]:
    """Для каждого пользователя — множество товаров, купленных в eval-периоде,
    за вычетом уже купленных в train-истории (чтобы не награждать модель
    за тривиальные «повтори то что уже купил»).
    """
    eval_pos = eval_df[eval_df["is_purchased"]] if only_purchases else eval_df
    seen_in_train = (
        train_history[train_history["is_purchased"]]
        .groupby("user_id")["item_id"]
        .agg(set)
        .to_dict()
    )
    out: dict[int, set[int]] = {}
    for uid, sub in eval_pos.groupby("user_id"):
        items = set(sub["item_id"].tolist())
        items -= seen_in_train.get(uid, set())
        if items:
            out[int(uid)] = items
    return out


def build_user_history(
    df: pd.DataFrame,
    only_purchases: bool = False,
) -> dict[int, set[int]]:
    """История взаимодействий пользователя — для filter-already-seen на инференсе."""
    src = df[df["is_purchased"]] if only_purchases else df
    return src.groupby("user_id")["item_id"].agg(set).to_dict()


# ──────────────────────────────────────────────────────────────────────────────
# ID encoders (для матричных/нейронных моделей)
# ──────────────────────────────────────────────────────────────────────────────
class IdEncoder:
    """Двусторонний мэппинг id → 0..N-1.

    Обучается на trainset (например, train_fit), потом применяется ко всему.
    Неизвестные id возвращают -1 (caller отвечает за фильтрацию).
    """

    def __init__(self) -> None:
        self.id2idx: dict[int, int] = {}
        self.idx2id: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.id2idx)

    def fit(self, ids: Iterable[int]) -> "IdEncoder":
        uniq = np.array(sorted(set(int(i) for i in ids)), dtype=np.int64)
        self.idx2id = uniq
        self.id2idx = {int(v): i for i, v in enumerate(uniq)}
        return self

    def transform(self, ids: Iterable[int]) -> np.ndarray:
        if isinstance(ids, pd.Series):
            return ids.map(self.id2idx).fillna(-1).astype(np.int64).values
        return np.fromiter((self.id2idx.get(int(i), -1) for i in ids), dtype=np.int64)

    def inverse_transform(self, idxs: np.ndarray) -> np.ndarray:
        assert self.idx2id is not None
        return self.idx2id[np.asarray(idxs)]


def fit_encoders(train_fit: pd.DataFrame, items: pd.DataFrame | None = None) -> tuple[IdEncoder, IdEncoder]:
    """User encoder — по train_fit. Item encoder — по объединению train_fit и каталога
    (чтобы знать про холодные товары)."""
    user_enc = IdEncoder().fit(train_fit["user_id"].unique())
    item_ids: set[int] = set(train_fit["item_id"].unique().tolist())
    if items is not None:
        item_ids.update(items["item_id"].unique().tolist())
    item_enc = IdEncoder().fit(item_ids)
    return user_enc, item_enc
