"""Training utilities: early stopping + per-epoch evaluation helpers.

Используется DSSM и SASRec для остановки обучения, когда метрика на val
перестаёт улучшаться, и для сохранения «лучших» весов модели.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np


class EarlyStopper:
    """Минимальный early-stopping: терпение `patience` эпох без улучшения ≥ `min_delta`.

    Использование:
        es = EarlyStopper(patience=2, min_delta=1e-3, mode='max')
        for epoch in range(max_epochs):
            ...
            metric = evaluate(...)
            if es.step(metric, snapshot=lambda: copy_state_dict(model)):
                break
        if es.best_state is not None:
            model.load_state_dict(es.best_state)
    """

    def __init__(
        self,
        patience: int = 2,
        min_delta: float = 1e-3,
        mode: str = "max",
    ) -> None:
        assert mode in {"max", "min"}
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_metric: float = -np.inf if mode == "max" else np.inf
        self.best_epoch: int = -1
        self.best_state: Optional[dict] = None
        self.num_bad_epochs: int = 0
        self.epoch: int = -1
        self.history: list[float] = []

    def _is_better(self, metric: float) -> bool:
        if self.mode == "max":
            return metric > self.best_metric + self.min_delta
        return metric < self.best_metric - self.min_delta

    def step(self, metric: float, snapshot_fn=None) -> bool:
        """Передаёт значение метрики; возвращает True если пора останавливаться.

        snapshot_fn — функция без аргументов, возвращает state-dict-like объект,
        который будет сохранён как `best_state`. Если None — не сохраняем.
        """
        self.epoch += 1
        self.history.append(float(metric))
        if self._is_better(metric):
            self.best_metric = float(metric)
            self.best_epoch = self.epoch
            self.num_bad_epochs = 0
            if snapshot_fn is not None:
                self.best_state = snapshot_fn()
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def summary(self) -> str:
        return (
            f"best metric={self.best_metric:.4f} at epoch {self.best_epoch + 1}; "
            f"history={['%.4f' % x for x in self.history]}"
        )


def snapshot_torch(*modules) -> dict:
    """Сохранить state_dict нескольких nn.Module как один dict (с CPU-копией)."""
    return {f"m{i}": {k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for i, m in enumerate(modules)}


def restore_torch(state: dict, *modules) -> None:
    """Восстановить state_dict, обратно к torch_snapshot()."""
    for i, m in enumerate(modules):
        m.load_state_dict(state[f"m{i}"])


def ndcg_at_k_from_recs(
    recs: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 20,
) -> float:
    """Простая реализация средне-юзерского NDCG@k поверх recs/gt-dict.

    Не зависит от src.metrics, чтобы избежать циклических импортов.
    """
    if not recs:
        return 0.0
    # precomputed log discounts
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    idcg_max = discounts.sum()  # все k попаданий
    total = 0.0
    n = 0
    for uid, items in recs.items():
        gt = ground_truth.get(uid)
        if not gt:
            continue
        hits = np.array([1.0 if it in gt else 0.0 for it in items[:k]])
        if hits.sum() == 0:
            total += 0.0
        else:
            dcg = (hits * discounts[: len(hits)]).sum()
            ideal_hits = min(len(gt), k)
            idcg = discounts[:ideal_hits].sum()
            total += dcg / (idcg if idcg > 0 else 1.0)
        n += 1
    return total / n if n else 0.0
