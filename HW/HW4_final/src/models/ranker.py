"""LightGBM LambdaRank reranker.

Train:
    - данные = `impressions_long` (1 группа = 1 показ слейта, 20 строк, 0/1 покупок)
    - фичи = retrieval scores/ranks (только для тех пар, что попали в импрешн)
             + user_stats + item_stats + content + time
    - target = `was_purchased` (binary)
    - group = `impression_id`
    - objective = `lambdarank`, eval = `ndcg@20`

Inference:
    - данные = candidates pool (200 на юзера) с теми же фичами
    - модель предсказывает scores → сортировка по user_id → top-20
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ._persist import Persistable


@dataclass
class LGBMRanker(Persistable):
    name: str = "LGBMRanker"
    n_estimators: int = 500
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_data_in_leaf: int = 50
    reg_lambda: float = 1.0
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    random_state: int = 42

    feature_cols: list[str] = field(default_factory=list)
    model: object | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        group_col: str,
        label_col: str = "label",
        feature_cols: list[str] | None = None,
        eval_df: pd.DataFrame | None = None,
        eval_group_col: str = "user_id",
        eval_at: tuple[int, ...] = (20,),
        verbose: int = 50,
    ) -> "LGBMRanker":
        import gc
        import lightgbm as lgb

        if feature_cols is None:
            drop = {"user_id", "item_id", "label", "impression_id", "timestamp"}
            feature_cols = [c for c in train_df.columns if c not in drop and train_df[c].dtype.kind in "iuf"]
        self.feature_cols = feature_cols

        # сортируем по группе — обязательное требование LightGBM
        train_df = train_df.sort_values(group_col).reset_index(drop=True)
        groups_train = train_df.groupby(group_col, sort=False).size().values

        # float32 + контигуоус-копия → меньше памяти; free_raw_data=True отдаёт
        # массив в LightGBM, после построения гистограмм он освобождается.
        X_train = np.ascontiguousarray(train_df[feature_cols].values, dtype=np.float32)
        y_train = train_df[label_col].values.astype(np.float32, copy=False)
        d_train = lgb.Dataset(
            X_train,
            label=y_train,
            group=groups_train,
            feature_name=feature_cols,
            free_raw_data=True,
        )
        valid_sets = [d_train]
        valid_names = ["train"]
        if eval_df is not None:
            eval_df = eval_df.sort_values(eval_group_col).reset_index(drop=True)
            groups_eval = eval_df.groupby(eval_group_col, sort=False).size().values
            X_eval = np.ascontiguousarray(eval_df[feature_cols].values, dtype=np.float32)
            y_eval = eval_df[label_col].values.astype(np.float32, copy=False)
            d_eval = lgb.Dataset(
                X_eval,
                label=y_eval,
                group=groups_eval,
                feature_name=feature_cols,
                reference=d_train,
                free_raw_data=True,
            )
            valid_sets.append(d_eval)
            valid_names.append("val")
        # после построения Dataset исходные большие массивы можно отдать GC
        # (lgb.Dataset бинаризует фичи и больше не нуждается в float матрице).
        gc.collect()

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": list(eval_at),
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l2": self.reg_lambda,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "verbosity": -1,
            "seed": self.random_state,
        }
        callbacks = [
            lgb.log_evaluation(period=verbose) if verbose else lgb.log_evaluation(period=0),
            lgb.early_stopping(50, verbose=bool(verbose)),
        ] if eval_df is not None else [lgb.log_evaluation(period=verbose) if verbose else lgb.log_evaluation(period=0)]

        self.model = lgb.train(
            params,
            d_train,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None
        return self.model.predict(df[self.feature_cols].values, num_iteration=getattr(self.model, "best_iteration", None))

    def recommend_from_candidates(
        self,
        candidates: pd.DataFrame,
        k: int = 20,
        exclude: dict[int, set[int]] | None = None,
    ) -> dict[int, list[int]]:
        """Сортируем кандидатов по predicted score внутри каждого user_id."""
        cand = candidates.copy()
        cand["score"] = self.predict(cand)
        cand = cand.sort_values(["user_id", "score"], ascending=[True, False])
        out: dict[int, list[int]] = {}
        exclude = exclude or {}
        for uid, sub in cand.groupby("user_id", sort=False):
            seen = exclude.get(int(uid), set())
            items = [int(i) for i in sub["item_id"].tolist() if int(i) not in seen][:k]
            out[int(uid)] = items
        return out

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        assert self.model is not None
        imp = self.model.feature_importance(importance_type=importance_type)
        return pd.DataFrame({"feature": self.feature_cols, "importance": imp}).sort_values(
            "importance", ascending=False
        )

    # ------------------------------------------------------------------
    # Persist hooks
    # ------------------------------------------------------------------
    def _pickle_skip(self) -> set[str]:
        return {"model"}

    def _save_extra(self, dir: Path) -> None:
        if self.model is not None:
            self.model.save_model(str(dir / "model.txt"))

    def _load_extra(self, dir: Path) -> None:
        import lightgbm as lgb
        path = dir / "model.txt"
        self.model = lgb.Booster(model_file=str(path)) if path.exists() else None
