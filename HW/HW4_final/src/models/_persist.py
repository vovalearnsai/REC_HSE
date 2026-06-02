"""Unified save / load для всех моделей HW4.

Каждая модель кладётся в каталог
    HW4/models_store/<model_name>/<tag>/
где `tag` либо передаётся вручную, либо генерируется как `YYYYMMDD_HHMMSS`.

Внутри:
    payload.pkl   — основной dict с numpy-массивами / encoder-ами / гиперами
    model.pt      — (только torch) state_dict
    model.txt     — (только LightGBM) native dump
    meta.json     — человекочитаемое meta (имя, время, ключи payload)

API:
    >>> path = mdl.save(tag='ease_lam500')          # либо tag=None → timestamp
    >>> mdl2 = EASE.load(path)
    >>> # или из стора:
    >>> mdl2 = EASE.load_latest()                   # последний по mtime
"""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

# ──────────────────────────────────────────────────────────────────────────────
# Глобальный корень стора. Можно переопределить переменной окружения HW4_STORE.
# ──────────────────────────────────────────────────────────────────────────────
def _default_store() -> Path:
    import os
    env = os.environ.get("HW4_STORE")
    if env:
        return Path(env)
    # ../../models_store относительно файла (HW4/src/models/_persist.py → HW4/models_store)
    return Path(__file__).resolve().parents[2] / "models_store"


MODELS_STORE: Path = _default_store()


# ──────────────────────────────────────────────────────────────────────────────
# Mixin
# ──────────────────────────────────────────────────────────────────────────────
class Persistable:
    """Базовый mixin: save(path)/load(path) через pickle of self.__dict__.

    Модели с тяжёлыми/нестандартными артефактами (torch, lightgbm, implicit) —
    переопределяют `_save_extra(dir)` и `_load_extra(dir)`, плюс `_pickle_skip()`
    чтобы не сериализовать handle-объекты.
    """

    #: имя модели для подкаталога в сторе (если None — используется class name)
    persist_name: ClassVar[str | None] = None

    # ---------- helpers ----------
    @classmethod
    def _name(cls) -> str:
        return cls.persist_name or cls.__name__

    @classmethod
    def _model_root(cls) -> Path:
        return MODELS_STORE / cls._name()

    @classmethod
    def latest_path(cls) -> Path | None:
        root = cls._model_root()
        if not root.exists():
            return None
        dirs = [p for p in root.iterdir() if p.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda p: p.stat().st_mtime)

    # ---------- subclass hooks ----------
    def _pickle_skip(self) -> set[str]:
        """Имена атрибутов, которые НЕ кладём в payload.pkl
        (handle-объекты, тензоры на устройстве, тяжёлые матрицы, выгружаемые отдельно)."""
        return set()

    def _save_extra(self, dir: Path) -> None:
        """Сериализовать тяжёлые артефакты (torch state_dict, lgbm model.txt)."""

    def _load_extra(self, dir: Path) -> None:
        """Восстановить тяжёлые артефакты после установки payload."""

    # ---------- save ----------
    def save(self, tag: str | None = None, path: str | Path | None = None) -> Path:
        if path is not None:
            out = Path(path)
        else:
            tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self._model_root() / tag
        out.mkdir(parents=True, exist_ok=True)

        skip = self._pickle_skip()
        payload = {k: v for k, v in self.__dict__.items() if k not in skip}
        with open(out / "payload.pkl", "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._save_extra(out)

        meta = {
            "class": type(self).__name__,
            "name": getattr(self, "name", self._name()),
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "payload_keys": sorted(payload.keys()),
            "skipped_keys": sorted(skip),
        }
        with open(out / "meta.json", "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        print(f"[save] {type(self).__name__} → {out}")
        return out

    # ---------- load ----------
    @classmethod
    def load(cls, path: str | Path) -> "Persistable":
        p = Path(path)
        with open(p / "payload.pkl", "rb") as fh:
            payload: dict[str, Any] = pickle.load(fh)
        obj = cls.__new__(cls)        # без вызова __init__
        obj.__dict__.update(payload)
        obj._load_extra(p)
        print(f"[load] {cls.__name__} ← {p}")
        return obj

    @classmethod
    def load_latest(cls) -> "Persistable":
        p = cls.latest_path()
        if p is None:
            raise FileNotFoundError(f"нет сохранённых моделей {cls._name()} в {cls._model_root()}")
        return cls.load(p)
