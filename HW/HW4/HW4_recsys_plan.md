# HW4 — План построения рекомендательной системы

**Цель:** топ-20 рекомендаций для 185 282 пользователей из `test_users.csv`.
**Главная метрика:** `NDCG@20` (с relevance = `is_purchased`, бинарный).
**Вспомогательные метрики (для дебага):** `Recall@20`, `MAP@20`, `HitRate@20`, `Coverage@20`, `Novelty`.

---

## 0. Ключевые выводы из EDA, влияющие на дизайн

| Факт | Следствие для архитектуры |
|---|---|
| 11.4M interactions, sparsity 99.9% | Подходят MF / two-tower / sequence; full-matrix модели не нужны |
| Top 1.5% товаров = 50% интеракций (Pareto) | Обязательны popularity-baseline и popularity-debiasing |
| `is_purchased` = 39% (баланс относительно норм для recsys) | Это естественный positive label. `rating > 0` ⇔ `is_purchased=True` → rating не даёт новой информации о «купит/нет», но даёт **силу** позитива |
| `impressions` (слейт из 20) присутствует у каждой строки, `item_id ∈ impressions` всегда | **Бесценный источник негативов**: 19 «показано, не выбрано» на каждую позитивную строку — гораздо качественнее random negatives |
| 0 cold-start users в test | Можно строить user embeddings; fallback по холодным **не нужен** для test, но полезен для робастности |
| 3 102 cold items в каталоге | Контент-фичи (`category_tags`/`series_id`/`author_ids`) обязательны для покрытия каталога; в retrieval это не критично (их в train нет → не предложим), но в reranker и для honest CV — да |
| Период 939 дней, чёткая суточная/недельная цикличность, два аномальных пика (янв 2015, янв 2016) | Time-based split обязателен; временные фичи (recency, hour, dow) — в reranker |
| `category_tags`: 189K unique (огромный словарь) | Использовать как мульти-hot/embedding bag, не one-hot; для tree-based — топ-K + остальное в "other" + count-фичи |
| `series_id`/`author_ids` малочастотны | Хороши как «collaborative-content» сигнал (item-item similarity внутри серии/автора) |

---

## 1. Разбиение данных: train / val / test (offline)

**Стратегия: time-based global split.** Random split в recsys приводит к лику будущего → завышает метрики.

```
[========= TRAIN =========][== VAL ==][= HOLDOUT =]
2014-05-15            ~2016-10    ~2016-11   2016-12-10
```

| Сплит | Период (≈) | Доля | Назначение |
|---|---|---|---|
| `train_fit` | до Q_85 по времени | ~85% | Обучение всех моделей и построение фичей |
| `val` | Q_85 → Q_95 | ~10% | Подбор гиперпараметров, early stopping, выбор кандидатов |
| `holdout` | Q_95 → конец | ~5% | Финальная оценка моделей перед сабмитом |

**Правила:**
- Сплитим по `timestamp` глобально, **не** per-user (иначе подсматриваем будущее одних юзеров через взаимодействия других).
- Для каждого пользователя в `val`/`holdout` ground-truth = множество **купленных** товаров (`is_purchased=True`) в его периоде.
- Из ground-truth исключаем товары, уже купленные в `train_fit` (избегаем тривиальных рекомендаций).
- Пользователи без покупок в val/holdout — выкидываются из оценки (но участвуют в обучении).
- Финальный submit обучаем на **всех** данных (`train_fit + val + holdout`) с лучшими гиперпараметрами.

**Нужна ли валидация отдельно от holdout?** Да, потому что мы будем тюнить гиперы (LightFM regularization, ALS factors, learning rate, кол-во кандидатов в reranker и т.д.). Без отдельного `holdout` мы рискуем переобучиться на `val`.

---

## 2. Формирование данных: Datamart

Чтобы фичи переиспользовались всеми моделями и не текли из будущего, организуем как 3 уровня (paradigma из лекции 4 — `images/datamart_*.png`).

### 2.1. RAW layer (`datamart/raw/`)
Парсинг и нормализация исходников:
- `interactions.parquet` — train, partitioned by `dt=YYYY-MM` (быстрая фильтрация по времени).
- `items.parquet` — каталог с распарсенными списками.
- `impressions_long.parquet` — explode `impressions` → `(user_id, item_id, shown_ts, was_clicked, was_purchased)`. Это база для CTR/impression-фич и для negative sampling.

### 2.2. AGG layer (`datamart/agg/`) — агрегаты «как было ДО момента T»
Все агрегаты считаются **только по `train_fit`** (cutoff = граница train/val), чтобы фичи не текли. Для финального обучения пересчитаем с другим cutoff.

| Файл | Ключ | Содержит |
|---|---|---|
| `user_stats.parquet` | `user_id` | n_interactions, n_purchases, purchase_rate, n_unique_items, mean_rating, recency_days, активность по hour/dow, любимые top-3 category_tags |
| `item_stats.parquet` | `item_id` | n_interactions, n_purchases, CTR, popularity_rank, n_unique_users, mean_rating, recency_days, age_in_catalog |
| `item_content.parquet` | `item_id` | one-row representations: top-K tags (multi-hot), series, author + embedding-bag id-lists |
| `user_item_pair.parquet` | `(user_id, item_id)` | n_interactions, n_purchases, last_seen, was_in_impression_count (для reranker) |
| `cooccurrence.parquet` | `(item_a, item_b)` | счётчики совместных покупок (для item2item baseline) |

### 2.3. FEAT layer (`datamart/feat/`) — финальные feature-таблицы
- `cand_features.parquet` — `(user_id, item_id, candidate_score_from_each_retriever, user_feats..., item_feats..., pair_feats...)`. Это вход в reranker.
- На уровне retrieval кэшируем embeddings: `user_emb.npy`, `item_emb.npy` для каждой модели.

### 2.4. Структура кода
```
HW4/
├── HSE_RecSys_HW4.ipynb        # главный ноутбук (EDA + orchestration)
├── dataset/              # исходные данные
├── datamart/
│   ├── raw/
│   ├── agg/
│   └── feat/
├── src/
│   ├── data.py                 # загрузка, time-split
│   ├── datamart.py             # построение всех слоёв
│   ├── metrics.py              # ndcg@k, recall@k, map@k (батчево)
│   ├── eval.py                 # eval loop: модель → рекомендации → метрики
│   ├── negatives.py            # negative sampling из impressions
│   └── models/
│       ├── baselines.py        # TopPop, UserPop, RecentPop
│       ├── item2item.py        # co-occurrence / cosine
│       ├── als.py              # implicit ALS
│       ├── lightfm.py          # WARP + content features
│       └── ranker.py           # LightGBM ranker
└── submission.csv
```

---

Вот обновленная версия разделов 3–6. Я внедрил **EASE** как флагманский совместный (collaborative) алгоритм, заменил неповоротливый LightFM на **Two-Tower (DSSM)** для качественной работы с контентом и холодными товарами, а также обновил логику группировки в **LightGBM**, чтобы по полной задействовать силу слейтов (`impressions`).

---

## 3. Implicit vs explicit feedback — стратегия использования сигналов

В наших данных есть **4 сигнала**: `impressions`, click (interaction без покупки), `is_purchased`, `rating`.

```
показан (impression) ⊃ кликнут (interaction) ⊃ куплен (purchased) ⊃ оценён (rated>0)
                                                                     ↳ значение rating ∈ {1..5}
```

**Стратегия по уровням модели:**

| Этап | Positive | Negative | Confidence/weight | Зачем |
|---|---|---|---|---|
| **Baselines** | `is_purchased=True` | — | — | Валидация пайплайна на простейших эвристиках. |
| **ALS / EASE** | Все взаимодействия | — (ALS через регуляризацию, EASE через 0 на диагонали) | Для ALS: `α·is_purchased + β·log(1+rating)`. Для EASE: бинарная матрица взаимодействия. | Используем всю матрицу (даже клики без покупки), чтобы выучить графы связей (collaborative). |
| **Two-Tower (DSSM)** | `is_purchased=True` | in-batch negatives + сэмплирование из `impressions` | Опционально weighted loss от рейтинга | Ввод в оборот холодных `item_features` и генерация контентных кандидатов. |
| **Reranker (LGBM)** | `is_purchased=True` | строки слейта `impressions`, где user **не купил** | label ∈ {0, 1}. **Group: `impression_id` (train), `user_id` (test/val)**. Опц. graded: 0..5 | Honest hard-negatives. Модель учится **ранжировать конкретный слейт** во время обучения! |

**Принципы:**
- `rating` **никогда не target классификации**, потому что 64% значений = 0 = «не оценил» (NMAR). Используем его только как confidence booster / graded weight.
- `impressions` — бесценный источник контрфактуалов. Пользователь видел эти 19 товаров и *не выбрал*. Это ядро для Reranker-а.
- **Группировка в Ranker:** При обучении мы группируем по `impression_id` (уникальный ID события в train) — так YetiRank/LambdaMART учится поднимать позитивы наверх *внутри реального показа*. На инференсе мы просто скорим пул кандидатов и сортируем по `user_id`.

---

## 4. Использование item features (`category_tags`, `series_id`, `author_ids`)

| Где | Как используем |
|---|---|
| **ALS / EASE** | Не умеют использовать фичи. Поддерживают только 31,221 «тёплых» товаров. Холодные товары проходят мимо на этом этапе. |
| **PyTorch Two-Tower (DSSM)** | Item Tower: `Linear(embed(item_id) + mean_pool(embed(tags)) + embed(series) + mean_pool(embed(authors)))`. Это наше **основное решение для 3 102 холодных товаров**! Если `item_id` холодный (Out-of-Vocabulary), фичи всё равно дадут отличный вектор. |
| **Reranker (LGBM)** | Мульти-hot по топ-100 тегов/авторов. Плюс Target Encoding (сглаженный CTR/Purchase Rate категории/автора). Добавляем Jaccard similarity: `len(user_fav_tags ∩ item_tags) / len(user_fav_tags ∪ item_tags)`. |

**Cold-item покрытие:** В отличие от старого плана, нам больше не нужны "костыли" с поиском соседей post-hoc. Нейросетевая Item Tower сгенерирует полноценные эмбеддинги для всех 34 323 товаров из каталога, включая непроданные.

---

## 5. Список моделей (по нарастающей сложности)

Формируем двухуровневый пайплайн (Retrieval → Ranking). Каждая Retrieval-модель генерирует кандидатов, которые оцениваются на `Reccal@100+`. Ранжировщик максимизирует `NDCG@20`.

### M0. Baselines (sanity floor)
1. **GlobalTopPop** — top-20 самых покупаемых товаров.
2. **UserHistoryTopPop** — top-20 популярных из любимых категорий юзера.
*Ожидаемая NDCG@20: 0.02–0.05.*

### M1. Collaborative Retrieval (Главные генераторы)
3. **ALS (implicit)** — `implicit.als.AlternatingLeastSquares(factors=128)`. С confidence-матрицей покупок и рейтингов.
4. **EASE (Embarrassingly Shallow Autoencoders)** — **must-have**. L2-регуляризованный линейный автоэнкодер. Быстро решается через `scipy.linalg.inv` (матрица 31k x 31k считается считанные минуты). Разрывает ALS метриками Recall на разреженных графах.
*Ожидаемая NDCG@20 (на прямой выдаче): 0.12–0.18.*

### M2. Content Retrieval (Покрытие холодных)
5. **Two-Tower DSSM (PyTorch)** — две нейросети: одна сжимает историю юзера (`mean_pool` его прошлых покупок + `user_id` + `recency`), другая — фичи товара. Максимизирует косинусное расстояние между ними (InfoNCE loss). Возвращает в пулы рекомендаций 3 102 cold items.
*Ожидаемая NDCG@20: 0.10–0.15 (но добавит огромное разнообразие/coverage!).*

### M3. Two-stage: Reranker (Ранжирование топа — The King)
6. **LightGBM Ranker (LambdaMART / YetiRank)**:
   - **Кандидаты:** `union_top_200 / user` = (Top-100 EASE + Top-100 ALS + Top-100 Two-Tower + Top-20 Pop). Дедуплицируем.
   - **Фичи (для join-а на cand_features.parquet):**
     - Retrieval-фичи: ранги и скоры от ALS, EASE, Two-Tower.
     - Статистики: CTR айтема, Recency последней покупки, frequency юзера.
     - Pairwise: Jaccard по тегам, сколько раз товар уже фигурировал в слейтах юзера.
     - Время: `hour`, `day_of_week`.
   - **Label & Group (Обучение):** На вход идут все строки из train (распарсенные `impressions`). 1 слейт = 1 группа (`impression_id`). Таргет — `is_purchased` (остальные 19 — жесткие негативы = 0).
   - **Label & Group (Инференс):** Подаем 200 кандидатов. Слейтов уже нет, поэтому Group = `user_id`. Модель просто возвращает score для сортировки.
*Ожидаемая NDCG@20: 0.20–0.27+ (Основной буст метрик).*

### M4. Опционально (Sequence SOTA)
7. **SASRec (Self-Attentive Sequential Recommendations)** — мощный нейросетевой трансформер поверх `timestamp` последовательностей. Если есть GPU и время — ставим в этап Retrieval вместо (или в помощь) ALS.

---

## 6. Финальный пайплайн submission

```python
# Псевдокод финального пайплайна для test_users.csv

# 1. Генерируем массив словарей с кандидатами от 1-го этапа
candidates = []
for user in test_users:
    cands = union(
        EASE.get_top(user, k=100),           # Основной recall
        ALS.get_top(user, k=100),            # Подстраховка
        DSSM_TwoTower.get_top(user, k=50),   # Для холодных / контентных айтемов
        TopPop.get_top(k=20)                 # Fallback
    )
    candidates.append(deduplicate(cands)) # Итого ~200 уникальных айтемов на юзера

# 2. Формируем плоский датафрейм (inference_df): user_id, item_id
inference_df = create_dataframe(candidates)

# 3. Джойним фичи (Datamart FEAT layer)
inference_df = join_features(inference_df, user_stats, item_stats, item_content)

# 4. Ранжируем с помощью обученного LightGBM 
inference_df['rank_score'] = LGBMRanker.predict(inference_df[features_list])

# 5. Оставляем Топ-20 и удаляем "уже купленное"
submission = (
    inference_df
    .groupby('user_id')
    .apply(filter_already_purchased) # Важно: убираем то, что уже было в train у юзера
    .sort_values(by='rank_score', ascending=False)
    .head(20)
    [['user_id', 'item_id']]
)

submission.to_csv('submission.csv', index=False)
```

## 7. Best practices, которые соблюдаем

- ✅ **Time-based split** (не random).
- ✅ **Никакой утечки**: все фичи на момент T считаются только по данным `< T`.
- ✅ **Negative sampling из impressions** вместо random.
- ✅ **Honest cold-item handling** через item content features.
- ✅ **Filter-already-seen** при инференсе (не рекомендуем уже купленное).
- ✅ **Recall@k > NDCG@k на retrieval stage** — на этом этапе важно поднять recall кандидатов до 0.5+, NDCG растягивает ranker.
- ✅ **Реproducibility**: `random_state=42`, версии моделей в `results.md`.
- ✅ **Чистый код**: фичи и метрики — отдельные модули (`src/`), ноутбук только оркестрирует.
- ✅ **Batched eval**: считаем NDCG@20 на сэмпле 10–20K юзеров для итерации, на полном `val` — для финальных цифр.

---

