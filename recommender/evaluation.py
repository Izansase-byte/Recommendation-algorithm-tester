from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import build_model


MODEL_NAMES = [
    "Modelo base",
    "K-NN usuario-usuario",
    "K-NN ítem-ítem",
    "Basado en contenido",
    "Modelo híbrido",
]


def train_test_split_by_user(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    seed: int = 42,
    min_user_ratings: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic holdout that always leaves at least two training ratings per user."""
    rng = np.random.default_rng(seed)
    test_indices: list[int] = []
    for _, group in ratings.groupby("userId", sort=True):
        if len(group) < min_user_ratings:
            continue
        n_test = min(max(1, int(round(len(group) * test_ratio))), len(group) - 2)
        test_indices.extend(rng.choice(group.index.to_numpy(), size=n_test, replace=False).tolist())
    test_mask = ratings.index.isin(test_indices)
    return ratings.loc[~test_mask].reset_index(drop=True), ratings.loc[test_mask].reset_index(drop=True)


def recommend_for_user(
    model,
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    n: int = 10,
    candidate_ids: list[int] | None = None,
    exclude_seen: bool = True,
    diversity_strength: float = 0.0,
    score_adjustments: dict[int, float] | None = None,
) -> pd.DataFrame:
    seen = set(ratings.loc[ratings["userId"] == user_id, "movieId"].astype(int))
    allowed = set(candidate_ids) if candidate_ids is not None else set(movies["movieId"].astype(int))
    candidates = [
        int(i)
        for i in movies["movieId"]
        if int(i) in allowed and (not exclude_seen or int(i) not in seen)
    ]
    if not candidates:
        return pd.DataFrame(columns=["movieId", "title", "genres", "predicción"])
    scores = model.predict(int(user_id), candidates)
    if score_adjustments:
        scores = np.clip(
            scores + np.array([float(score_adjustments.get(item, 0.0)) for item in candidates]),
            float(ratings["rating"].min()),
            float(ratings["rating"].max()),
        )
    if diversity_strength > 0 and len(candidates) > 1:
        strength = float(np.clip(diversity_strength, 0.0, 1.0))
        score_range = float(scores.max() - scores.min())
        normalized = (scores - scores.min()) / score_range if score_range > 1e-12 else np.ones(len(scores))
        candidate_movies = movies.set_index("movieId").loc[candidates]
        text = candidate_movies["genres"].fillna("").str.replace("|", " ", regex=False)
        features = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b").fit_transform(text)
        selected: list[int] = []
        remaining = set(range(len(candidates)))
        while remaining and len(selected) < n:
            if not selected:
                chosen = max(remaining, key=lambda idx: normalized[idx])
            else:
                remaining_list = list(remaining)
                similarity = cosine_similarity(features[remaining_list], features[selected]).max(axis=1)
                mmr = (1.0 - strength) * normalized[remaining_list] + strength * (1.0 - similarity)
                chosen = remaining_list[int(np.argmax(mmr))]
            selected.append(chosen)
            remaining.remove(chosen)
        top = np.asarray(selected, dtype=int)
    else:
        top = np.argsort(scores)[::-1][:n]
    ranked = pd.DataFrame({"movieId": np.array(candidates)[top], "predicción": scores[top]})
    ranked.insert(0, "posición", np.arange(1, len(ranked) + 1))
    return ranked.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")[
        ["posición", "movieId", "title", "genres", "predicción"]
    ]


def _catalog_content_similarity(movies: pd.DataFrame):
    text = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    features = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b").fit_transform(text)
    return features, {int(item): pos for pos, item in enumerate(movies["movieId"])}


def _ranking_metrics(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    top_k: int,
    relevance_threshold: float,
) -> dict[str, float]:
    all_items = set(movies["movieId"].astype(int))
    seen_by_user = train.groupby("userId")["movieId"].apply(lambda s: set(s.astype(int))).to_dict()
    test_by_user = {int(u): g for u, g in test.groupby("userId")}
    item_popularity = train["movieId"].value_counts().to_dict()
    n_users = max(1, train["userId"].nunique())
    content, item_pos = _catalog_content_similarity(movies)

    precisions, recalls, ndcgs, diversities, novelties = [], [], [], [], []
    recommended_catalog: set[int] = set()
    ranking_seconds = 0.0
    for user, user_test in test_by_user.items():
        relevant = set(user_test.loc[user_test["rating"] >= relevance_threshold, "movieId"].astype(int))
        candidates = sorted(all_items - seen_by_user.get(user, set()))
        if not candidates:
            continue
        started = perf_counter()
        scores = model.predict(user, candidates)
        ranking_seconds += perf_counter() - started
        order = np.argsort(scores)[::-1][:top_k]
        recommended = [candidates[i] for i in order]
        recommended_catalog.update(recommended)

        if relevant:
            hits = [1 if item in relevant else 0 for item in recommended]
            precisions.append(sum(hits) / top_k)
            recalls.append(sum(hits) / len(relevant))
            dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal_hits = min(len(relevant), top_k)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
            ndcgs.append(dcg / idcg if idcg else 0.0)

        positions = [item_pos[item] for item in recommended if item in item_pos]
        if len(positions) > 1:
            sim = cosine_similarity(content[positions])
            upper = sim[np.triu_indices(len(positions), k=1)]
            diversities.append(float(1.0 - upper.mean()))
        item_novelty = [
            -np.log2(max(item_popularity.get(item, 0) / n_users, 1.0 / (n_users + 1)))
            for item in recommended
        ]
        novelties.append(float(np.mean(item_novelty)))

    return {
        f"Precisión@{top_k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{top_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "Cobertura": len(recommended_catalog) / max(1, len(all_items)),
        "Diversidad": float(np.mean(diversities)) if diversities else 0.0,
        "Novedad (bits)": float(np.mean(novelties)) if novelties else 0.0,
        "Tiempo recomendación (s)": ranking_seconds,
    }


def evaluate_models(
    model_names: list[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    hybrid_params: dict | None = None,
    top_k: int = 10,
    relevance_threshold: float = 4.0,
    model_params: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit every model on one split and return comparable rating/ranking metrics."""
    results: list[dict[str, float | str]] = []
    fitted: dict[str, object] = {}
    grouped_test = {int(u): g for u, g in test.groupby("userId")}
    model_params = model_params or {}

    for model_name in model_names:
        params = model_params.get(model_name)
        if params is None and model_name == "Modelo híbrido":
            params = hybrid_params
        model = build_model(model_name, params)
        started = perf_counter()
        model.fit(train, movies)
        fit_seconds = perf_counter() - started

        truths: list[float] = []
        predictions: list[float] = []
        started = perf_counter()
        for user, group in grouped_test.items():
            item_ids = group["movieId"].astype(int).tolist()
            predictions.extend(model.predict(user, item_ids).tolist())
            truths.extend(group["rating"].astype(float).tolist())
        rating_seconds = perf_counter() - started
        truth = np.asarray(truths)
        pred = np.asarray(predictions)
        errors = pred - truth
        row: dict[str, float | str] = {
            "Algoritmo": model_name,
            "RMSE": float(np.sqrt(np.mean(errors**2))),
            "MAE": float(np.mean(np.abs(errors))),
            "Tiempo entrenamiento (s)": fit_seconds,
            "Tiempo predicción (s)": rating_seconds,
            "Predicciones/s": len(pred) / max(rating_seconds, 1e-9),
        }
        row.update(_ranking_metrics(model, train, test, movies, top_k, relevance_threshold))
        results.append(row)
        fitted[model_name] = model

    return pd.DataFrame(results), fitted


def optimize_hybrid_weights(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    movies: pd.DataFrame,
    model_params: dict[str, dict] | None = None,
    trials: int = 80,
    objective: str = "RMSE",
    seed: int = 42,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune hybrid weights efficiently by reusing the four component predictions."""
    model_params = model_params or {}
    components = {
        "base": "Modelo base",
        "user": "K-NN usuario-usuario",
        "item": "K-NN ítem-ítem",
        "content": "Basado en contenido",
    }
    truths: list[float] = []
    predictions: dict[str, list[float]] = {key: [] for key in components}
    grouped = {int(user): group for user, group in validation.groupby("userId")}
    for key, model_name in components.items():
        model = build_model(model_name, model_params.get(model_name)).fit(train, movies)
        for user, group in grouped.items():
            predictions[key].extend(model.predict(user, group["movieId"].astype(int).tolist()).tolist())
        if not truths:
            for _, group in grouped.items():
                truths.extend(group["rating"].astype(float).tolist())

    truth = np.asarray(truths)
    matrix = np.column_stack([predictions[key] for key in components])
    rng = np.random.default_rng(seed)
    candidates = [
        np.full(4, 0.25),
        np.array([0.10, 0.30, 0.30, 0.30]),
        *np.eye(4),
        *rng.dirichlet(np.ones(4), size=max(1, trials)),
    ]
    rows: list[dict[str, float]] = []
    for values in candidates:
        pred = matrix @ values
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        mae = float(np.mean(np.abs(pred - truth)))
        rows.append(
            {
                "base": float(values[0]),
                "user": float(values[1]),
                "item": float(values[2]),
                "content": float(values[3]),
                "RMSE": rmse,
                "MAE": mae,
            }
        )
    results = pd.DataFrame(rows).sort_values(objective, ascending=True).reset_index(drop=True)
    best = results.iloc[0]
    return {key: float(best[key]) for key in components}, results
