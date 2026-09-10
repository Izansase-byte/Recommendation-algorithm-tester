from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _clip(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), *bounds)


@dataclass
class BaselineRecommender:
    regularization: float = 10.0
    iterations: int = 10
    name: str = "Modelo base"

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        self.global_mean = float(ratings["rating"].mean())
        self.bounds = (float(ratings["rating"].min()), float(ratings["rating"].max()))
        self.user_bias = {int(u): 0.0 for u in ratings["userId"].unique()}
        self.item_bias = {int(i): 0.0 for i in movies["movieId"].unique()}
        by_user = {int(u): g for u, g in ratings.groupby("userId")}
        by_item = {int(i): g for i, g in ratings.groupby("movieId")}
        for _ in range(self.iterations):
            for user, group in by_user.items():
                residual = group["rating"].to_numpy() - self.global_mean - np.array(
                    [self.item_bias.get(int(i), 0.0) for i in group["movieId"]]
                )
                self.user_bias[user] = float(residual.sum() / (self.regularization + len(group)))
            for item, group in by_item.items():
                residual = group["rating"].to_numpy() - self.global_mean - np.array(
                    [self.user_bias.get(int(u), 0.0) for u in group["userId"]]
                )
                self.item_bias[item] = float(residual.sum() / (self.regularization + len(group)))
        return self

    def predict(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        items = list(item_ids)
        values = self.global_mean + self.user_bias.get(int(user_id), 0.0) + np.array(
            [self.item_bias.get(int(item), 0.0) for item in items]
        )
        return _clip(values, self.bounds)


@dataclass
class UserKNNRecommender:
    k: int = 25
    min_similarity: float = 0.0
    shrinkage: float = 10.0
    baseline_regularization: float = 10.0
    baseline_iterations: int = 10
    name: str = "K-NN usuario-usuario"

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings, self.movies = ratings.copy(), movies.copy()
        self.baseline = BaselineRecommender(self.baseline_regularization, self.baseline_iterations).fit(ratings, movies)
        pivot = ratings.pivot(index="userId", columns="movieId", values="rating")
        self.user_ids = pivot.index.to_numpy()
        self.item_ids = pivot.columns.to_numpy()
        self.user_pos = {int(v): i for i, v in enumerate(self.user_ids)}
        self.item_pos = {int(v): i for i, v in enumerate(self.item_ids)}
        self.values = pivot.to_numpy(dtype=float)
        self.means = np.nanmean(self.values, axis=1)
        centered = np.nan_to_num(self.values - self.means[:, None], nan=0.0)
        raw_sim = cosine_similarity(centered)
        observed = ~np.isnan(self.values)
        overlap = observed.astype(float) @ observed.astype(float).T
        self.similarity = raw_sim * overlap / (overlap + self.shrinkage)
        np.fill_diagonal(self.similarity, 0.0)
        return self

    def predict(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        items = list(item_ids)
        fallback = self.baseline.predict(user_id, items)
        if int(user_id) not in self.user_pos:
            return fallback
        u = self.user_pos[int(user_id)]
        output = fallback.copy()
        for n, item in enumerate(items):
            pos = self.item_pos.get(int(item))
            if pos is None:
                continue
            raters = np.where(~np.isnan(self.values[:, pos]))[0]
            sims = self.similarity[u, raters]
            valid = sims > self.min_similarity
            raters, sims = raters[valid], sims[valid]
            if len(sims) == 0:
                continue
            order = np.argsort(sims)[-self.k :]
            raters, sims = raters[order], sims[order]
            deviations = self.values[raters, pos] - self.means[raters]
            denom = np.abs(sims).sum()
            if denom > 1e-12:
                output[n] = self.means[u] + float(sims @ deviations / denom)
        return _clip(output, self.baseline.bounds)


@dataclass
class ItemKNNRecommender:
    k: int = 25
    min_similarity: float = 0.0
    shrinkage: float = 10.0
    baseline_regularization: float = 10.0
    baseline_iterations: int = 10
    name: str = "K-NN ítem-ítem"

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings, self.movies = ratings.copy(), movies.copy()
        self.baseline = BaselineRecommender(self.baseline_regularization, self.baseline_iterations).fit(ratings, movies)
        pivot = ratings.pivot(index="userId", columns="movieId", values="rating")
        self.user_ids, self.item_ids = pivot.index.to_numpy(), pivot.columns.to_numpy()
        self.user_pos = {int(v): i for i, v in enumerate(self.user_ids)}
        self.item_pos = {int(v): i for i, v in enumerate(self.item_ids)}
        self.values = pivot.to_numpy(dtype=float)
        user_means = np.nanmean(self.values, axis=1)
        centered = np.nan_to_num(self.values - user_means[:, None], nan=0.0).T
        raw_sim = cosine_similarity(centered)
        observed = (~np.isnan(self.values)).astype(float).T
        overlap = observed @ observed.T
        self.similarity = raw_sim * overlap / (overlap + self.shrinkage)
        np.fill_diagonal(self.similarity, 0.0)
        return self

    def predict(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        items = list(item_ids)
        fallback = self.baseline.predict(user_id, items)
        u = self.user_pos.get(int(user_id))
        if u is None:
            return fallback
        rated_pos = np.where(~np.isnan(self.values[u]))[0]
        rated_values = self.values[u, rated_pos]
        rated_ids = self.item_ids[rated_pos]
        baseline_rated = self.baseline.predict(user_id, rated_ids)
        deviations = rated_values - baseline_rated
        output = fallback.copy()
        for n, item in enumerate(items):
            pos = self.item_pos.get(int(item))
            if pos is None:
                continue
            sims = self.similarity[pos, rated_pos]
            valid = sims > self.min_similarity
            sims_valid, dev_valid = sims[valid], deviations[valid]
            if len(sims_valid) == 0:
                continue
            order = np.argsort(sims_valid)[-self.k :]
            sims_valid, dev_valid = sims_valid[order], dev_valid[order]
            denom = np.abs(sims_valid).sum()
            if denom > 1e-12:
                output[n] = fallback[n] + float(sims_valid @ dev_valid / denom)
        return _clip(output, self.baseline.bounds)


@dataclass
class ContentBasedRecommender:
    use_title: bool = False
    genre_weight: int = 3
    baseline_regularization: float = 10.0
    baseline_iterations: int = 10
    name: str = "Basado en contenido"

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings, self.movies = ratings.copy(), movies.copy()
        self.baseline = BaselineRecommender(self.baseline_regularization, self.baseline_iterations).fit(ratings, movies)
        genres = movies["genres"].fillna("").str.replace("|", " ", regex=False)
        text = ((genres + " ") * max(1, int(self.genre_weight))).str.strip()
        if self.use_title:
            clean_title = movies["title"].fillna("").str.replace(r"\(\d{4}\)", "", regex=True)
            text = text + " " + clean_title
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
        self.features = self.vectorizer.fit_transform(text)
        self.item_pos = {int(v): i for i, v in enumerate(movies["movieId"].to_numpy())}
        self.user_history = {
            int(user): group[["movieId", "rating"]].copy() for user, group in ratings.groupby("userId")
        }
        return self

    def predict(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        items = list(item_ids)
        fallback = self.baseline.predict(user_id, items)
        history = self.user_history.get(int(user_id))
        if history is None or history.empty:
            return fallback
        valid_history = history[history["movieId"].isin(self.item_pos)]
        if valid_history.empty:
            return fallback
        hist_ids = valid_history["movieId"].astype(int).tolist()
        hist_pos = [self.item_pos[i] for i in hist_ids]
        hist_ratings = valid_history["rating"].to_numpy(dtype=float)
        hist_base = self.baseline.predict(user_id, hist_ids)
        deviations = hist_ratings - hist_base
        output = fallback.copy()
        for n, item in enumerate(items):
            pos = self.item_pos.get(int(item))
            if pos is None:
                continue
            sims = cosine_similarity(self.features[pos], self.features[hist_pos]).ravel()
            positive = sims > 0
            if positive.any():
                output[n] = fallback[n] + float(sims[positive] @ deviations[positive] / sims[positive].sum())
        return _clip(output, self.baseline.bounds)

    def item_similarity(self, item_ids: Iterable[int]) -> np.ndarray:
        positions = [self.item_pos[int(i)] for i in item_ids if int(i) in self.item_pos]
        return cosine_similarity(self.features[positions]) if positions else np.empty((0, 0))


@dataclass
class HybridRecommender:
    weights: dict[str, float] = field(
        default_factory=lambda: {"base": 0.10, "user": 0.30, "item": 0.30, "content": 0.30}
    )
    user_k: int = 25
    item_k: int = 25
    min_similarity: float = 0.0
    shrinkage: float = 10.0
    user_min_similarity: float | None = None
    item_min_similarity: float | None = None
    user_shrinkage: float | None = None
    item_shrinkage: float | None = None
    content_use_title: bool = False
    content_genre_weight: int = 3
    baseline_regularization: float = 10.0
    baseline_iterations: int = 10
    name: str = "Modelo híbrido"

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.components = {
            "base": BaselineRecommender(self.baseline_regularization, self.baseline_iterations),
            "user": UserKNNRecommender(
                k=self.user_k,
                min_similarity=self.min_similarity if self.user_min_similarity is None else self.user_min_similarity,
                shrinkage=self.shrinkage if self.user_shrinkage is None else self.user_shrinkage,
                baseline_regularization=self.baseline_regularization,
                baseline_iterations=self.baseline_iterations,
            ),
            "item": ItemKNNRecommender(
                k=self.item_k,
                min_similarity=self.min_similarity if self.item_min_similarity is None else self.item_min_similarity,
                shrinkage=self.shrinkage if self.item_shrinkage is None else self.item_shrinkage,
                baseline_regularization=self.baseline_regularization,
                baseline_iterations=self.baseline_iterations,
            ),
            "content": ContentBasedRecommender(
                use_title=self.content_use_title,
                genre_weight=self.content_genre_weight,
                baseline_regularization=self.baseline_regularization,
                baseline_iterations=self.baseline_iterations,
            ),
        }
        for model in self.components.values():
            model.fit(ratings, movies)
        return self

    def predict(self, user_id: int, item_ids: Iterable[int]) -> np.ndarray:
        items = list(item_ids)
        active = {key: max(0.0, float(value)) for key, value in self.weights.items() if key in self.components}
        total = sum(active.values())
        if total <= 0:
            active = {"base": 1.0}
            total = 1.0
        result = np.zeros(len(items), dtype=float)
        for key, weight in active.items():
            result += (weight / total) * self.components[key].predict(user_id, items)
        return result

    def component_predictions(self, user_id: int, item_ids: Iterable[int]) -> dict[str, np.ndarray]:
        """Return each component score for transparent hybrid explanations."""
        items = list(item_ids)
        return {key: model.predict(user_id, items) for key, model in self.components.items()}


def build_model(name: str, params: dict | None = None):
    aliases = {
        "Modelo base": BaselineRecommender,
        "K-NN usuario-usuario": UserKNNRecommender,
        "K-NN ítem-ítem": ItemKNNRecommender,
        "Basado en contenido": ContentBasedRecommender,
    }
    if name == "Modelo híbrido":
        return HybridRecommender(**(params or {}))
    if name not in aliases:
        raise ValueError(f"Modelo desconocido: {name}")
    return aliases[name](**(params or {}))
