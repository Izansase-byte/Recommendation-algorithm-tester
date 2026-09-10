from __future__ import annotations

import numpy as np
import pandas as pd


def movie_statistics(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Return one analysis-ready row per movie, including zero-rating titles."""
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(votos="size", media="mean", mediana="median", desviación="std", mínimo="min", máximo="max")
        .reset_index()
    )
    result = movies[["movieId", "title", "genres"]].merge(stats, on="movieId", how="left")
    result["año"] = result["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("Float64")
    result["votos"] = result["votos"].fillna(0).astype(int)
    return result


def user_statistics(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    stats = (
        ratings.groupby("userId")["rating"]
        .agg(valoraciones="size", media="mean", mediana="median", desviación="std", mínima="min", máxima="max")
        .reset_index()
    )
    genre_rows = ratings.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    genre_rows = genre_rows.assign(género=genre_rows["genres"].fillna("").str.split("|")).explode("género")
    favorite = (
        genre_rows.groupby(["userId", "género"])
        .agg(n=("rating", "size"), nota=("rating", "mean"))
        .reset_index()
        .sort_values(["userId", "nota", "n"], ascending=[True, False, False])
        .drop_duplicates("userId")
        .rename(columns={"género": "género_favorito"})[["userId", "género_favorito"]]
    )
    return stats.merge(favorite, on="userId", how="left")


def genre_statistics(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    expanded = movies[["movieId", "genres"]].copy()
    expanded["género"] = expanded["genres"].fillna("Sin género").str.split("|")
    expanded = expanded.explode("género")
    movie_counts = expanded.groupby("género")["movieId"].nunique().rename("películas")
    rated = expanded.merge(ratings, on="movieId", how="left")
    rating_stats = rated.groupby("género").agg(
        valoraciones=("rating", "count"),
        usuarios=("userId", "nunique"),
        media=("rating", "mean"),
        desviación=("rating", "std"),
    )
    return movie_counts.to_frame().join(rating_stats).reset_index().sort_values("valoraciones", ascending=False)


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[values >= 0]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    ordered = np.sort(values)
    n = len(ordered)
    return float((2 * np.sum(np.arange(1, n + 1) * ordered) / (n * ordered.sum())) - (n + 1) / n)


def data_quality_report(movies: pd.DataFrame, ratings: pd.DataFrame) -> dict[str, float | int | str]:
    movie_count = int(movies["movieId"].nunique())
    user_count = int(ratings["userId"].nunique())
    rating_count = int(len(ratings))
    item_counts = ratings["movieId"].value_counts()
    user_counts = ratings["userId"].value_counts()
    missing = int(movies[["title", "genres"]].isna().sum().sum() + ratings[["userId", "movieId", "rating"]].isna().sum().sum())
    duplicates = int(ratings.duplicated(["userId", "movieId"]).sum())
    report: dict[str, float | int | str] = {
        "Películas": movie_count,
        "Usuarios": user_count,
        "Valoraciones": rating_count,
        "Densidad (%)": 100 * rating_count / max(1, movie_count * user_count),
        "Esparsidad (%)": 100 * (1 - rating_count / max(1, movie_count * user_count)),
        "Valores ausentes": missing,
        "Pares duplicados": duplicates,
        "Usuarios con < 5 votos": int((user_counts < 5).sum()),
        "Películas con < 5 votos": int((item_counts < 5).sum()),
        "Películas sin votos": int(movie_count - item_counts.index.nunique()),
        "Votos medios por usuario": float(user_counts.mean()),
        "Votos medios por película": float(item_counts.mean()),
        "Desigualdad de popularidad (Gini)": _gini(item_counts.to_numpy()),
        "Nota mínima": float(ratings["rating"].min()),
        "Nota máxima": float(ratings["rating"].max()),
        "Nota media": float(ratings["rating"].mean()),
    }
    if "timestamp" in ratings and ratings["timestamp"].notna().any():
        dates = pd.to_datetime(ratings["timestamp"], unit="s", errors="coerce")
        if dates.notna().any():
            report["Primera valoración"] = str(dates.min().date())
            report["Última valoración"] = str(dates.max().date())
    return report


def popularity_curve(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Catalogue popularity and cumulative interaction share for long-tail analysis."""
    counts = ratings["movieId"].value_counts().rename("votos")
    result = movies[["movieId", "title"]].merge(counts, left_on="movieId", right_index=True, how="left")
    result["votos"] = result["votos"].fillna(0).astype(int)
    result = result.sort_values("votos", ascending=False).reset_index(drop=True)
    result["posición"] = np.arange(1, len(result) + 1)
    total = max(1, result["votos"].sum())
    result["porcentaje_acumulado"] = 100 * result["votos"].cumsum() / total
    return result
