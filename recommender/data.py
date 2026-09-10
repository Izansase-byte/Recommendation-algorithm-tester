from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


MOVIES = [
    (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
    (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
    (3, "Heat (1995)", "Action|Crime|Thriller"),
    (4, "Sabrina (1995)", "Comedy|Romance"),
    (5, "GoldenEye (1995)", "Action|Adventure|Thriller"),
    (6, "Sense and Sensibility (1995)", "Drama|Romance"),
    (7, "Seven (1995)", "Mystery|Thriller"),
    (8, "Usual Suspects, The (1995)", "Crime|Mystery|Thriller"),
    (9, "Apollo 13 (1995)", "Adventure|Drama"),
    (10, "Babe (1995)", "Children|Drama"),
    (11, "Dead Man Walking (1995)", "Crime|Drama"),
    (12, "Clueless (1995)", "Comedy|Romance"),
    (13, "Braveheart (1995)", "Action|Drama|War"),
    (14, "Taxi Driver (1976)", "Crime|Drama|Thriller"),
    (15, "Star Wars: Episode IV (1977)", "Action|Adventure|Sci-Fi"),
    (16, "Fargo (1996)", "Comedy|Crime|Drama|Thriller"),
    (17, "Independence Day (1996)", "Action|Adventure|Sci-Fi|Thriller"),
    (18, "Trainspotting (1996)", "Comedy|Crime|Drama"),
    (19, "Mission: Impossible (1996)", "Action|Adventure|Mystery|Thriller"),
    (20, "Twister (1996)", "Action|Adventure|Romance|Thriller"),
    (21, "Godfather, The (1972)", "Crime|Drama"),
    (22, "Casablanca (1942)", "Drama|Romance"),
    (23, "Psycho (1960)", "Crime|Horror"),
    (24, "2001: A Space Odyssey (1968)", "Adventure|Drama|Sci-Fi"),
    (25, "Alien (1979)", "Horror|Sci-Fi"),
    (26, "Blade Runner (1982)", "Action|Sci-Fi|Thriller"),
    (27, "Back to the Future (1985)", "Adventure|Comedy|Sci-Fi"),
    (28, "Princess Bride, The (1987)", "Action|Adventure|Comedy|Fantasy|Romance"),
    (29, "Cinema Paradiso (1988)", "Drama|Romance"),
    (30, "Silence of the Lambs, The (1991)", "Crime|Horror|Thriller"),
    (31, "Jurassic Park (1993)", "Action|Adventure|Sci-Fi|Thriller"),
    (32, "Schindler's List (1993)", "Drama|War"),
    (33, "Pulp Fiction (1994)", "Comedy|Crime|Drama|Thriller"),
    (34, "Shawshank Redemption, The (1994)", "Crime|Drama"),
    (35, "Forrest Gump (1994)", "Comedy|Drama|Romance|War"),
    (36, "Lion King, The (1994)", "Adventure|Animation|Children|Drama|Musical"),
    (37, "Matrix, The (1999)", "Action|Sci-Fi|Thriller"),
    (38, "Fight Club (1999)", "Action|Crime|Drama|Thriller"),
    (39, "Green Mile, The (1999)", "Crime|Drama|Fantasy"),
    (40, "American Beauty (1999)", "Drama|Romance"),
    (41, "Gladiator (2000)", "Action|Adventure|Drama"),
    (42, "Memento (2000)", "Mystery|Thriller"),
    (43, "Amélie (2001)", "Comedy|Romance"),
    (44, "Spirited Away (2001)", "Adventure|Animation|Fantasy"),
    (45, "Lord of the Rings: Fellowship (2001)", "Adventure|Fantasy"),
    (46, "Finding Nemo (2003)", "Adventure|Animation|Children|Comedy"),
    (47, "Eternal Sunshine (2004)", "Drama|Romance|Sci-Fi"),
    (48, "Incredibles, The (2004)", "Action|Adventure|Animation|Children|Comedy"),
    (49, "Batman Begins (2005)", "Action|Crime|IMAX"),
    (50, "Pan's Labyrinth (2006)", "Drama|Fantasy|War"),
    (51, "Ratatouille (2007)", "Animation|Children|Drama"),
    (52, "Dark Knight, The (2008)", "Action|Crime|Drama|IMAX"),
    (53, "WALL-E (2008)", "Adventure|Animation|Children|Romance|Sci-Fi"),
    (54, "Up (2009)", "Adventure|Animation|Children|Drama"),
    (55, "Inception (2010)", "Action|Crime|Drama|Mystery|Sci-Fi|Thriller"),
    (56, "Intouchables (2011)", "Comedy|Drama"),
    (57, "Interstellar (2014)", "Drama|Sci-Fi"),
    (58, "Inside Out (2015)", "Adventure|Animation|Children|Comedy|Drama|Fantasy"),
    (59, "La La Land (2016)", "Comedy|Drama|Musical|Romance"),
    (60, "Parasite (2019)", "Comedy|Drama|Thriller"),
]


def _generate_demo_ratings(movies: pd.DataFrame, n_users: int = 90, seed: int = 42) -> pd.DataFrame:
    """Create a dense-enough, preference-driven demo set (not random white noise)."""
    rng = np.random.default_rng(seed)
    genres = sorted({g for value in movies["genres"] for g in value.split("|")})
    genre_index = {genre: i for i, genre in enumerate(genres)}
    movie_features = np.zeros((len(movies), len(genres)))
    for row, value in enumerate(movies["genres"]):
        for genre in value.split("|"):
            movie_features[row, genre_index[genre]] = 1.0

    # A small quality signal gives popularity recommenders something realistic.
    quality = rng.normal(0.0, 0.32, len(movies))
    quality[[7, 20, 23, 31, 33, 36, 43, 51, 54, 56, 59]] += 0.45
    rows: list[tuple[int, int, float, int]] = []
    timestamp = 1_600_000_000
    for user_id in range(1, n_users + 1):
        tastes = rng.normal(0.0, 0.45, len(genres))
        preferred = rng.choice(len(genres), size=3, replace=False)
        tastes[preferred] += rng.uniform(0.9, 1.5, 3)
        activity = int(rng.integers(12, 29))
        interest = movie_features @ tastes + quality + rng.normal(0, 0.3, len(movies))
        probs = np.exp(interest - interest.max()) + 0.08
        chosen = rng.choice(len(movies), size=activity, replace=False, p=probs / probs.sum())
        generosity = rng.normal(0.0, 0.28)
        for idx in chosen:
            raw = 2.65 + generosity + 0.62 * interest[idx] + rng.normal(0, 0.48)
            rating = float(np.clip(np.round(raw * 2) / 2, 0.5, 5.0))
            rows.append((user_id, int(movies.iloc[idx]["movieId"]), rating, timestamp))
            timestamp += int(rng.integers(100, 5000))
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def load_demo_data(db_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the bundled demo, optionally persisting it as a small SQLite database."""
    movies = pd.DataFrame(MOVIES, columns=["movieId", "title", "genres"])
    ratings = _generate_demo_ratings(movies)
    if db_path is not None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            movies.to_sql("movies", connection, if_exists="replace", index=False)
            ratings.to_sql("ratings", connection, if_exists="replace", index=False)
    return movies, ratings


def load_sqlite_database(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a SQLite database containing MovieLens-compatible movies/ratings tables."""
    with sqlite3.connect(Path(db_path)) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if not {"movies", "ratings"}.issubset(tables):
            raise ValueError("La base SQLite debe contener las tablas 'movies' y 'ratings'.")
        movies = pd.read_sql_query("SELECT * FROM movies", connection)
        ratings = pd.read_sql_query("SELECT * FROM ratings", connection)
    return _normalize_dataframes(movies, ratings)


def save_sqlite_database(movies: pd.DataFrame, ratings: pd.DataFrame, db_path: str | Path) -> None:
    """Persist the active catalogue in a portable SQLite database."""
    with sqlite3.connect(Path(db_path)) as connection:
        movies.to_sql("movies", connection, if_exists="replace", index=False)
        ratings.to_sql("ratings", connection, if_exists="replace", index=False)


def _read_csv_file(file_or_path) -> pd.DataFrame:
    if hasattr(file_or_path, "getvalue"):
        return pd.read_csv(BytesIO(file_or_path.getvalue()))
    return pd.read_csv(file_or_path)


def _normalize_dataframes(movies: pd.DataFrame, ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_movies = {"movieId", "title", "genres"}
    required_ratings = {"userId", "movieId", "rating"}
    if not required_movies.issubset(movies.columns):
        raise ValueError(f"movies.csv debe contener: {', '.join(sorted(required_movies))}")
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"ratings.csv debe contener: {', '.join(sorted(required_ratings))}")
    movies = movies[["movieId", "title", "genres"]].copy()
    keep = ["userId", "movieId", "rating"] + (["timestamp"] if "timestamp" in ratings.columns else [])
    ratings = ratings[keep].dropna(subset=["userId", "movieId", "rating"]).copy()
    movies["movieId"] = movies["movieId"].astype(int)
    ratings[["userId", "movieId"]] = ratings[["userId", "movieId"]].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
    return movies.drop_duplicates("movieId"), ratings.drop_duplicates(["userId", "movieId"], keep="last")


def load_movielens_files(movies_file, ratings_file) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and normalize MovieLens-compatible CSV files."""
    movies = _read_csv_file(movies_file)
    ratings = _read_csv_file(ratings_file)
    return _normalize_dataframes(movies, ratings)
