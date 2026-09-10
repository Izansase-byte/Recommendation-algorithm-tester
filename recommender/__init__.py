"""Motor de recomendación y evaluación para MovieLab."""

from .data import load_demo_data, load_movielens_files, load_sqlite_database, save_sqlite_database
from .analysis import data_quality_report, genre_statistics, movie_statistics, popularity_curve, user_statistics
from .evaluation import evaluate_models, optimize_hybrid_weights, train_test_split_by_user
from .models import (
    BaselineRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    UserKNNRecommender,
    build_model,
)

__all__ = [
    "BaselineRecommender",
    "ContentBasedRecommender",
    "HybridRecommender",
    "ItemKNNRecommender",
    "UserKNNRecommender",
    "build_model",
    "data_quality_report",
    "evaluate_models",
    "optimize_hybrid_weights",
    "load_demo_data",
    "load_movielens_files",
    "load_sqlite_database",
    "genre_statistics",
    "movie_statistics",
    "popularity_curve",
    "save_sqlite_database",
    "train_test_split_by_user",
    "user_statistics",
]
