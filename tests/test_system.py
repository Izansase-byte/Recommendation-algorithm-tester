import unittest

import numpy as np

from recommender.analysis import data_quality_report, genre_statistics, movie_statistics, popularity_curve, user_statistics
from recommender.data import load_demo_data
from recommender.evaluation import (
    MODEL_NAMES,
    evaluate_models,
    optimize_hybrid_weights,
    recommend_for_user,
    train_test_split_by_user,
)
from recommender.models import build_model


class RecommenderSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies, cls.ratings = load_demo_data()
        cls.train, cls.test = train_test_split_by_user(cls.ratings, seed=7)

    def test_split_has_no_overlap_and_keeps_history(self):
        train_pairs = set(map(tuple, self.train[["userId", "movieId"]].to_numpy()))
        test_pairs = set(map(tuple, self.test[["userId", "movieId"]].to_numpy()))
        self.assertFalse(train_pairs & test_pairs)
        self.assertGreaterEqual(self.train.groupby("userId").size().min(), 2)

    def test_every_model_predicts_in_rating_range(self):
        sample_user = int(self.test.iloc[0]["userId"])
        sample_items = self.test[self.test["userId"] == sample_user]["movieId"].tolist()
        for name in MODEL_NAMES:
            model = build_model(name).fit(self.train, self.movies)
            predictions = model.predict(sample_user, sample_items)
            self.assertEqual(len(predictions), len(sample_items), name)
            self.assertTrue(np.isfinite(predictions).all(), name)
            self.assertTrue(((predictions >= 0.5) & (predictions <= 5.0)).all(), name)

    def test_evaluation_and_recommendations(self):
        names = ["Modelo base", "Modelo híbrido"]
        results, fitted = evaluate_models(names, self.train, self.test, self.movies, top_k=5)
        self.assertEqual(set(results["Algoritmo"]), set(names))
        self.assertTrue((results["RMSE"] > 0).all())
        recommendations = recommend_for_user(fitted["Modelo híbrido"], 1, self.movies, self.train, n=5)
        self.assertEqual(len(recommendations), 5)
        seen = set(self.train.loc[self.train["userId"] == 1, "movieId"])
        self.assertFalse(seen & set(recommendations["movieId"]))

    def test_diversity_reranking_and_hybrid_optimization(self):
        weights, attempts = optimize_hybrid_weights(
            self.train, self.test, self.movies, trials=10, objective="RMSE", seed=4
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreaterEqual(len(attempts), 16)
        model = build_model("Modelo híbrido", {"weights": weights}).fit(self.train, self.movies)
        recommendations = recommend_for_user(
            model, 2, self.movies, self.train, n=7, diversity_strength=0.5
        )
        self.assertEqual(len(recommendations), 7)
        self.assertEqual(recommendations["movieId"].nunique(), 7)

    def test_database_analysis_tables(self):
        movies = movie_statistics(self.movies, self.ratings)
        users = user_statistics(self.movies, self.ratings)
        genres = genre_statistics(self.movies, self.ratings)
        quality = data_quality_report(self.movies, self.ratings)
        tail = popularity_curve(self.movies, self.ratings)
        self.assertEqual(len(movies), len(self.movies))
        self.assertEqual(len(users), self.ratings["userId"].nunique())
        self.assertIn("Drama", set(genres["género"]))
        self.assertGreater(quality["Esparsidad (%)"], 0)
        self.assertAlmostEqual(float(tail.iloc[-1]["porcentaje_acumulado"]), 100.0)


if __name__ == "__main__":
    unittest.main()
