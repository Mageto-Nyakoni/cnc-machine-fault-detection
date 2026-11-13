"""Check shared partitions and scoring with small classification fixtures."""

import unittest

import pandas as pd

from utils.classification import classification_splits, compare_classifiers


class ClassificationTests(unittest.TestCase):
    def test_splits_reproduce_window_sampling_and_hold_out_whole_recordings(self):
        features = pd.DataFrame([
            {"source_file": str(file), "filename": str(file), "healthy": file < 10}
            for file in range(11)
            for _ in range(5)
        ])

        splits = classification_splits(features, train_fraction=0.8, seed=42)
        window_train, window_holdouts = splits["anomaly"]
        expected_train = features.loc[features["healthy"]].sample(n=44, random_state=42)
        pd.testing.assert_frame_equal(window_train, expected_train)

        parts = [window_train, *window_holdouts.values()]
        self.assertEqual(sorted(pd.concat(parts).index), features.index.tolist())
        self.assertTrue(window_train["healthy"].all())
        self.assertTrue(any(not part["healthy"].all() for part in window_holdouts.values()))

        recording_train, recording_holdouts = splits["recording"]
        test = recording_holdouts["test"]
        self.assertTrue(set(recording_train["source_file"]).isdisjoint(test["source_file"]))
        self.assertEqual(test["source_file"].nunique(), 2)
        self.assertEqual(
            sorted(pd.concat([recording_train, test]).index),
            features.index[features["healthy"]].tolist(),
        )

    def test_both_models_keep_unseen_classes_in_scoring(self):
        train = pd.DataFrame({"feature": range(13), "operation": [3] * 10 + [7] * 3})
        holdout = pd.DataFrame({"feature": [0, 1, 2], "operation": [11] * 3})
        metrics, support = compare_classifiers(
            train, {"test": holdout}, ["feature"], {"operation": "operation"}, seed=42
        )

        self.assertEqual(metrics["model"].tolist(), ["Random Forest", "XGBoost"])
        self.assertEqual(metrics["accuracy"].tolist(), [0.0, 0.0])
        self.assertEqual(metrics["macro_f1"].tolist(), [0.0, 0.0])
        self.assertTrue(metrics["fit_seconds"].ge(0).all())
        unseen = support.loc[support["class"] == 11]
        self.assertEqual(unseen["train_windows"].tolist(), [0, 0])
        self.assertEqual(unseen["holdout_windows"].tolist(), [3, 3])

    def test_one_training_class_reports_unavailable_xgboost(self):
        train = pd.DataFrame({"feature": [1, 2], "machine": [3, 3]})
        metrics, _ = compare_classifiers(
            train, {"test": train}, ["feature"], {"machine": "machine"}, seed=42
        )

        forest, xgboost = metrics.to_dict("records")
        self.assertEqual(forest["accuracy"], 1.0)
        self.assertEqual(xgboost["status"], "XGBoost needs at least two training classes")
        self.assertTrue(pd.isna(xgboost["fit_seconds"]))


if __name__ == "__main__":
    unittest.main()
