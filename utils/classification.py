"""Shared partitions and evaluation for the vibration classification benchmark."""

from time import perf_counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def classification_splits(features, train_fraction, seed):
    """Reproduce both of the original notebook's split protocols."""
    healthy = features.loc[features["healthy"]]
    faulty = features.loc[~features["healthy"]].sort_values("filename", kind="stable")
    train_count = round(train_fraction * len(features))
    if not 0 < train_count < len(healthy):
        raise ValueError("The anomaly split needs enough healthy windows for training and holdout.")

    window_train = healthy.sample(n=train_count, random_state=seed)
    healthy_holdout = healthy.drop(window_train.index)
    healthy_midpoint = len(healthy_holdout) // 2
    faulty_midpoint = len(faulty) // 2
    window_holdouts = {
        "selection": pd.concat([healthy_holdout.iloc[:healthy_midpoint], faulty.iloc[:faulty_midpoint]]),
        "final test": pd.concat([healthy_holdout.iloc[healthy_midpoint:], faulty.iloc[faulty_midpoint:]]),
    }

    splitter = GroupShuffleSplit(n_splits=1, test_size=1 - train_fraction, random_state=seed)
    train_idx, test_idx = next(splitter.split(healthy, groups=healthy["source_file"]))
    recording_train = healthy.iloc[train_idx]
    recording_test = healthy.iloc[test_idx]
    if set(recording_train["source_file"]) & set(recording_test["source_file"]):
        raise RuntimeError("The recording split contains overlapping training and test files.")

    return {
        "anomaly": (window_train, window_holdouts),
        "recording": (recording_train, {"test": recording_test}),
    }


def compare_classifiers(train, holdouts, feature_columns, targets, seed):
    """Fit each model once per target and score every holdout without dropping rows."""
    metrics = []
    support = []
    X_train = train[feature_columns]
    X_holdouts = {name: partition[feature_columns] for name, partition in holdouts.items()}

    for target, column in targets.items():
        y_train = train[column]
        encoder = LabelEncoder().fit(y_train)
        estimators = {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                bootstrap=True,
                max_samples=min(1024, len(train)),
                max_features=0.7,
                random_state=seed,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300, tree_method="hist", random_state=seed, n_jobs=-1
            ),
        }

        for name, estimator in estimators.items():
            fit_seconds = None
            if name == "Random Forest" or len(encoder.classes_) >= 2:
                if name == "XGBoost":
                    labels = encoder.transform(y_train)
                    fit_args = {"sample_weight": compute_sample_weight("balanced", y_train)}
                else:
                    labels = y_train
                    fit_args = {}
                start = perf_counter()
                estimator.fit(X_train, labels, **fit_args)
                fit_seconds = perf_counter() - start

            for split, partition in holdouts.items():
                actual = partition[column]
                row = {
                    "target": target,
                    "model": name,
                    "holdout": split,
                    "accuracy": None,
                    "macro_f1": None,
                    "fit_seconds": fit_seconds,
                    "status": "ok",
                }
                if fit_seconds is None:
                    row["status"] = "XGBoost needs at least two training classes"
                elif actual.empty:
                    row["status"] = "empty holdout"
                else:
                    predictions = estimator.predict(X_holdouts[split])
                    if name == "XGBoost":
                        predictions = encoder.inverse_transform(predictions)
                    report = classification_report(
                        actual, predictions, labels=sorted(actual.unique()),
                        output_dict=True, zero_division=0,
                    )
                    row["accuracy"] = accuracy_score(actual, predictions)
                    row["macro_f1"] = report["macro avg"]["f1-score"]
                    for label in sorted(set(y_train) | set(actual)):
                        class_scores = report.get(str(label), {})
                        support.append({
                            "target": target,
                            "model": name,
                            "holdout": split,
                            "class": label,
                            "train_windows": int((y_train == label).sum()),
                            "holdout_windows": int((actual == label).sum()),
                            "f1": class_scores.get("f1-score"),
                        })
                metrics.append(row)

    return pd.DataFrame(metrics), pd.DataFrame(support)
