"""Command line training entry point for YouTube spam detection."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .features import build_feature_transformer
from .utils import LABEL_COLUMN, TEXT_COLUMN, load_comment_dataset


def build_classifiers(random_state: int = 42) -> list[tuple[str, object]]:
    """Create the classifier set used by the experiment."""
    classifiers: list[tuple[str, object]] = [
        (
            "lr",
            LogisticRegression(C=1.5, penalty="l2", solver="lbfgs", max_iter=1_000),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        classifiers.append(
            (
                "xgb",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        classifiers.append(
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    reg_alpha=0.1,
                    random_state=random_state,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    except ImportError:
        pass

    return classifiers


def evaluate_model(model: Pipeline, x_test, y_test) -> dict[str, float]:
    """Calculate common binary classification metrics."""
    start = time.perf_counter()
    prediction = model.predict(x_test)
    inference_seconds = time.perf_counter() - start

    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "inference_seconds": inference_seconds,
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    """Train and evaluate the selected feature set."""
    frame = load_comment_dataset(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[TEXT_COLUMN],
        frame[LABEL_COLUMN],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=frame[LABEL_COLUMN],
    )

    classifiers = build_classifiers(args.random_state)
    if args.classifier == "ensemble":
        estimator = VotingClassifier(classifiers, voting="soft", n_jobs=-1)
    else:
        matches = [model for name, model in classifiers if name == args.classifier]
        if not matches:
            available = ", ".join(["ensemble", *[name for name, _ in classifiers]])
            raise ValueError(f"Unknown or unavailable classifier '{args.classifier}'. Available: {available}")
        estimator = matches[0]

    pipeline = Pipeline(
        [
            ("features", build_feature_transformer(args.feature_set)),
            ("classifier", estimator),
        ]
    )

    pipeline.fit(x_train, y_train)
    metrics = evaluate_model(pipeline, x_test, y_test)

    if args.model_out:
        output_path = Path(args.model_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, output_path)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YouTube spam detection model.")
    parser.add_argument("--data", required=True, help="Path to a CSV file with CONTENT and CLASS columns.")
    parser.add_argument(
        "--feature-set",
        choices=["tfidf", "roberta", "hybrid"],
        default="tfidf",
        help="Feature representation to train.",
    )
    parser.add_argument(
        "--classifier",
        choices=["ensemble", "lr", "rf", "xgb", "lgbm"],
        default="ensemble",
        help="Classifier to train. Boosted models are available only if installed.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--model-out", help="Optional path for a saved joblib model.")
    return parser.parse_args()


def main() -> None:
    metrics = train(parse_args())
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
