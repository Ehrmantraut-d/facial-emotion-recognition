"""Train and evaluate educational KNN and multilayer perceptron classifiers."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def candidate_models(seed: int = 42) -> list[tuple[str, Any]]:
    """Return baseline candidates; MLP training is deterministic for the given seed."""
    return [(f"knn_k{k}", KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1))
            for k in (3, 5, 7, 11)] + [
        ("mlp_128_64", MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
         solver="adam", alpha=1e-4, batch_size=128, learning_rate_init=1e-3,
         max_iter=50, early_stopping=True, validation_fraction=0.1,
         n_iter_no_change=8, random_state=seed, verbose=False))]


def predict_with_flip_average(model: Any, X: np.ndarray, enabled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Predict from original and mirrored 48x48 faces, averaging class probabilities."""
    values = np.asarray(X, dtype=np.float32)
    probabilities = model.predict_proba(values)
    if enabled:
        flipped = np.ascontiguousarray(np.flip(values.reshape(-1, 48, 48), axis=2).reshape(values.shape))
        probabilities = (probabilities + model.predict_proba(flipped)) / 2.0
    classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
    predictions = classes[np.argmax(probabilities, axis=1)]
    return predictions, probabilities


def fit_and_score(name: str, model: Any, Xtrain: np.ndarray, ytrain: np.ndarray,
                  Xval: np.ndarray, yval: np.ndarray) -> dict:
    """Fit one candidate and score its validation predictions."""
    start = time.perf_counter()
    model.fit(Xtrain, ytrain)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    pred = model.predict(Xval)
    latency_ms = (time.perf_counter() - start) * 1000 / max(1, len(yval))
    return {"name": name, "estimator": model, "train_seconds": train_seconds,
            "prediction_latency_ms_per_sample": latency_ms,
            "accuracy": float(accuracy_score(yval, pred)),
            "macro_f1": float(f1_score(yval, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(yval, pred, labels=np.arange(7)).tolist(),
            "classification_report": classification_report(yval, pred, labels=np.arange(7),
                target_names=CLASS_NAMES, zero_division=0, output_dict=True),
            "train_samples": int(len(ytrain)), "validation_samples": int(len(yval))}


def final_metrics(model: Any, X: np.ndarray, y: np.ndarray,
                  flip_average: bool = False) -> dict:
    """Calculate final metrics, optionally using validation-selected flip averaging."""
    pred, _ = predict_with_flip_average(model, X, enabled=flip_average)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y, pred, labels=np.arange(7)).tolist(),
            "classification_report": classification_report(y, pred, labels=np.arange(7),
                target_names=CLASS_NAMES, zero_division=0, output_dict=True)}
