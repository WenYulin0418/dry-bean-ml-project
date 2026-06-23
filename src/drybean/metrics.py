from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }


def median_inference_seconds(model, x, repeats: int = 20) -> float:
    model.predict(x[: min(len(x), 32)])
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        model.predict(x)
        samples.append(perf_counter() - started)
    return float(median(samples))


def serialized_model_size_mb(model, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return float(path.stat().st_size / (1024**2))

