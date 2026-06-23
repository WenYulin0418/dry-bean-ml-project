from pathlib import Path

from drybean.metrics import classification_metrics


def test_classification_metrics_contains_required_scores():
    result = classification_metrics(["A", "A", "B"], ["A", "B", "B"])

    assert set(result) == {
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
    }
    assert 0.0 <= result["accuracy"] <= 1.0


def test_classification_metrics_can_be_serialized_as_floats(tmp_path: Path):
    result = classification_metrics(["A", "B"], ["A", "B"])

    assert all(type(value) is float for value in result.values())

