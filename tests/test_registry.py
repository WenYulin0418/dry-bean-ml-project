from drybean.models import build_model_specs


def test_registry_contains_five_required_models():
    specs = build_model_specs()

    assert set(specs) == {
        "logistic_regression",
        "knn",
        "random_forest",
        "xgboost",
        "handmade_gnb",
    }
    assert sum(spec.iterative_loss for spec in specs.values()) == 2

