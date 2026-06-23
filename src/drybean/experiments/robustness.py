from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from drybean.config import PATHS, RANDOM_SEED, TARGET, ensure_artifact_dirs
from drybean.data import load_splits
from drybean.features import add_engineered_features
from drybean.metrics import classification_metrics
from drybean.models import build_model_specs
from drybean.noise import add_feature_noise, flip_labels
from drybean.preprocessing import TabularPreprocessor

NOISE_LEVELS = {
    "gaussian": (0.05, 0.10, 0.20),
    "impulse": (0.01, 0.03, 0.05),
    "missing": (0.05, 0.10, 0.20),
    "label": (0.05, 0.10, 0.20),
}


def fit_robust_model(model_key, model, x, y, n_classes):
    if model_key == "logistic_regression":
        classes = np.arange(n_classes)
        for epoch in range(70):
            order = np.random.default_rng(RANDOM_SEED + epoch).permutation(
                len(y)
            )
            model.partial_fit(x.iloc[order], y[order], classes=classes)
        return model
    model.fit(x, y)
    return model


def run_robustness_experiment() -> pd.DataFrame:
    ensure_artifact_dirs()
    splits = load_splits()
    train = splits.train.drop_duplicates().reset_index(drop=True)
    raw_x_train = add_engineered_features(train.drop(columns=TARGET))
    raw_x_test = add_engineered_features(splits.test.drop(columns=TARGET))
    encoder = LabelEncoder().fit(train[TARGET].astype(str))
    y_train = encoder.transform(train[TARGET].astype(str))
    y_test = encoder.transform(splits.test[TARGET].astype(str))
    rows = []
    for model_key, spec in build_model_specs().items():
        print(f"[robustness] clean baseline: {model_key}")
        clean_processor = TabularPreprocessor(scale=spec.scale)
        clean_x_train = clean_processor.fit_transform(raw_x_train)
        clean_x_test = clean_processor.transform(raw_x_test)
        clean_model = fit_robust_model(
            model_key,
            build_model_specs()[model_key].estimator,
            clean_x_train,
            y_train,
            len(encoder.classes_),
        )
        clean_scores = classification_metrics(
            y_test, clean_model.predict(clean_x_test)
        )
        for noise_type, levels in NOISE_LEVELS.items():
            for strength in levels:
                print(f"[robustness] {model_key}: {noise_type}={strength}")
                noisy_x = raw_x_train.to_numpy(copy=True)
                noisy_y = y_train.copy()
                if noise_type == "label":
                    noisy_y = flip_labels(
                        noisy_y,
                        strength,
                        len(encoder.classes_),
                        RANDOM_SEED,
                    )
                else:
                    noisy_x = add_feature_noise(
                        noisy_x,
                        noise_type,
                        strength,
                        RANDOM_SEED,
                    )
                noisy_frame = pd.DataFrame(
                    noisy_x,
                    columns=raw_x_train.columns,
                )
                processor = TabularPreprocessor(scale=spec.scale)
                processed_train = processor.fit_transform(noisy_frame)
                processed_test = processor.transform(raw_x_test)
                model = build_model_specs()[model_key].estimator
                started = perf_counter()
                model = fit_robust_model(
                    model_key,
                    model,
                    processed_train,
                    noisy_y,
                    len(encoder.classes_),
                )
                scores = classification_metrics(
                    y_test, model.predict(processed_test)
                )
                rows.append(
                    {
                        "model_key": model_key,
                        "model": spec.name,
                        "noise_type": noise_type,
                        "strength": strength,
                        "test_accuracy": scores["accuracy"],
                        "test_macro_f1": scores["macro_f1"],
                        "accuracy_drop": (
                            clean_scores["accuracy"] - scores["accuracy"]
                        ),
                        "macro_f1_drop": (
                            clean_scores["macro_f1"] - scores["macro_f1"]
                        ),
                        "train_seconds": perf_counter() - started,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(PATHS.results / "robustness.csv", index=False)
    return result

