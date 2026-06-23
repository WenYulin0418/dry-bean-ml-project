from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder

from drybean.config import PATHS, RANDOM_SEED, TARGET, ensure_artifact_dirs
from drybean.data import load_splits
from drybean.features import add_engineered_features
from drybean.metrics import (
    classification_metrics,
    median_inference_seconds,
    serialized_model_size_mb,
)
from drybean.models import build_model_specs
from drybean.preprocessing import TabularPreprocessor


@dataclass
class PreparedSplit:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    encoder: LabelEncoder


def prepare_data(
    engineered: bool,
    scale: bool,
) -> tuple[PreparedSplit, TabularPreprocessor]:
    splits = load_splits()
    train = splits.train.drop_duplicates().reset_index(drop=True)
    transform = add_engineered_features if engineered else lambda frame: frame.copy()
    x_train = transform(train.drop(columns=TARGET))
    x_val = transform(splits.val.drop(columns=TARGET))
    x_test = transform(splits.test.drop(columns=TARGET))
    processor = TabularPreprocessor(scale=scale)
    x_train = processor.fit_transform(x_train)
    x_val = processor.transform(x_val)
    x_test = processor.transform(x_test)
    encoder = LabelEncoder().fit(train[TARGET].astype(str))
    return (
        PreparedSplit(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=encoder.transform(train[TARGET].astype(str)),
            y_val=encoder.transform(splits.val[TARGET].astype(str)),
            y_test=encoder.transform(splits.test[TARGET].astype(str)),
            encoder=encoder,
        ),
        processor,
    )


def fit_logistic_with_history(model, data: PreparedSplit, epochs: int = 100):
    history = []
    classes = np.arange(len(data.encoder.classes_))
    for epoch in range(epochs):
        order = np.random.default_rng(RANDOM_SEED + epoch).permutation(
            len(data.y_train)
        )
        model.partial_fit(
            data.x_train.iloc[order],
            data.y_train[order],
            classes=classes,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(
                    log_loss(data.y_train, model.predict_proba(data.x_train))
                ),
                "val_loss": float(
                    log_loss(data.y_val, model.predict_proba(data.x_val))
                ),
            }
        )
    return model, history


def _fit_model(model_key: str, model, data: PreparedSplit):
    if model_key == "logistic_regression":
        return fit_logistic_with_history(model, data)
    if model_key == "xgboost":
        model.fit(
            data.x_train,
            data.y_train,
            eval_set=[
                (data.x_train, data.y_train),
                (data.x_val, data.y_val),
            ],
            verbose=False,
        )
        evals = model.evals_result()
        history = [
            {
                "epoch": index + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            for index, (train_loss, val_loss) in enumerate(
                zip(
                    evals["validation_0"]["mlogloss"],
                    evals["validation_1"]["mlogloss"],
                )
            )
        ]
        return model, history
    model.fit(data.x_train, data.y_train)
    return model, []


def _save_diagnostics(model_key: str, model, data: PreparedSplit, test_pred) -> None:
    report = classification_report(
        data.y_test,
        test_pred,
        labels=np.arange(len(data.encoder.classes_)),
        target_names=data.encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        PATHS.results / f"class_report_{model_key}.csv"
    )
    pd.DataFrame(
        confusion_matrix(data.y_test, test_pred),
        index=data.encoder.classes_,
        columns=data.encoder.classes_,
    ).to_csv(PATHS.results / f"confusion_{model_key}.csv")
    if hasattr(model, "feature_importances_"):
        pd.DataFrame(
            {
                "feature": data.x_train.columns,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False).to_csv(
            PATHS.results / f"feature_importance_{model_key}.csv",
            index=False,
        )


def train_one(
    model_key: str,
    engineered: bool = True,
    save_artifacts: bool = True,
) -> tuple[dict, list[dict]]:
    ensure_artifact_dirs()
    spec = build_model_specs()[model_key]
    data, processor = prepare_data(engineered=engineered, scale=spec.scale)
    started = perf_counter()
    model, history = _fit_model(model_key, spec.estimator, data)
    train_seconds = perf_counter() - started
    train_pred = model.predict(data.x_train)
    test_pred = model.predict(data.x_test)
    train_scores = classification_metrics(data.y_train, train_pred)
    test_scores = classification_metrics(data.y_test, test_pred)
    inference_seconds = median_inference_seconds(model, data.x_test)
    suffix = "" if engineered else "_raw_features"
    bundle_path = PATHS.models / f"{model_key}{suffix}.joblib"
    bundle = {"model": model, "processor": processor, "encoder": data.encoder}
    model_size = serialized_model_size_mb(bundle, bundle_path)
    result = {
        "model_key": model_key,
        "model": spec.name,
        **{f"train_{key}": value for key, value in train_scores.items()},
        **{f"test_{key}": value for key, value in test_scores.items()},
        "accuracy_gap": train_scores["accuracy"] - test_scores["accuracy"],
        "macro_f1_gap": train_scores["macro_f1"] - test_scores["macro_f1"],
        "train_seconds": float(train_seconds),
        "inference_seconds": inference_seconds,
        "inference_ms_per_sample": inference_seconds * 1000 / len(data.x_test),
        "model_size_mb": model_size,
        "engineered_features": engineered,
    }
    if save_artifacts:
        _save_diagnostics(model_key, model, data, test_pred)
    return result, history


def run_baseline_experiment(engineered: bool = True) -> pd.DataFrame:
    rows = []
    histories = []
    for model_key in build_model_specs():
        print(f"[train] {model_key}")
        result, history = train_one(model_key, engineered=engineered)
        rows.append(result)
        histories.extend(
            {"model_key": model_key, **record} for record in history
        )
    results = pd.DataFrame(rows).sort_values(
        "test_accuracy", ascending=False
    )
    results.to_csv(PATHS.results / "model_comparison.csv", index=False)
    pd.DataFrame(
        histories,
        columns=["model_key", "epoch", "train_loss", "val_loss"],
    ).to_csv(PATHS.results / "loss_history.csv", index=False)
    return results

