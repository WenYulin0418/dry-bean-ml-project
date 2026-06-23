from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from drybean.config import RANDOM_SEED
from drybean.models.gaussian_nb import HandmadeGaussianNB


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    scale: bool
    iterative_loss: bool


def build_model_specs() -> dict[str, ModelSpec]:
    return {
        "logistic_regression": ModelSpec(
            name="逻辑回归",
            estimator=SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=1,
                tol=None,
                random_state=RANDOM_SEED,
            ),
            scale=True,
            iterative_loss=True,
        ),
        "knn": ModelSpec(
            name="KNN",
            estimator=KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                n_jobs=-1,
            ),
            scale=True,
            iterative_loss=False,
        ),
        "random_forest": ModelSpec(
            name="随机森林",
            estimator=RandomForestClassifier(
                n_estimators=250,
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            scale=False,
            iterative_loss=False,
        ),
        "xgboost": ModelSpec(
            name="XGBoost",
            estimator=XGBClassifier(
                n_estimators=220,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            scale=False,
            iterative_loss=True,
        ),
        "handmade_gnb": ModelSpec(
            name="手写高斯朴素贝叶斯",
            estimator=HandmadeGaussianNB(),
            scale=False,
            iterative_loss=False,
        ),
    }

