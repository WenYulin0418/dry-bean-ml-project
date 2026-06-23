import pandas as pd

from drybean.config import PATHS, ensure_artifact_dirs
from drybean.experiments.baseline import train_one
from drybean.models import build_model_specs


def run_feature_ablation() -> pd.DataFrame:
    ensure_artifact_dirs()
    rows = []
    for engineered in (False, True):
        for model_key in build_model_specs():
            print(f"[ablation] {model_key}, engineered={engineered}")
            result, _ = train_one(
                model_key,
                engineered=engineered,
                save_artifacts=False,
            )
            rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(PATHS.results / "feature_ablation.csv", index=False)
    return frame

