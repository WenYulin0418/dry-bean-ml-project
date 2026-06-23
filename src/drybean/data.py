from dataclasses import dataclass

import pandas as pd

from drybean.config import PATHS, RAW_FEATURES, TARGET
from drybean.labels import clean_label_series


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def coerce_numeric_features(
    frame: pd.DataFrame,
    feature_columns,
) -> pd.DataFrame:
    result = frame.copy()
    numeric_pattern = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    for column in feature_columns:
        extracted = result[column].astype("string").str.extract(
            numeric_pattern,
            expand=False,
        )
        result[column] = pd.to_numeric(
            extracted,
            errors="coerce",
        ).astype("float64")
    return result


def _read_split(filename: str) -> pd.DataFrame:
    path = PATHS.data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    frame = pd.read_csv(path)
    expected = [*RAW_FEATURES, TARGET]
    if frame.columns.tolist() != expected:
        raise ValueError(f"{filename} 列结构不符合预期")
    frame = coerce_numeric_features(frame, RAW_FEATURES)
    frame[TARGET] = clean_label_series(frame[TARGET])
    return frame


def load_splits() -> DatasetSplits:
    return DatasetSplits(
        train=_read_split("Dry_Bean_Dataset_Dirty_train.csv"),
        val=_read_split("Dry_Bean_Dataset_Dirty_val.csv"),
        test=_read_split("Dry_Bean_Dataset_Dirty_test.csv"),
    )


def profile_frame(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": (
            {
                str(k): int(v)
                for k, v in frame[TARGET].value_counts(dropna=False).items()
            }
            if TARGET in frame
            else {}
        ),
        "numeric_summary": frame.select_dtypes("number").describe().to_dict(),
    }
