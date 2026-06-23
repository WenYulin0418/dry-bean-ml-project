import pandas as pd

VALID_CLASSES = (
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
)


def normalize_label(value: object) -> str:
    label = str(value).strip().upper().replace("0", "O").replace("3", "E")
    if label not in VALID_CLASSES:
        raise ValueError(f"无法识别的标签: {value!r} -> {label!r}")
    return label


def clean_label_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_label).astype(
        pd.CategoricalDtype(categories=VALID_CLASSES)
    )

