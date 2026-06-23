import numpy as np
import pandas as pd

ENGINEERED_FEATURES = (
    "ConvexGap",
    "AxisLengthDiff",
    "AreaConvexRatio",
    "PerimeterDiameterRatio",
)


def _safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    return left.div(right.replace(0, np.nan))


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["ConvexGap"] = result["ConvexArea"] - result["Area"]
    result["AxisLengthDiff"] = (
        result["MajorAxisLength"] - result["MinorAxisLength"]
    )
    result["AreaConvexRatio"] = _safe_divide(
        result["Area"], result["ConvexArea"]
    )
    result["PerimeterDiameterRatio"] = _safe_divide(
        result["Perimeter"], result["EquivDiameter"]
    )
    return result

