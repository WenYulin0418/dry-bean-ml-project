from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularPreprocessor:
    scale: bool
    clip_quantiles: tuple[float, float] | None = (0.005, 0.995)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.columns_ = frame.columns.tolist()
        numeric = frame.astype(float).replace([np.inf, -np.inf], np.nan)
        self.medians_ = numeric.median()
        filled = numeric.fillna(self.medians_)
        if self.clip_quantiles is None:
            self.lower_ = None
            self.upper_ = None
        else:
            low, high = self.clip_quantiles
            self.lower_ = filled.quantile(low)
            self.upper_ = filled.quantile(high)
            filled = filled.clip(self.lower_, self.upper_, axis=1)
        self.scaler_ = StandardScaler().fit(filled) if self.scale else None
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise RuntimeError("预处理器尚未拟合")
        if frame.columns.tolist() != self.columns_:
            raise ValueError("特征列或顺序与训练集不一致")
        result = (
            frame.astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians_)
        )
        if self.lower_ is not None:
            result = result.clip(self.lower_, self.upper_, axis=1)
        if self.scaler_ is not None:
            result = pd.DataFrame(
                self.scaler_.transform(result),
                columns=self.columns_,
                index=frame.index,
            )
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)

