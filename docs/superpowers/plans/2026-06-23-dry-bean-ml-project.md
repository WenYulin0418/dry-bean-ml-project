# Dry Bean 多分类机器学习期末项目 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可通过统一命令复现的 Dry Bean 多分类项目，生成五算法实验结果、鲁棒性分析、Streamlit 展示、GitHub README 和经渲染检查的 Word 论文。

**Architecture:** 使用 `src/drybean` Python 包承载数据、预处理、特征、模型、实验与绘图逻辑，所有离线结果写入 `artifacts/`，Streamlit 和论文只读取这些结果。训练集拟合全部预处理器，验证集负责模型选择，测试集仅用于最终评价；所有实验由 Typer CLI 调度。

**Tech Stack:** Python 3.11、pandas、NumPy、scikit-learn、XGBoost、Matplotlib、Seaborn、Joblib、Typer、Streamlit、pytest、python-docx。

---

## 文件结构

```text
.
├── DryBeanDataset/
├── pyproject.toml
├── README.md
├── src/drybean/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── labels.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── noise.py
│   ├── metrics.py
│   ├── plots.py
│   ├── cli.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gaussian_nb.py
│   │   └── registry.py
│   └── experiments/
│       ├── __init__.py
│       ├── baseline.py
│       ├── ablation.py
│       └── robustness.py
├── app/
│   └── streamlit_app.py
├── scripts/
│   └── build_paper.py
├── tests/
│   ├── test_labels.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_gaussian_nb.py
│   ├── test_noise.py
│   ├── test_metrics.py
│   └── test_cli.py
├── artifacts/
│   ├── data/
│   ├── figures/
│   ├── models/
│   ├── results/
│   └── screenshots/
└── paper/
    ├── paper_metadata.json
    └── Dry_Bean_机器学习课程论文.docx
```

## Task 1: 建立项目骨架和依赖

**Files:**
- Create: `pyproject.toml`
- Create: `src/drybean/__init__.py`
- Create: `src/drybean/config.py`
- Create: `src/drybean/models/__init__.py`
- Create: `src/drybean/experiments/__init__.py`

- [ ] **Step 1: 写入包元数据和命令入口**

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "drybean-ml-project"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
  "joblib>=1.3",
  "matplotlib>=3.8",
  "numpy>=1.26",
  "pandas>=2.1",
  "python-docx>=1.1",
  "scikit-learn>=1.4",
  "seaborn>=0.13",
  "streamlit>=1.35",
  "typer>=0.12",
  "xgboost>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
drybean = "drybean.cli:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: 定义统一路径与随机种子**

```python
# src/drybean/config.py
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    data_dir: Path = ROOT / "DryBeanDataset"
    artifacts: Path = ROOT / "artifacts"
    results: Path = ROOT / "artifacts" / "results"
    figures: Path = ROOT / "artifacts" / "figures"
    models: Path = ROOT / "artifacts" / "models"
    screenshots: Path = ROOT / "artifacts" / "screenshots"
    paper: Path = ROOT / "paper"

PATHS = Paths()
RANDOM_SEED = 42
TARGET = "Class"
RAW_FEATURES = (
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
)

def ensure_artifact_dirs() -> None:
    for path in (
        PATHS.artifacts, PATHS.results, PATHS.figures,
        PATHS.models, PATHS.screenshots, PATHS.paper,
        PATHS.artifacts / "data",
    ):
        path.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: 安装项目并验证导入**

Run: `python -m pip install -e ".[dev]"`

Expected: 安装成功，`python -c "import drybean; print(drybean.__name__)"` 输出 `drybean`。

- [ ] **Step 4: 提交骨架**

```bash
git add pyproject.toml src/drybean
git commit -m "build: scaffold dry bean project"
```

## Task 2: 实现标签清洗与数据加载

**Files:**
- Create: `tests/test_labels.py`
- Create: `src/drybean/labels.py`
- Create: `src/drybean/data.py`

- [ ] **Step 1: 编写标签清洗失败测试**

```python
# tests/test_labels.py
import pandas as pd
import pytest
from drybean.labels import clean_label_series

def test_clean_label_series_repairs_known_pollution():
    raw = pd.Series(["dermason", "DERMASON ", "D3RMAS0N", "H0R0Z", "S3K3R"])
    assert clean_label_series(raw).tolist() == [
        "DERMASON", "DERMASON", "DERMASON", "HOROZ", "SEKER"
    ]

def test_clean_label_series_rejects_unknown_label():
    with pytest.raises(ValueError, match="UNKNOWN"):
        clean_label_series(pd.Series(["UNKNOWN"]))
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_labels.py -v`

Expected: FAIL，原因是 `drybean.labels` 尚不存在。

- [ ] **Step 3: 实现合法类别和确定性映射**

```python
# src/drybean/labels.py
import pandas as pd

VALID_CLASSES = (
    "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"
)

def normalize_label(value: object) -> str:
    label = str(value).strip().upper().replace("0", "O").replace("3", "E")
    if label not in VALID_CLASSES:
        raise ValueError(f"无法识别的标签: {value!r} -> {label!r}")
    return label

def clean_label_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_label).astype("category")
```

- [ ] **Step 4: 实现三份数据加载和列检查**

```python
# src/drybean/data.py
from dataclasses import dataclass
import pandas as pd
from drybean.config import PATHS, RAW_FEATURES, TARGET
from drybean.labels import clean_label_series

@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

def _read_split(filename: str) -> pd.DataFrame:
    frame = pd.read_csv(PATHS.data_dir / filename)
    expected = [*RAW_FEATURES, TARGET]
    if frame.columns.tolist() != expected:
        raise ValueError(f"{filename} 列结构不符合预期")
    frame[TARGET] = clean_label_series(frame[TARGET])
    return frame

def load_splits() -> DatasetSplits:
    return DatasetSplits(
        train=_read_split("Dry_Bean_Dataset_Dirty_train.csv"),
        val=_read_split("Dry_Bean_Dataset_Dirty_val.csv"),
        test=_read_split("Dry_Bean_Dataset_Dirty_test.csv"),
    )
```

- [ ] **Step 5: 运行测试并提交**

Run: `pytest tests/test_labels.py -v`

Expected: 2 passed。

```bash
git add src/drybean/labels.py src/drybean/data.py tests/test_labels.py
git commit -m "feat: load splits and normalize polluted labels"
```

## Task 3: 实现数据画像和污染报告

**Files:**
- Modify: `src/drybean/data.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: 编写画像输出测试**

```python
# tests/test_metrics.py
import pandas as pd
from drybean.data import profile_frame

def test_profile_frame_reports_quality_counts():
    frame = pd.DataFrame({"x": [1.0, None, 1.0], "Class": ["A", "B", "A"]})
    report = profile_frame(frame)
    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["missing"]["x"] == 1
    assert report["duplicate_rows"] == 1
```

- [ ] **Step 2: 运行单测并确认失败**

Run: `pytest tests/test_metrics.py::test_profile_frame_reports_quality_counts -v`

Expected: FAIL，无法导入 `profile_frame`。

- [ ] **Step 3: 实现可序列化的数据画像**

```python
# append to src/drybean/data.py
def profile_frame(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing": {k: int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {
            str(k): int(v) for k, v in frame[TARGET].value_counts().items()
        } if TARGET in frame else {},
        "numeric_summary": frame.select_dtypes("number").describe().to_dict(),
    }
```

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/test_metrics.py::test_profile_frame_reports_quality_counts -v`

Expected: PASS。

```bash
git add src/drybean/data.py tests/test_metrics.py
git commit -m "feat: add serializable data quality profile"
```

## Task 4: 实现衍生特征

**Files:**
- Create: `tests/test_features.py`
- Create: `src/drybean/features.py`

- [ ] **Step 1: 编写特征计算测试**

```python
# tests/test_features.py
import pandas as pd
from drybean.features import add_engineered_features

def test_add_engineered_features_uses_geometric_definitions():
    frame = pd.DataFrame({
        "Area": [80.0], "ConvexArea": [100.0],
        "MajorAxisLength": [12.0], "MinorAxisLength": [8.0],
        "Perimeter": [40.0], "EquivDiameter": [10.0],
    })
    result = add_engineered_features(frame)
    assert result.loc[0, "ConvexGap"] == 20.0
    assert result.loc[0, "AxisLengthDiff"] == 4.0
    assert result.loc[0, "AreaConvexRatio"] == 0.8
    assert result.loc[0, "PerimeterDiameterRatio"] == 4.0
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_features.py -v`

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现无副作用的特征工程函数**

```python
# src/drybean/features.py
import numpy as np
import pandas as pd

ENGINEERED_FEATURES = (
    "ConvexGap", "AxisLengthDiff", "AreaConvexRatio", "PerimeterDiameterRatio"
)

def _safe_divide(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = right.replace(0, np.nan)
    return left.div(denominator)

def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["ConvexGap"] = result["ConvexArea"] - result["Area"]
    result["AxisLengthDiff"] = result["MajorAxisLength"] - result["MinorAxisLength"]
    result["AreaConvexRatio"] = _safe_divide(result["Area"], result["ConvexArea"])
    result["PerimeterDiameterRatio"] = _safe_divide(
        result["Perimeter"], result["EquivDiameter"]
    )
    return result
```

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/test_features.py -v`

Expected: PASS。

```bash
git add src/drybean/features.py tests/test_features.py
git commit -m "feat: add geometric feature engineering"
```

## Task 5: 实现防泄漏预处理

**Files:**
- Create: `tests/test_preprocessing.py`
- Create: `src/drybean/preprocessing.py`

- [ ] **Step 1: 编写训练集拟合和列顺序测试**

```python
# tests/test_preprocessing.py
import numpy as np
import pandas as pd
from drybean.preprocessing import TabularPreprocessor

def test_preprocessor_learns_imputation_only_from_train():
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 4.0, 6.0]})
    test = pd.DataFrame({"a": [100.0, np.nan], "b": [8.0, 10.0]})
    processor = TabularPreprocessor(scale=False, clip_quantiles=None)
    processor.fit(train)
    transformed = processor.transform(test)
    assert transformed.loc[1, "a"] == 2.0
    assert transformed.columns.tolist() == ["a", "b"]

def test_preprocessor_rejects_changed_columns():
    processor = TabularPreprocessor(scale=False, clip_quantiles=None)
    processor.fit(pd.DataFrame({"a": [1.0], "b": [2.0]}))
    try:
        processor.transform(pd.DataFrame({"b": [2.0], "a": [1.0], "c": [3.0]}))
    except ValueError as exc:
        assert "特征列" in str(exc)
    else:
        raise AssertionError("应拒绝不同特征列")
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_preprocessing.py -v`

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现中位数、分位数截尾和可选标准化**

```python
# src/drybean/preprocessing.py
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
        self.medians_ = frame.median(numeric_only=True)
        filled = frame.fillna(self.medians_)
        if self.clip_quantiles is None:
            self.lower_ = self.upper_ = None
        else:
            low, high = self.clip_quantiles
            self.lower_ = filled.quantile(low)
            self.upper_ = filled.quantile(high)
            filled = filled.clip(self.lower_, self.upper_, axis=1)
        self.scaler_ = StandardScaler().fit(filled) if self.scale else None
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.columns.tolist() != self.columns_:
            raise ValueError("特征列或顺序与训练集不一致")
        result = frame.replace([np.inf, -np.inf], np.nan).fillna(self.medians_)
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
```

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/test_preprocessing.py -v`

Expected: 2 passed。

```bash
git add src/drybean/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add leakage-safe tabular preprocessing"
```

## Task 6: 手写高斯朴素贝叶斯

**Files:**
- Create: `tests/test_gaussian_nb.py`
- Create: `src/drybean/models/gaussian_nb.py`

- [ ] **Step 1: 编写拟合、预测和概率测试**

```python
# tests/test_gaussian_nb.py
import numpy as np
from drybean.models.gaussian_nb import HandmadeGaussianNB

def test_handmade_gaussian_nb_fits_and_predicts_separable_data():
    x = np.array([[0.0, 0.1], [0.2, 0.0], [5.0, 5.1], [5.2, 4.9]])
    y = np.array(["A", "A", "B", "B"])
    model = HandmadeGaussianNB(var_smoothing=1e-9).fit(x, y)
    assert model.predict(np.array([[0.1, 0.1], [5.1, 5.0]])).tolist() == ["A", "B"]
    probabilities = model.predict_proba(np.array([[0.1, 0.1]]))
    assert probabilities.shape == (1, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_gaussian_nb.py -v`

Expected: FAIL，分类器不存在。

- [ ] **Step 3: 实现对数高斯后验**

```python
# src/drybean/models/gaussian_nb.py
import numpy as np

class HandmadeGaussianNB:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        self.class_prior_ = counts / counts.sum()
        self.theta_ = np.vstack([x[y == label].mean(axis=0) for label in self.classes_])
        variances = np.vstack([x[y == label].var(axis=0) for label in self.classes_])
        epsilon = self.var_smoothing * np.var(x, axis=0).max()
        self.var_ = variances + epsilon
        return self

    def _joint_log_likelihood(self, x):
        x = np.asarray(x, dtype=float)
        results = []
        for index, _ in enumerate(self.classes_):
            prior = np.log(self.class_prior_[index])
            normalization = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[index]))
            distance = -0.5 * np.sum(
                ((x - self.theta_[index]) ** 2) / self.var_[index], axis=1
            )
            results.append(prior + normalization + distance)
        return np.column_stack(results)

    def predict(self, x):
        scores = self._joint_log_likelihood(x)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, x):
        scores = self._joint_log_likelihood(x)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
```

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/test_gaussian_nb.py -v`

Expected: PASS。

```bash
git add src/drybean/models/gaussian_nb.py tests/test_gaussian_nb.py
git commit -m "feat: implement gaussian naive bayes from scratch"
```

## Task 7: 建立五模型注册表和 Loss 记录接口

**Files:**
- Create: `src/drybean/models/registry.py`
- Modify: `src/drybean/models/__init__.py`

- [ ] **Step 1: 实现模型规格对象**

```python
# src/drybean/models/registry.py
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from drybean.config import RANDOM_SEED
from drybean.models.gaussian_nb import HandmadeGaussianNB

@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    scale: bool
    iterative_loss: bool

def build_model_specs() -> dict[str, ModelSpec]:
    return {
        "logistic_regression": ModelSpec(
            "逻辑回归",
            SGDClassifier(
                loss="log_loss", penalty="l2", alpha=1e-4,
                max_iter=1, tol=None, random_state=RANDOM_SEED,
            ),
            True, True,
        ),
        "knn": ModelSpec(
            "KNN", KNeighborsClassifier(n_neighbors=7, weights="distance"), True, False
        ),
        "random_forest": ModelSpec(
            "随机森林",
            RandomForestClassifier(
                n_estimators=300, max_features="sqrt",
                n_jobs=-1, random_state=RANDOM_SEED,
            ),
            False, False,
        ),
        "xgboost": ModelSpec(
            "XGBoost",
            XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=RANDOM_SEED, n_jobs=-1,
            ),
            False, True,
        ),
        "handmade_gnb": ModelSpec(
            "手写高斯朴素贝叶斯", HandmadeGaussianNB(), False, False
        ),
    }
```

- [ ] **Step 2: 暴露公共接口并验证五模型**

```python
# src/drybean/models/__init__.py
from drybean.models.registry import ModelSpec, build_model_specs

__all__ = ["ModelSpec", "build_model_specs"]
```

Run: `python -c "from drybean.models import build_model_specs; assert len(build_model_specs()) == 5"`

Expected: 无输出且退出码为 0。

- [ ] **Step 3: 提交模型注册表**

```bash
git add src/drybean/models
git commit -m "feat: register five classification models"
```

## Task 8: 实现统一指标、计时和模型文件大小

**Files:**
- Modify: `tests/test_metrics.py`
- Create: `src/drybean/metrics.py`

- [ ] **Step 1: 增加指标结构测试**

```python
# append to tests/test_metrics.py
from drybean.metrics import classification_metrics

def test_classification_metrics_contains_required_scores():
    result = classification_metrics(["A", "A", "B"], ["A", "B", "B"])
    assert set(result) == {"accuracy", "macro_f1", "macro_precision", "macro_recall"}
    assert 0.0 <= result["accuracy"] <= 1.0
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_metrics.py::test_classification_metrics_contains_required_scores -v`

Expected: FAIL，`drybean.metrics` 不存在。

- [ ] **Step 3: 实现指标和稳定计时**

```python
# src/drybean/metrics.py
from pathlib import Path
from statistics import median
from time import perf_counter
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def classification_metrics(y_true, y_pred) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }

def median_inference_seconds(model, x, repeats: int = 30) -> float:
    model.predict(x[: min(len(x), 32)])
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        model.predict(x)
        samples.append(perf_counter() - start)
    return float(median(samples))

def serialized_model_size_mb(model, path: Path) -> float:
    joblib.dump(model, path)
    return path.stat().st_size / (1024 ** 2)
```

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/test_metrics.py -v`

Expected: 全部通过。

```bash
git add src/drybean/metrics.py tests/test_metrics.py
git commit -m "feat: add model quality and efficiency metrics"
```

## Task 9: 实现基线训练与最终评价

**Files:**
- Create: `src/drybean/experiments/baseline.py`

- [ ] **Step 1: 实现统一数据准备**

```python
# initial section of src/drybean/experiments/baseline.py
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from drybean.config import PATHS, RANDOM_SEED, RAW_FEATURES, TARGET
from drybean.data import load_splits
from drybean.features import add_engineered_features
from drybean.metrics import (
    classification_metrics, median_inference_seconds, serialized_model_size_mb
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

def prepare_data(engineered: bool, scale: bool) -> tuple[PreparedSplit, TabularPreprocessor]:
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
    encoder = LabelEncoder().fit(train[TARGET])
    prepared = PreparedSplit(
        x_train, x_val, x_test,
        encoder.transform(train[TARGET]),
        encoder.transform(splits.val[TARGET]),
        encoder.transform(splits.test[TARGET]),
        encoder,
    )
    return prepared, processor
```

- [ ] **Step 2: 实现逻辑回归逐轮 Loss**

```python
def fit_logistic_with_history(model, data: PreparedSplit, epochs: int = 120):
    history = []
    classes = np.arange(len(data.encoder.classes_))
    for epoch in range(epochs):
        order = np.random.default_rng(RANDOM_SEED + epoch).permutation(len(data.y_train))
        model.partial_fit(data.x_train.iloc[order], data.y_train[order], classes=classes)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(log_loss(data.y_train, model.predict_proba(data.x_train))),
            "val_loss": float(log_loss(data.y_val, model.predict_proba(data.x_val))),
        })
    return model, history
```

- [ ] **Step 3: 实现单模型训练与结果记录**

```python
def train_one(model_key: str, engineered: bool = True) -> tuple[dict, list[dict]]:
    spec = build_model_specs()[model_key]
    data, processor = prepare_data(engineered=engineered, scale=spec.scale)
    model = spec.estimator
    started = perf_counter()
    history: list[dict] = []
    if model_key == "logistic_regression":
        model, history = fit_logistic_with_history(model, data)
    elif model_key == "xgboost":
        model.fit(
            data.x_train, data.y_train,
            eval_set=[(data.x_train, data.y_train), (data.x_val, data.y_val)],
            verbose=False,
        )
        evals = model.evals_result()
        history = [
            {"epoch": i + 1, "train_loss": train, "val_loss": val}
            for i, (train, val) in enumerate(zip(
                evals["validation_0"]["mlogloss"],
                evals["validation_1"]["mlogloss"],
            ))
        ]
    else:
        model.fit(data.x_train, data.y_train)
    train_seconds = perf_counter() - started
    train_pred = model.predict(data.x_train)
    test_pred = model.predict(data.x_test)
    train_scores = classification_metrics(data.y_train, train_pred)
    test_scores = classification_metrics(data.y_test, test_pred)
    bundle = {"model": model, "processor": processor, "encoder": data.encoder}
    model_path = PATHS.models / f"{model_key}.joblib"
    result = {
        "model_key": model_key,
        "model": spec.name,
        **{f"train_{k}": v for k, v in train_scores.items()},
        **{f"test_{k}": v for k, v in test_scores.items()},
        "accuracy_gap": train_scores["accuracy"] - test_scores["accuracy"],
        "macro_f1_gap": train_scores["macro_f1"] - test_scores["macro_f1"],
        "train_seconds": train_seconds,
        "inference_seconds": median_inference_seconds(model, data.x_test),
        "inference_ms_per_sample": (
            median_inference_seconds(model, data.x_test) * 1000 / len(data.x_test)
        ),
        "model_size_mb": serialized_model_size_mb(bundle, model_path),
        "engineered_features": engineered,
    }
    pd.DataFrame(
        classification_report(
            data.y_test, test_pred,
            target_names=data.encoder.classes_, output_dict=True, zero_division=0,
        )
    ).transpose().to_csv(PATHS.results / f"class_report_{model_key}.csv")
    pd.DataFrame(
        confusion_matrix(data.y_test, test_pred),
        index=data.encoder.classes_, columns=data.encoder.classes_,
    ).to_csv(PATHS.results / f"confusion_{model_key}.csv")
    if hasattr(model, "feature_importances_"):
        pd.DataFrame({
            "feature": data.x_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).to_csv(
            PATHS.results / f"feature_importance_{model_key}.csv", index=False
        )
    return result, history
```

- [ ] **Step 4: 实现五模型批量运行和结构化落盘**

```python
def run_baseline_experiment(engineered: bool = True) -> pd.DataFrame:
    rows, histories = [], []
    for model_key in build_model_specs():
        result, history = train_one(model_key, engineered=engineered)
        rows.append(result)
        histories.extend({"model_key": model_key, **item} for item in history)
    results = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)
    results.to_csv(PATHS.results / "model_comparison.csv", index=False)
    pd.DataFrame(histories).to_csv(PATHS.results / "loss_history.csv", index=False)
    return results
```

- [ ] **Step 5: 运行一次烟雾训练**

Run: `python -c "from drybean.config import ensure_artifact_dirs; ensure_artifact_dirs(); from drybean.experiments.baseline import train_one; print(train_one('handmade_gnb')[0]['test_accuracy'])"`

Expected: 输出 0 到 1 之间的准确率，并生成 `artifacts/models/handmade_gnb.joblib`。

- [ ] **Step 6: 提交基线实验**

```bash
git add src/drybean/experiments/baseline.py
git commit -m "feat: add unified five-model baseline experiment"
```

## Task 10: 实现四类训练噪声

**Files:**
- Create: `tests/test_noise.py`
- Create: `src/drybean/noise.py`

- [ ] **Step 1: 编写零强度和输入不变测试**

```python
# tests/test_noise.py
import numpy as np
from drybean.noise import add_feature_noise, flip_labels

def test_zero_noise_returns_equal_copy():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = add_feature_noise(x, "gaussian", 0.0, seed=42)
    assert np.array_equal(result, x)
    assert result is not x

def test_noise_does_not_mutate_original():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    original = x.copy()
    add_feature_noise(x, "missing", 0.5, seed=42)
    assert np.array_equal(x, original)

def test_label_noise_keeps_valid_class_range():
    y = np.array([0, 1, 2, 0, 1, 2])
    noisy = flip_labels(y, 0.5, n_classes=3, seed=42)
    assert set(noisy).issubset({0, 1, 2})
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_noise.py -v`

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现高斯、脉冲、缺失和标签噪声**

```python
# src/drybean/noise.py
import numpy as np

def add_feature_noise(x, noise_type: str, strength: float, seed: int):
    result = np.asarray(x, dtype=float).copy()
    if strength == 0:
        return result
    rng = np.random.default_rng(seed)
    if noise_type == "gaussian":
        scale = np.nanstd(result, axis=0, keepdims=True)
        result += rng.normal(0.0, strength, size=result.shape) * scale
    elif noise_type == "impulse":
        mask = rng.random(result.shape) < strength
        low = np.nanquantile(result, 0.001, axis=0)
        high = np.nanquantile(result, 0.999, axis=0)
        extremes = rng.choice([-1.0, 1.0], size=result.shape)
        result[mask] = np.where(extremes[mask] < 0, np.broadcast_to(low, result.shape)[mask],
                                np.broadcast_to(high, result.shape)[mask])
    elif noise_type == "missing":
        result[rng.random(result.shape) < strength] = np.nan
    else:
        raise ValueError(f"未知特征噪声: {noise_type}")
    return result

def flip_labels(y, strength: float, n_classes: int, seed: int):
    result = np.asarray(y).copy()
    if strength == 0:
        return result
    rng = np.random.default_rng(seed)
    mask = rng.random(len(result)) < strength
    offsets = rng.integers(1, n_classes, size=mask.sum())
    result[mask] = (result[mask] + offsets) % n_classes
    return result
```

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/test_noise.py -v`

Expected: 3 passed。

```bash
git add src/drybean/noise.py tests/test_noise.py
git commit -m "feat: add reproducible training noise injectors"
```

## Task 11: 实现鲁棒性与特征消融实验

**Files:**
- Create: `src/drybean/experiments/robustness.py`
- Create: `src/drybean/experiments/ablation.py`

- [ ] **Step 1: 定义固定噪声矩阵**

```python
# src/drybean/experiments/robustness.py
NOISE_LEVELS = {
    "gaussian": (0.05, 0.10, 0.20),
    "impulse": (0.01, 0.03, 0.05),
    "missing": (0.05, 0.10, 0.20),
    "label": (0.05, 0.10, 0.20),
}
```

- [ ] **Step 2: 实现按噪声重新拟合与干净测试**

```python
# append to src/drybean/experiments/robustness.py
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from drybean.config import PATHS, RANDOM_SEED, TARGET
from drybean.data import load_splits
from drybean.features import add_engineered_features
from drybean.metrics import classification_metrics
from drybean.models import build_model_specs
from drybean.noise import add_feature_noise, flip_labels
from drybean.preprocessing import TabularPreprocessor

def fit_robust_model(model_key, model, x, y, n_classes):
    if model_key == "logistic_regression":
        classes = np.arange(n_classes)
        for epoch in range(120):
            order = np.random.default_rng(RANDOM_SEED + epoch).permutation(len(y))
            model.partial_fit(x.iloc[order], y[order], classes=classes)
        return model
    model.fit(x, y)
    return model

def run_robustness_experiment() -> pd.DataFrame:
    rows = []
    for model_key, spec in build_model_specs().items():
        splits = load_splits()
        train = splits.train.drop_duplicates().reset_index(drop=True)
        raw_x_train = add_engineered_features(train.drop(columns=TARGET))
        raw_x_test = add_engineered_features(splits.test.drop(columns=TARGET))
        encoder = LabelEncoder().fit(train[TARGET])
        y_train = encoder.transform(train[TARGET])
        y_test = encoder.transform(splits.test[TARGET])
        clean_processor = TabularPreprocessor(scale=spec.scale)
        clean_x_train = clean_processor.fit_transform(raw_x_train)
        clean_x_test = clean_processor.transform(raw_x_test)
        clean_model = build_model_specs()[model_key].estimator
        clean_model = fit_robust_model(
            model_key, clean_model, clean_x_train, y_train, len(encoder.classes_)
        )
        clean_accuracy = classification_metrics(
            y_test, clean_model.predict(clean_x_test)
        )["accuracy"]
        for noise_type, levels in NOISE_LEVELS.items():
            for strength in levels:
                noisy_x = raw_x_train.to_numpy(copy=True)
                noisy_y = y_train.copy()
                if noise_type == "label":
                    noisy_y = flip_labels(
                        noisy_y, strength, len(encoder.classes_), RANDOM_SEED
                    )
                else:
                    noisy_x = add_feature_noise(
                        noisy_x, noise_type, strength, RANDOM_SEED
                    )
                noisy_x = pd.DataFrame(noisy_x, columns=raw_x_train.columns)
                noisy_processor = TabularPreprocessor(scale=spec.scale)
                processed_noisy_x = noisy_processor.fit_transform(noisy_x)
                processed_clean_test = noisy_processor.transform(raw_x_test)
                model = build_model_specs()[model_key].estimator
                started = perf_counter()
                model = fit_robust_model(
                    model_key, model, processed_noisy_x, noisy_y, len(encoder.classes_)
                )
                scores = classification_metrics(
                    y_test, model.predict(processed_clean_test)
                )
                rows.append({
                    "model_key": model_key,
                    "model": spec.name,
                    "noise_type": noise_type,
                    "strength": strength,
                    "test_accuracy": scores["accuracy"],
                    "test_macro_f1": scores["macro_f1"],
                    "accuracy_drop": clean_accuracy - scores["accuracy"],
                    "train_seconds": perf_counter() - started,
                })
    result = pd.DataFrame(rows)
    result.to_csv(PATHS.results / "robustness.csv", index=False)
    return result
```

- [ ] **Step 3: 实现原始与衍生特征消融**

```python
# src/drybean/experiments/ablation.py
import pandas as pd
from drybean.config import PATHS
from drybean.experiments.baseline import train_one
from drybean.models import build_model_specs

def run_feature_ablation() -> pd.DataFrame:
    rows = []
    for engineered in (False, True):
        for model_key in build_model_specs():
            result, _ = train_one(model_key, engineered=engineered)
            rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(PATHS.results / "feature_ablation.csv", index=False)
    return frame
```

- [ ] **Step 4: 运行单模型消融烟雾检查**

Run: `python -c "from drybean.config import ensure_artifact_dirs; ensure_artifact_dirs(); from drybean.experiments.baseline import train_one; a=train_one('knn', False)[0]; b=train_one('knn', True)[0]; print(a['test_accuracy'], b['test_accuracy'])"`

Expected: 输出两个 0 到 1 之间的值。

- [ ] **Step 5: 提交扩展实验**

```bash
git add src/drybean/experiments/robustness.py src/drybean/experiments/ablation.py
git commit -m "feat: add robustness and feature ablation experiments"
```

## Task 12: 生成论文与网页共用图表

**Files:**
- Create: `src/drybean/plots.py`

- [ ] **Step 1: 实现统一绘图样式和保存函数**

```python
# src/drybean/plots.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from drybean.config import PATHS

COLORS = ["#2C6E63", "#D28C45", "#4C78A8", "#A65D57", "#7568A6"]

def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False

def save_figure(fig, filename: str) -> Path:
    path = PATHS.figures / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
```

- [ ] **Step 2: 实现模型性能、效率和过拟合图**

```python
def plot_model_comparison(frame: pd.DataFrame) -> None:
    configure_style()
    ordered = frame.sort_values("test_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=ordered, x="test_accuracy", y="model", palette=COLORS, ax=ax)
    ax.set(xlabel="测试集 Accuracy", ylabel="", xlim=(0.7, 1.0))
    save_figure(fig, "model_accuracy.png")

    melted = frame.melt(
        id_vars="model",
        value_vars=["train_seconds", "inference_ms_per_sample", "model_size_mb"],
        var_name="dimension", value_name="value",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, dimension in zip(axes, melted["dimension"].unique()):
        subset = melted[melted["dimension"] == dimension]
        sns.barplot(data=subset, x="value", y="model", ax=ax, color=COLORS[0])
        ax.set(ylabel="", title=dimension)
    save_figure(fig, "model_efficiency.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = frame.sort_values("accuracy_gap", ascending=False)
    sns.barplot(data=ordered, x="accuracy_gap", y="model", ax=ax, color=COLORS[1])
    ax.set(xlabel="训练 Accuracy - 测试 Accuracy", ylabel="")
    save_figure(fig, "overfitting_gap.png")
```

- [ ] **Step 3: 实现 Loss、鲁棒性和消融图**

```python
def plot_loss_history(frame: pd.DataFrame) -> None:
    configure_style()
    for model_key, subset in frame.groupby("model_key"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(subset["epoch"], subset["train_loss"], label="训练集")
        ax.plot(subset["epoch"], subset["val_loss"], label="验证集")
        ax.set(xlabel="迭代轮数", ylabel="Log Loss", title=f"{model_key} Loss 曲线")
        ax.legend()
        save_figure(fig, f"loss_{model_key}.png")

def plot_robustness(frame: pd.DataFrame) -> None:
    configure_style()
    for noise_type, subset in frame.groupby("noise_type"):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=subset, x="strength", y="accuracy_drop",
            hue="model", marker="o", ax=ax,
        )
        ax.set(xlabel="噪声强度", ylabel="Accuracy 下降", title=noise_type)
        save_figure(fig, f"robustness_{noise_type}.png")

def plot_ablation(frame: pd.DataFrame) -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=frame, x="test_accuracy", y="model",
        hue="engineered_features", ax=ax,
    )
    ax.set(xlabel="测试集 Accuracy", ylabel="")
    save_figure(fig, "feature_ablation.png")

def plot_confusion_and_importance() -> None:
    configure_style()
    for path in PATHS.results.glob("confusion_*.csv"):
        matrix = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt="g", cmap="YlGnBu", ax=ax)
        ax.set(xlabel="预测类别", ylabel="真实类别")
        save_figure(fig, f"{path.stem}.png")
    for path in PATHS.results.glob("feature_importance_*.csv"):
        importance = pd.read_csv(path).head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=importance, x="importance", y="feature", ax=ax, color=COLORS[0])
        ax.set(xlabel="特征重要性", ylabel="")
        save_figure(fig, f"{path.stem}.png")
```

- [ ] **Step 4: 提交绘图模块**

```bash
git add src/drybean/plots.py
git commit -m "feat: add publication-ready experiment charts"
```

## Task 13: 实现统一 CLI

**Files:**
- Create: `tests/test_cli.py`
- Create: `src/drybean/cli.py`

- [ ] **Step 1: 编写命令存在性测试**

```python
# tests/test_cli.py
from typer.testing import CliRunner
from drybean.cli import app

runner = CliRunner()

def test_cli_exposes_required_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("analyze", "train", "ablation", "robustness", "plot", "all"):
        assert command in result.stdout
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL，CLI 不存在。

- [ ] **Step 3: 实现阶段命令**

```python
# src/drybean/cli.py
import json
import pandas as pd
import typer
from drybean.config import PATHS, ensure_artifact_dirs
from drybean.data import load_splits, profile_frame
from drybean.experiments.ablation import run_feature_ablation
from drybean.experiments.baseline import run_baseline_experiment
from drybean.experiments.robustness import run_robustness_experiment
from drybean.plots import (
    plot_ablation, plot_confusion_and_importance, plot_loss_history,
    plot_model_comparison, plot_robustness
)

app = typer.Typer(help="Dry Bean 多分类实验命令行")

@app.command()
def analyze():
    ensure_artifact_dirs()
    splits = load_splits()
    report = {name: profile_frame(getattr(splits, name)) for name in ("train", "val", "test")}
    (PATHS.results / "data_profile.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

@app.command()
def train():
    ensure_artifact_dirs()
    run_baseline_experiment(engineered=True)

@app.command()
def ablation():
    ensure_artifact_dirs()
    run_feature_ablation()

@app.command()
def robustness():
    ensure_artifact_dirs()
    run_robustness_experiment()

@app.command()
def plot():
    ensure_artifact_dirs()
    plot_model_comparison(pd.read_csv(PATHS.results / "model_comparison.csv"))
    loss_path = PATHS.results / "loss_history.csv"
    if loss_path.exists() and loss_path.stat().st_size:
        plot_loss_history(pd.read_csv(loss_path))
    plot_robustness(pd.read_csv(PATHS.results / "robustness.csv"))
    plot_ablation(pd.read_csv(PATHS.results / "feature_ablation.csv"))
    plot_confusion_and_importance()

@app.command(name="all")
def run_all():
    analyze()
    train()
    ablation()
    robustness()
    plot()

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: 运行测试并提交**

Run: `pytest tests/test_cli.py -v`

Expected: PASS。

```bash
git add src/drybean/cli.py tests/test_cli.py
git commit -m "feat: add unified experiment command line"
```

## Task 14: 运行完整实验并核验结果

**Files:**
- Generate: `artifacts/results/data_profile.json`
- Generate: `artifacts/results/model_comparison.csv`
- Generate: `artifacts/results/loss_history.csv`
- Generate: `artifacts/results/feature_ablation.csv`
- Generate: `artifacts/results/robustness.csv`
- Generate: `artifacts/figures/*.png`
- Generate: `artifacts/models/*.joblib`

- [ ] **Step 1: 运行全部自动测试**

Run: `pytest -v`

Expected: 所有测试通过。

- [ ] **Step 2: 运行完整实验**

Run: `drybean all`

Expected: 命令退出码为 0，五种模型均完成训练，结果与图表文件全部生成。

- [ ] **Step 3: 检查结果完整性**

Run:

```powershell
@'
import pandas as pd
from pathlib import Path
root = Path("artifacts")
comparison = pd.read_csv(root / "results/model_comparison.csv")
robustness = pd.read_csv(root / "results/robustness.csv")
ablation = pd.read_csv(root / "results/feature_ablation.csv")
assert comparison["model_key"].nunique() == 5
assert set(robustness["noise_type"]) == {"gaussian", "impulse", "missing", "label"}
assert robustness.groupby(["model_key", "noise_type"]).size().eq(3).all()
assert set(ablation["engineered_features"]) == {False, True}
assert len(list((root / "models").glob("*.joblib"))) == 5
assert len(list((root / "figures").glob("*.png"))) >= 9
print(comparison[["model", "test_accuracy", "test_macro_f1"]])
'@ | python -
```

Expected: 所有断言通过并打印五模型测试结果。

- [ ] **Step 4: 记录环境和复现信息**

Run: `python -m pip freeze > artifacts/results/environment_freeze.txt`

Expected: 文件非空。

- [ ] **Step 5: 提交可复现实验产物**

```bash
git add artifacts/results artifacts/figures
git commit -m "results: add reproducible dry bean experiments"
```

模型二进制文件默认不提交 Git，README 中说明运行命令会本地生成。

## Task 15: 构建 Streamlit 展示系统

**Files:**
- Create: `app/streamlit_app.py`

- [ ] **Step 1: 实现离线结果加载**

```python
# app/streamlit_app.py
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
FIGURES = ROOT / "artifacts" / "figures"

st.set_page_config(page_title="Dry Bean 多分类实验", page_icon="🫘", layout="wide")

@st.cache_data
def load_results():
    return {
        "profile": json.loads((RESULTS / "data_profile.json").read_text(encoding="utf-8")),
        "comparison": pd.read_csv(RESULTS / "model_comparison.csv"),
        "robustness": pd.read_csv(RESULTS / "robustness.csv"),
        "ablation": pd.read_csv(RESULTS / "feature_ablation.csv"),
    }

data = load_results()
```

- [ ] **Step 2: 实现完整展示章节**

```python
st.title("Dry Bean Dataset 多分类机器学习项目")
st.caption("数据分析 · 数据清洗 · 五算法对比 · 鲁棒性 · 过拟合 · 特征消融")

tabs = st.tabs([
    "项目概览", "数据处理", "算法对比", "鲁棒性",
    "过拟合与消融", "结论",
])

with tabs[0]:
    train = data["profile"]["train"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("训练样本", f"{train['rows']:,}")
    c2.metric("输入特征", 16)
    c3.metric("真实类别", 7)
    c4.metric("参与算法", 5)
    st.markdown("项目使用教师预先划分的训练、验证和测试集，测试集仅用于最终评价。")

with tabs[1]:
    st.subheader("发现的数据污染")
    st.json({
        "缺失列": {"Perimeter": train["missing"]["Perimeter"],
                  "Solidity": train["missing"]["Solidity"]},
        "训练集重复行": train["duplicate_rows"],
        "标签污染": "大小写、尾随空格、数字替字",
    })
    st.markdown("处理流程：标签规范化 → 去除训练重复 → 中位数填补 → 分位数截尾 → 按模型标准化。")

with tabs[2]:
    comparison = data["comparison"].copy()
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.image(str(FIGURES / "model_accuracy.png"), use_container_width=True)
    st.image(str(FIGURES / "model_efficiency.png"), use_container_width=True)
    for key in ("logistic_regression", "xgboost"):
        image = FIGURES / f"loss_{key}.png"
        if image.exists():
            st.image(str(image), use_container_width=True)

with tabs[3]:
    st.dataframe(data["robustness"], use_container_width=True, hide_index=True)
    for noise in ("gaussian", "impulse", "missing", "label"):
        st.image(str(FIGURES / f"robustness_{noise}.png"), use_container_width=True)

with tabs[4]:
    st.image(str(FIGURES / "overfitting_gap.png"), use_container_width=True)
    st.image(str(FIGURES / "feature_ablation.png"), use_container_width=True)
    st.dataframe(data["ablation"], use_container_width=True, hide_index=True)

with tabs[5]:
    best = data["comparison"].sort_values("test_accuracy", ascending=False).iloc[0]
    st.success(
        f"最高测试准确率模型为 {best['model']}，"
        f"Accuracy={best['test_accuracy']:.4f}，Macro-F1={best['test_macro_f1']:.4f}。"
    )
    st.markdown("最终结论同时考虑精度、鲁棒性、训练/推理成本和特征工程收益。")
```

- [ ] **Step 3: 启动并检查页面**

Run: `streamlit run app/streamlit_app.py`

Expected: 本地页面无异常，六个页签、结果表和所有图表均正常显示。

- [ ] **Step 4: 保存系统截图**

使用浏览器在 1440×1000 视口截取项目概览、算法对比、鲁棒性和消融页面，保存为：

```text
artifacts/screenshots/01_overview.png
artifacts/screenshots/02_model_comparison.png
artifacts/screenshots/03_robustness.png
artifacts/screenshots/04_ablation.png
```

- [ ] **Step 5: 提交展示系统**

```bash
git add app/streamlit_app.py artifacts/screenshots
git commit -m "feat: add streamlit experiment showcase"
```

## Task 16: 编写 GitHub README

**Files:**
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: 配置忽略项**

```gitignore
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
artifacts/models/
.streamlit/
.superpowers/
~$*
```

- [ ] **Step 2: 编写结果驱动的 README**

README 必须按此顺序包含实际内容：

```markdown
# Dry Bean Dataset 多分类机器学习项目

一句话项目结论和最佳模型实际测试结果。

## 项目亮点
- 三类原始数据污染及对应处理
- 五种算法，含两个课堂外算法和一个手写算法
- 评分要求的五类实验
- 四项额外对比维度

## 数据集
训练/验证/测试样本数、16 个特征、7 个类别及污染统计。

## 数据处理
标签清洗、重复处理、缺失填补、异常值、标准化、衍生特征和防泄漏说明。

## 算法
逻辑回归、KNN、随机森林、XGBoost、手写高斯朴素贝叶斯。

## 核心实验结果
从 `artifacts/results/model_comparison.csv` 复制最终五模型表。

## 图表
嵌入 Accuracy、效率、Loss、鲁棒性、过拟合和消融图。

## 项目结构
列出源码、应用、测试、实验产物和论文位置。

## 快速开始
python -m pip install -e ".[dev]"
drybean all
streamlit run app/streamlit_app.py

## 实验复现说明
随机种子、测试集使用边界、硬件导致计时差异的说明。

## 论文与系统展示
论文路径、系统截图，以及发布后补充的实际 GitHub/Streamlit 链接。
```

- [ ] **Step 3: 验证 README 中命令**

Run: `drybean --help`

Expected: 帮助文本列出全部阶段命令。

Run: `python -c "from pathlib import Path; text=Path('README.md').read_text(encoding='utf-8'); assert len(text) > 2000"`

Expected: 无输出且退出码为 0。

- [ ] **Step 4: 提交 README**

```bash
git add README.md .gitignore
git commit -m "docs: add github project showcase"
```

## Task 17: 生成 Word 课程论文

**Files:**
- Create: `scripts/build_paper.py`
- Create: `paper/paper_metadata.json`
- Generate: `paper/Dry_Bean_机器学习课程论文.docx`

- [ ] **Step 1: 读取文档设计规范**

执行前完整读取：

```text
C:/Users/霖/.codex/plugins/cache/openai-primary-runtime/documents/26.619.11828/skills/documents/references/design_presets.md
C:/Users/霖/.codex/plugins/cache/openai-primary-runtime/documents/26.619.11828/skills/documents/references/header_templates.md
C:/Users/霖/.codex/plugins/cache/openai-primary-runtime/documents/26.619.11828/skills/documents/tasks/create_edit.md
C:/Users/霖/.codex/plugins/cache/openai-primary-runtime/documents/26.619.11828/skills/documents/tasks/verify_render.md
```

选择正式报告对应的设计预设，并将页面、字体、标题、段落、表格和图注参数显式写入构建脚本。

- [ ] **Step 2: 实现论文数据读取与章节结构**

先创建可编辑的个人信息配置：

```json
{
  "course": "机器学习与项目实践",
  "title": "基于 Dry Bean Dataset 的多分类机器学习实验研究",
  "name": "",
  "student_id": "",
  "class_name": "",
  "date": "2026年6月"
}
```

```python
# opening section of scripts/build_paper.py
from pathlib import Path
import json
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
FIGURES = ROOT / "artifacts" / "figures"
SCREENSHOTS = ROOT / "artifacts" / "screenshots"
OUTPUT = ROOT / "paper" / "Dry_Bean_机器学习课程论文.docx"
metadata = json.loads(
    (ROOT / "paper" / "paper_metadata.json").read_text(encoding="utf-8")
)

comparison = pd.read_csv(RESULTS / "model_comparison.csv")
robustness = pd.read_csv(RESULTS / "robustness.csv")
ablation = pd.read_csv(RESULTS / "feature_ablation.csv")
profile = json.loads((RESULTS / "data_profile.json").read_text(encoding="utf-8"))
best = comparison.sort_values("test_accuracy", ascending=False).iloc[0]

doc = Document()
section = doc.sections[0]
section.top_margin = Cm(2.54)
section.bottom_margin = Cm(2.54)
section.left_margin = Cm(2.8)
section.right_margin = Cm(2.5)
```

- [ ] **Step 3: 生成全部论文内容**

构建脚本必须实际写入以下全部实质内容：

```text
封面：题目、课程、姓名、学号、班级、日期；个人信息从
`paper/paper_metadata.json` 读取，未提供的字段显示可编辑下划线
摘要：问题、数据污染、方法、五算法、最佳实际结果、结论
关键词：干豆分类、数据清洗、多分类、鲁棒性、特征工程
目录：Word TOC 字段
第1章 绪论
第2章 数据集与污染分析
第3章 数据清洗与特征工程
第4章 五种分类算法原理与实现
第5章 实验设计与评价指标
第6章 实验结果与额外对比
第7章 噪声鲁棒性与过拟合分析
第8章 系统设计和网页展示
第9章 课程总结
参考文献
```

正文数字必须通过 `comparison`、`robustness`、`ablation` 和 `profile` 变量格式化插入。表格必须使用实际结果；图 6-1 至图 7-N 使用 `artifacts/figures`；系统章节使用四张 Streamlit 截图。

- [ ] **Step 4: 设置标题样式、页码、图表标题和目录字段**

脚本中实现：

```python
styles = doc.styles
styles["Normal"].font.name = "宋体"
styles["Normal"].font.size = Pt(10.5)
for style_name, size in (("Heading 1", 16), ("Heading 2", 14), ("Heading 3", 12)):
    styles[style_name].font.name = "黑体"
    styles[style_name].font.size = Pt(size)
    styles[style_name].font.color.rgb = RGBColor(31, 78, 61)
```

同时用 OOXML 插入 TOC 和页码字段，为每张图和表添加连续编号与中文标题。所有模型结果表根据列内容设置不等宽列，允许自动换行，不设置固定行高。

- [ ] **Step 5: 生成 DOCX**

Run: `python scripts/build_paper.py`

Expected: `paper/Dry_Bean_机器学习课程论文.docx` 存在且文件大小大于 500 KB。

- [ ] **Step 6: 渲染论文进行逐页检查**

Run:

```powershell
python "C:\Users\霖\.codex\plugins\cache\openai-primary-runtime\documents\26.619.11828\skills\documents\render_docx.py" `
  "paper\Dry_Bean_机器学习课程论文.docx" `
  --output_dir "artifacts\paper_render" `
  --emit_pdf
```

Expected: 每页生成 `page-*.png`，并生成 PDF。

- [ ] **Step 7: 逐页视觉 QA 并修复**

逐页检查：

- 中文无乱码或缺字。
- 封面信息不拥挤。
- 目录、章标题和正文分页自然。
- 图表不跨页截断，图题与图片相邻。
- 表格无溢出，表头清楚，数值小数位一致。
- 系统截图清晰可读。
- 页眉页脚与页码位置一致。

发现问题后修改 `scripts/build_paper.py`，重新执行 Step 5 至 Step 7，直到所有页面通过。

- [ ] **Step 8: 提交论文和构建脚本**

```bash
git add scripts/build_paper.py paper/Dry_Bean_机器学习课程论文.docx
git commit -m "docs: generate verified course paper"
```

## Task 18: 最终全链路验证和交付整理

**Files:**
- Modify: `README.md`
- Verify: all project files

- [ ] **Step 1: 运行完整测试**

Run: `pytest -v`

Expected: 全部通过。

- [ ] **Step 2: 从空结果目录验证可复现性**

先将现有 `artifacts/results` 和 `artifacts/figures` 暂存到工作区外，再运行：

Run: `drybean all`

Expected: 所有结果和图表重新生成；不得删除原始数据、论文或用户文件。

- [ ] **Step 3: 验证 Streamlit**

Run: `streamlit run app/streamlit_app.py --server.headless true`

Expected: 页面启动无 traceback；用浏览器检查关键页面和控制台。

- [ ] **Step 4: 核对评分覆盖**

逐项确认：

```text
数据分析：数据用途、特征、污染和类别分布
数据处理：标签、重复、缺失、异常值、标准化、衍生特征、防泄漏
算法：五种模型、至少一种课外、一个手写
对比：Accuracy、Loss、推理速度、鲁棒性、过拟合
额外：Macro-F1、训练时间、模型大小、特征消融
系统：模块化目录、统一 CLI、无训练 UI、Streamlit
展示：README、结果表、图表、截图
论文：上述内容全部成章，含网页链接位置
```

- [ ] **Step 5: 更新实际仓库与网页链接**

只有在用户提供或确认 GitHub 仓库后，才把实际 URL 写入 README 和论文。推送和公开发布属于外部写操作，在执行前确认目标仓库。

- [ ] **Step 6: 检查 Git 状态和最近提交**

Run: `git status --short`

Expected: 仅保留明确不提交的本地模型或原始教师文件；源码、结果、README 和论文均已跟踪。

Run: `git log --oneline -12`

Expected: 提交历史按数据、模型、实验、展示、论文分阶段清晰排列。

- [ ] **Step 7: 最终交付清单**

向用户交付以下路径：

```text
README.md
paper/Dry_Bean_机器学习课程论文.docx
app/streamlit_app.py
artifacts/results/model_comparison.csv
artifacts/results/robustness.csv
artifacts/results/feature_ablation.csv
artifacts/figures/
```
