import numpy as np
import pandas as pd
import pytest

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

    with pytest.raises(ValueError, match="特征列"):
        processor.transform(pd.DataFrame({"b": [2.0], "a": [1.0], "c": [3.0]}))

