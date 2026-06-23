import pandas as pd
import pytest

from drybean.labels import clean_label_series


def test_clean_label_series_repairs_known_pollution():
    raw = pd.Series(["dermason", "DERMASON ", "D3RMAS0N", "H0R0Z", "S3K3R"])

    assert clean_label_series(raw).astype(str).tolist() == [
        "DERMASON",
        "DERMASON",
        "DERMASON",
        "HOROZ",
        "SEKER",
    ]


def test_clean_label_series_rejects_unknown_label():
    with pytest.raises(ValueError, match="UNKNOWN"):
        clean_label_series(pd.Series(["UNKNOWN"]))

