import pandas as pd

from drybean.data import coerce_numeric_features, profile_frame


def test_profile_frame_reports_quality_counts():
    frame = pd.DataFrame(
        {"x": [1.0, None, 1.0], "Class": ["A", "B", "A"]}
    )

    report = profile_frame(frame)

    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["missing"]["x"] == 1
    assert report["duplicate_rows"] == 1


def test_coerce_numeric_features_handles_sentinels_and_units():
    frame = pd.DataFrame(
        {
            "Solidity": ["?", "0.98"],
            "Compactness": ["0.7600 cm", "0.82"],
            "Class": ["A", "B"],
        }
    )

    result = coerce_numeric_features(frame, ["Solidity", "Compactness"])

    assert pd.isna(result.loc[0, "Solidity"])
    assert result.loc[0, "Compactness"] == 0.76
    assert frame.loc[0, "Compactness"] == "0.7600 cm"
