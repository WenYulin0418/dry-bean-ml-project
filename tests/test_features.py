import pandas as pd

from drybean.features import add_engineered_features


def test_add_engineered_features_uses_geometric_definitions():
    frame = pd.DataFrame(
        {
            "Area": [80.0],
            "ConvexArea": [100.0],
            "MajorAxisLength": [12.0],
            "MinorAxisLength": [8.0],
            "Perimeter": [40.0],
            "EquivDiameter": [10.0],
        }
    )

    result = add_engineered_features(frame)

    assert result.loc[0, "ConvexGap"] == 20.0
    assert result.loc[0, "AxisLengthDiff"] == 4.0
    assert result.loc[0, "AreaConvexRatio"] == 0.8
    assert result.loc[0, "PerimeterDiameterRatio"] == 4.0
    assert "ConvexGap" not in frame.columns

