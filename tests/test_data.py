import pandas as pd

from drybean.data import profile_frame


def test_profile_frame_reports_quality_counts():
    frame = pd.DataFrame(
        {"x": [1.0, None, 1.0], "Class": ["A", "B", "A"]}
    )

    report = profile_frame(frame)

    assert report["rows"] == 3
    assert report["columns"] == 2
    assert report["missing"]["x"] == 1
    assert report["duplicate_rows"] == 1

