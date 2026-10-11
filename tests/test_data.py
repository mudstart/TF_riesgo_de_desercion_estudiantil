import pandas as pd
import pytest

from src.data import build_target


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "Admission grade": [120.0, 135.5, 110.0, 142.3],
            "Target": ["Dropout", "Graduate", "Enrolled", "Dropout"],
        }
    )


def test_build_target_binary_maps_dropout_to_one(df):
    y = build_target(df)
    assert y.tolist() == [1, 0, 0, 1]


def test_build_target_binary_keeps_enrolled_rows(df):
    y = build_target(df)
    assert len(y) == len(df)
    assert y[df["Target"] == "Enrolled"].tolist() == [0]


def test_build_target_multiclass_keeps_original_classes(df):
    y = build_target(df, binary=False)
    assert y.tolist() == ["Dropout", "Graduate", "Enrolled", "Dropout"]


def test_build_target_does_not_modify_input(df):
    original = df.copy()
    build_target(df)
    build_target(df, binary=False)
    pd.testing.assert_frame_equal(df, original)


def test_build_target_missing_column_raises(df):
    with pytest.raises(ValueError, match="Target"):
        build_target(df.drop(columns="Target"))


def test_build_target_unknown_class_raises(df):
    df.loc[0, "Target"] = "Transfer"
    with pytest.raises(ValueError, match="Transfer"):
        build_target(df)
