import pandas as pd
import pytest

from src.config import (
    FEATURES_INGRESO,
    FEATURES_PROHIBIDAS,
    FEATURES_SEM1,
    FEATURES_SEM2,
    FEATURES_SENSIBILIDAD,
    TARGET_COL,
)
from src.features import feature_columns, select_features

ALL_GROUPS = FEATURES_INGRESO + FEATURES_SENSIBILIDAD + FEATURES_SEM1 + FEATURES_SEM2


@pytest.fixture
def df():
    columns = list(ALL_GROUPS) + [TARGET_COL]
    return pd.DataFrame([[0] * len(columns)], columns=columns)


def test_feature_groups_cover_36_columns_without_overlap():
    assert len(ALL_GROUPS) == 36
    assert len(set(ALL_GROUPS)) == 36


def test_prohibited_are_both_semesters():
    assert set(FEATURES_PROHIBIDAS) == set(FEATURES_SEM1) | set(FEATURES_SEM2)


def test_select_features_principal_uses_only_ingreso(df):
    X = select_features(df)
    assert list(X.columns) == list(FEATURES_INGRESO)


def test_select_features_never_returns_prohibited_or_target(df):
    for sensibilidad in (False, True):
        columns = set(select_features(df, sensibilidad=sensibilidad).columns)
        assert not columns & set(FEATURES_PROHIBIDAS)
        assert TARGET_COL not in columns


def test_select_features_sensitivity_adds_six(df):
    X = select_features(df, sensibilidad=True)
    assert len(X.columns) == len(FEATURES_INGRESO) + 6
    assert set(FEATURES_SENSIBILIDAD) <= set(X.columns)
    assert not set(FEATURES_SENSIBILIDAD) & set(feature_columns())


def test_select_features_missing_column_raises(df):
    with pytest.raises(ValueError, match="Admission grade"):
        select_features(df.drop(columns="Admission grade"))
