import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import (
    FEATURES_BINARIAS,
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    FEATURES_REDUNDANTES,
)
from src.features import model_columns
from src.models.train_rf import SEARCH_SPACE, build_rf, importance_table, search

N = 200


@pytest.fixture
def data():
    """Train sintético con las columnas permitidas y señal en Age at enrollment."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            **{c: rng.choice([1, 2, 3], size=N) for c in FEATURES_NOMINALES},
            **{c: rng.normal(100, 15, size=N) for c in FEATURES_NUMERICAS},
            **{c: rng.integers(0, 2, size=N) for c in FEATURES_BINARIAS + FEATURES_REDUNDANTES},
        }
    )
    y = pd.Series((X["Age at enrollment"] > 100).astype(int))
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    return X, y, folds


def test_search_space_uses_valid_pipeline_parameters():
    valid = build_rf().get_params()
    assert set(SEARCH_SPACE) <= set(valid)


def test_random_forest_uses_balanced_class_weight():
    assert build_rf().named_steps["modelo"].class_weight == "balanced"


def test_search_returns_values_from_the_space(data):
    X, y, folds = data
    small = {**SEARCH_SPACE, "modelo__n_estimators": [10, 20]}
    best, table, elapsed = search(X, y, folds, n_iter=3, space=small)
    assert len(table) == 3
    assert table["f1_macro_media"].is_monotonic_decreasing
    for param, value in best.items():
        assert value in small[f"modelo__{param}"]
    assert elapsed > 0


def test_importance_has_one_row_per_model_column(data):
    X, y, folds = data
    table = importance_table(build_rf(n_estimators=10), X, y, folds)
    assert set(table.index) == set(model_columns())
    assert table["importancia_media"].is_monotonic_decreasing
    assert table.index[0] == "Age at enrollment"
