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
from src.models.train_gb import EARLY_STOPPING, SEARCH_SPACE, build_gb, early_stopping_iterations, search

N = 300


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
    assert set(SEARCH_SPACE) <= set(build_gb().get_params())


def test_gradient_boosting_uses_balanced_class_weight_and_early_stopping():
    model = build_gb().named_steps["modelo"]
    assert model.class_weight == "balanced"
    assert model.early_stopping is True
    assert model.validation_fraction == 0.1


def test_search_returns_values_from_the_space(data):
    X, y, folds = data
    best, table, _ = search(X, y, folds, n_iter=3)
    assert len(table) == 3
    for param, value in best.items():
        assert value in SEARCH_SPACE[f"modelo__{param}"]


def test_early_stopping_stops_before_max_iter(data):
    X, y, folds = data
    iterations = early_stopping_iterations(build_gb(learning_rate=0.2), X, y, folds)
    assert len(iterations) == len(folds)
    assert all(0 < n < EARLY_STOPPING["max_iter"] for n in iterations)

