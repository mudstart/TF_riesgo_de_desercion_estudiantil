import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import FEATURES_BINARIAS, FEATURES_NOMINALES, FEATURES_NUMERICAS, FEATURES_REDUNDANTES
from src.train_baseline import C_GRID, build_baseline, coefficient_table, top_coefficients

N = 400


@pytest.fixture
def data():
    """Columnas reales; menor Admission grade -> más riesgo, el resto es ruido."""
    rng = np.random.default_rng(0)
    X = {c: rng.choice([1, 2, 3], size=N) for c in FEATURES_NOMINALES}
    X |= {c: rng.normal(100, 15, size=N) for c in FEATURES_NUMERICAS}
    X |= {c: rng.integers(0, 2, size=N) for c in FEATURES_BINARIAS + FEATURES_REDUNDANTES}
    X = pd.DataFrame(X)
    y = pd.Series((X["Admission grade"] + rng.normal(0, 8, size=N) < 95).astype(int))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    return X, y, folds


def test_baseline_is_balanced_logistic_regression():
    model = build_baseline(C=0.1).named_steps["modelo"]
    assert type(model).__name__ == "LogisticRegression"
    assert model.class_weight == "balanced"
    assert model.C == 0.1


def test_c_grid_goes_from_simplest_to_most_complex():
    assert C_GRID == sorted(C_GRID)
    assert C_GRID[0] == pytest.approx(1e-3) and C_GRID[-1] == pytest.approx(100)


def test_coefficients_are_named_and_have_the_expected_sign(data):
    X, y, folds = data
    pipeline = build_baseline(C=1.0).fit(X, y)
    table = coefficient_table(pipeline, X, y, folds)

    names = pipeline.named_steps["preprocesamiento"].get_feature_names_out()
    assert sorted(table.index) == sorted(names)
    assert table["coef"].is_monotonic_decreasing
    assert table["signo_estable"].between(0, 1).all()
    # Nota de admisión más baja = más riesgo: coeficiente negativo y estable.
    assert table.loc["Admission grade", "coef"] < 0
    assert table.loc["Admission grade", "signo_estable"] == 1
    assert table.loc["Admission grade", "odds_ratio"] == pytest.approx(np.exp(table.loc["Admission grade", "coef"]))


def test_top_coefficients_split_by_direction(data):
    X, y, folds = data
    table = coefficient_table(build_baseline().fit(X, y), X, y, folds)
    top = top_coefficients(table, n=3)
    assert all(r["coef"] > 0 for r in top["aumentan_riesgo"])
    assert all(r["coef"] < 0 for r in top["reducen_riesgo"])
    assert top["reducen_riesgo"][0]["variable"] == "Admission grade"  # el más negativo primero
