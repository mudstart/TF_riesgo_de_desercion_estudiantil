import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.evaluation import (
    bootstrap_ci,
    check_test_unused,
    choose_threshold,
    classification_metrics,
    cross_validate_model,
    cv_threshold_metrics,
    paired_bootstrap,
    predict_at,
    recall_riesgo,
    select_model,
    threshold_curve,
)

N = 400


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
    y = pd.Series((X["a"] + rng.normal(scale=1.0, size=N) > 0.5).astype(int))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    return X, y, folds


@pytest.fixture
def proba(data):
    X, y, folds = data
    return cross_validate_model(LogisticRegression(), X, y, folds)[1]


def test_threshold_curve_matches_sklearn(data, proba):
    _, y, _ = data
    curve = threshold_curve(y, proba)
    for _, row in curve.sample(20, random_state=0).iterrows():
        m = classification_metrics(y, predict_at(proba, row["umbral"]))
        assert row["f1_macro"] == pytest.approx(m["f1_macro"])
        assert row["recall_riesgo"] == pytest.approx(m["recall_riesgo"])


def test_threshold_respects_f1_floor_and_maximizes_recall(data, proba):
    _, y, _ = data
    curve = threshold_curve(y, proba)
    floor = 0.6
    t = choose_threshold(y, proba, min_f1_macro=floor)
    m = classification_metrics(y, predict_at(proba, t))
    assert m["f1_macro"] >= floor
    assert m["recall_riesgo"] == pytest.approx(curve.loc[curve["f1_macro"] >= floor, "recall_riesgo"].max())
    # Un piso más bajo permite un umbral más bajo y más recall.
    assert choose_threshold(y, proba, min_f1_macro=0.4) <= t


def test_threshold_per_fold_is_chosen_without_that_fold(data, proba):
    _, y, folds = data
    proba_mod = proba.copy()
    va = folds[0][1]
    proba_mod[va] = 1 - proba_mod[va]  # cambiar solo el fold 0
    t = cv_threshold_metrics(y, proba, folds)["umbral"]
    t_mod = cv_threshold_metrics(y, proba_mod, folds)["umbral"]
    assert t[0] == t_mod[0]


def test_bootstrap_of_identical_models_is_zero():
    y = np.array([0, 1] * 50)
    pred = np.array([0, 1, 1, 0] * 25)
    ic = paired_bootstrap(y, pred, pred, recall_riesgo, n_boot=200)
    assert ic == {"diferencia": 0.0, "ic_inf": 0.0, "ic_sup": 0.0}


def test_bootstrap_ci_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    pred = np.where(rng.random(300) < 0.8, y, 1 - y)
    point = classification_metrics(y, pred)
    for metric, (low, high) in bootstrap_ci(y, pred, n_boot=300).items():
        assert low <= point[metric] <= high


def test_baseline_wins_tie_as_simplest():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    base = np.where(rng.random(500) < 0.8, y, 1 - y)
    otro = base.copy()
    otro[np.flatnonzero(y == 1)[:2]] = 1  # apenas más recall: ruido
    sel = select_model(y, {"baseline": base, "otro": otro}, min_f1_macro=0.5, n_boot=300)
    assert sel["modelo"] == "baseline"


def test_model_below_f1_floor_is_not_eligible():
    y = np.array([0] * 50 + [1] * 50)
    base = y.copy()
    base[:10] = 1  # F1-macro alto
    todo_riesgo = np.ones(100, dtype=int)  # recall 1 pero F1-macro bajo
    sel = select_model(y, {"baseline": base, "todo_riesgo": todo_riesgo}, min_f1_macro=0.6, n_boot=100)
    assert sel["elegibles"] == ["baseline"]
    assert sel["modelo"] == "baseline"


def test_clearly_better_candidate_wins():
    y = np.array([0] * 300 + [1] * 300)
    base = y.copy()
    base[300:450] = 0  # recall 0.5
    cand = y.copy()
    cand[300:310] = 0  # recall ~0.97
    sel = select_model(y, {"baseline": base, "cand": cand}, min_f1_macro=0.6, n_boot=300)
    assert sel["modelo"] == "cand"


def test_no_eligible_model_stops_selection():
    y = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match="Ningún modelo"):
        select_model(y, {"baseline": 1 - y}, min_f1_macro=0.6, n_boot=10)


def test_test_set_can_only_be_used_once(tmp_path):
    path = tmp_path / "test_final.json"
    check_test_unused(path)
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="ya se usó"):
        check_test_unused(path)
