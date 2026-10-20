import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES_BINARIAS, FEATURES_NOMINALES, FEATURES_NUMERICAS
from src.errores import (
    NO_CONCLUYENTE,
    RIESGO,
    SIN_RIESGO,
    abstention_table,
    age_band,
    clasificar,
    dominio_entrenamiento,
    error_type,
    neutralize,
    revisar_entrada,
    subgroup_metrics,
)
from src.evaluation import classification_metrics, predict_at
from src.features import model_columns

N = 200


@pytest.fixture
def X():
    """Train sintético con las columnas del modelo; el código 9 de Course es infrecuente."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            **{c: rng.choice([1, 2, 3], size=N) for c in FEATURES_NOMINALES},
            **{c: rng.uniform(0, 100, size=N) for c in FEATURES_NUMERICAS},
            **{c: rng.integers(0, 2, size=N) for c in FEATURES_BINARIAS},
        }
    )
    X.loc[:2, "Course"] = 9
    return X[model_columns()]


def test_subgroup_metrics_recall_fpr_and_alert():
    # Grupo A: 40 desertores, detecta 36 (0.9); grupo B: 40 desertores, detecta 20 (0.5).
    y = np.r_[np.ones(40), np.zeros(60), np.ones(40), np.zeros(60)].astype(int)
    pred = np.r_[np.ones(36), np.zeros(4), np.ones(30), np.zeros(30), np.ones(20), np.zeros(20), np.ones(6), np.zeros(54)]
    groups = np.r_[["A"] * 100, ["B"] * 100]
    t = subgroup_metrics(y, pred.astype(int), groups, n_boot=500).set_index("subgrupo")

    assert t.loc["A", "recall_riesgo"] == pytest.approx(0.9)
    assert t.loc["B", "recall_riesgo"] == pytest.approx(0.5)
    assert t.loc["A", "fpr"] == pytest.approx(0.5)
    assert t.loc["B", "fpr"] == pytest.approx(0.1)
    assert t.loc["B", "brecha_recall"] == pytest.approx(-0.4)
    assert t.loc["B", "alerta"] and not t.loc["A", "alerta"]
    lo, hi = t.loc["B", "recall_ic"]
    assert lo <= 0.5 <= hi


def test_subgroup_without_enough_dropouts_is_not_evaluable():
    y = np.r_[np.ones(5), np.zeros(45), np.ones(40), np.zeros(10)].astype(int)
    pred = np.r_[np.zeros(50), np.ones(50)].astype(int)
    t = subgroup_metrics(y, pred, np.r_[["A"] * 50, ["B"] * 50], n_boot=200).set_index("subgrupo")
    assert not t.loc["A", "evaluable"]
    assert not t.loc["A", "alerta"]  # recall 0, pero sin evidencia suficiente


def test_age_band_limits():
    bands = age_band(pd.Series([17, 20, 21, 25, 26, 70]), (20, 25)).astype(str).tolist()
    assert bands == ["Hasta 20", "Hasta 20", "21 a 25", "21 a 25", "Más de 25", "Más de 25"]


def test_error_type():
    assert error_type([1, 1, 0, 0], [1, 0, 1, 0]).tolist() == ["TP", "FN", "FP", "TN"]


def test_clasificar_abstention_zone():
    out = clasificar([0.10, 0.34, 0.38, 0.42, 0.90], threshold=0.38, ancho=0.05)
    assert out.tolist() == [SIN_RIESGO, NO_CONCLUYENTE, NO_CONCLUYENTE, NO_CONCLUYENTE, RIESGO]
    # Sin zona, coincide con predict_at.
    plain = clasificar([0.10, 0.38, 0.90], threshold=0.38, ancho=0.0)
    assert plain.tolist() == [SIN_RIESGO, RIESGO, RIESGO]


def test_abstention_table_without_zone_matches_metrics():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.3 + rng.uniform(0, 0.7, 300), 0, 1)
    table = abstention_table(y, proba, 0.5, anchos=(0.0, 0.1)).set_index("ancho")
    expected = classification_metrics(y, predict_at(proba, 0.5))
    assert table.loc[0.0, "n_zona"] == 0
    assert table.loc[0.0, "recall_riesgo_decididos"] == pytest.approx(expected["recall_riesgo"])
    assert table.loc[0.1, "n_zona"] == int((np.abs(proba - 0.5) < 0.1).sum())


def test_revisar_entrada_accepts_train_rows(X):
    review = revisar_entrada(X.iloc[10:], dominio_entrenamiento(X))
    assert not review["abstenerse"].any()
    assert not review["escalar"].any()


def test_revisar_entrada_abstains_and_escalates(X):
    dominio = dominio_entrenamiento(X)
    new = X.iloc[[10, 11, 12, 13, 0]].copy().reset_index(drop=True)
    new.loc[0, "Age at enrollment"] = 1000  # fuera de rango
    new.loc[1, "Mother's occupation"] = 77  # código no visto
    new.loc[2, "Gender"] = np.nan  # falta un valor
    review = revisar_entrada(new, dominio)

    assert review["abstenerse"].tolist() == [True, True, True, False, False]
    assert "Age at enrollment fuera del rango" in review.loc[0, "motivos"]
    # Fila 4: Course = 9, visto solo 3 veces en train -> se da resultado con revisión humana.
    assert review["escalar"].tolist() == [False, False, False, False, True]
    assert "Course" in review.loc[4, "motivos"]


def test_revisar_entrada_handles_empty_and_non_numeric_values(X):
    """Un campo vacío (None) o un texto no rompen la revisión: se abstiene con el motivo."""
    new = X.iloc[[10, 11, 12]].copy().reset_index(drop=True).astype(object)
    new.loc[0, "Age at enrollment"] = None
    new.loc[1, "Course"] = "abc"
    review = revisar_entrada(new, dominio_entrenamiento(X))

    assert review["abstenerse"].tolist() == [True, True, False]
    assert "Age at enrollment sin valor" in review.loc[0, "motivos"]
    assert "Course: valor no numérico" in review.loc[1, "motivos"]


def test_revisar_entrada_missing_column_abstains(X):
    review = revisar_entrada(X.drop(columns="Admission grade"), dominio_entrenamiento(X))
    assert review["abstenerse"].all()


def test_neutralize_fixes_columns_without_modifying_input(X):
    before = X.copy()
    out = neutralize(X, ["Gender", "Course"])
    assert out["Gender"].nunique() == 1 and out["Course"].nunique() == 1
    pd.testing.assert_frame_equal(X, before)
    pd.testing.assert_series_equal(out["Admission grade"], X["Admission grade"])
