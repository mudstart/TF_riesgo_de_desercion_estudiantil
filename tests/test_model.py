import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.config import (
    BASELINE,
    FEATURES_BINARIAS,
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    FEATURES_REDUNDANTES,
    GRADIENT_BOOSTING,
    METRICS_DIR,
    MIN_F1_MACRO,
    RANDOM_FOREST,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from src.data import build_target, cv_folds, load_raw, load_splits
from src.evaluation import cross_validate_model, f1_macro, predict_at
from src.features import select_features
from src.models import build_pipeline
from src.models.train_baseline import build_baseline
from src.models.train_gb import build_gb
from src.models.train_rf import build_rf

N = 300

BUILDERS = {BASELINE: build_baseline, RANDOM_FOREST: build_rf, GRADIENT_BOOSTING: build_gb}

# Versiones pequeñas de los tres modelos para que las pruebas sean rápidas.
SMALL = {
    BASELINE: lambda: build_baseline(),
    RANDOM_FOREST: lambda: build_rf(n_estimators=20),
    GRADIENT_BOOSTING: lambda: build_gb(max_iter=20),
}

# Tolerancia bajo el piso de F1-macro de la decisión 008 antes de dar por
# degradado el modelo final.
MARGIN = 0.02  # ver docs/decisiones/009_pruebas_y_ci.md


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
    y = pd.Series((X["Age at enrollment"] + rng.normal(0, 10, size=N) > 100).astype(int))
    return X.iloc[:200], y.iloc[:200], X.iloc[200:]


# Pruebas de los modelos (#16, pruebas 5, 6 y 8 de 8).


@pytest.mark.parametrize("name", list(SMALL))
def test_reproducibilidad(data, name):
    """Prueba 5: entrenar dos veces con la misma semilla da el mismo modelo.

    Protege la reproducibilidad del proyecto (RANDOM_STATE en config.py):
    cada constructor real (#9, #11, #12) fija su semilla, así que dos
    entrenamientos independientes con los mismos datos producen
    probabilidades idénticas, no solo parecidas.
    """
    X_train, y_train, X_new = data
    first = SMALL[name]().fit(X_train, y_train).predict_proba(X_new)
    second = SMALL[name]().fit(X_train, y_train).predict_proba(X_new)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("name", list(SMALL))
def test_salida_valida(data, name):
    """Prueba 6: las salidas del modelo son probabilidades y clases válidas.

    Protege a quien usa las predicciones (el umbral de la decisión 008 y la
    lista de riesgo): hay una probabilidad por clase, todas en [0, 1] y que
    suman 1 por fila; las clases son 0 (no deserta) y 1 (Dropout), y tanto
    predict como el umbral ajustado solo devuelven esas clases.
    """
    X_train, y_train, X_new = data
    pipe = SMALL[name]().fit(X_train, y_train)

    proba = pipe.predict_proba(X_new)
    assert proba.shape == (len(X_new), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.sum(axis=1) == pytest.approx(np.ones(len(X_new)))

    assert list(pipe.classes_) == [0, 1]
    assert set(pipe.predict(X_new)) <= {0, 1}
    assert set(predict_at(proba[:, 1], 0.38)) <= {0, 1}


def test_desempeno_minimo():
    """Prueba 8: el modelo final supera al DummyClassifier y no se degrada.

    Prueba de regresión sobre los datos reales. Reconstruye el modelo
    elegido en #13 con los hiperparámetros y el umbral versionados en
    reports/metrics/ y calcula su F1-macro out-of-fold en los 5 folds de
    train. Exige que supere a los clasificadores triviales (clase mayoritaria
    y aleatorio estratificado) y que no caiga más de MARGIN bajo el piso de
    la decisión 008. No usa test: la decisión 008 solo permite evaluarlo una vez.
    """
    if not RAW_DATA_PATH.exists():
        pytest.skip("Falta data/raw/data.csv: uv run python -m src.data")
    raw, splits = load_raw(), load_splits()
    X, y = select_features(raw), build_target(raw)
    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    folds = cv_folds(splits)

    cv = json.loads((METRICS_DIR / "comparacion_cv.json").read_text(encoding="utf-8"))
    name = cv["seleccion"]["modelo"]
    report = json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    params = {f"modelo__{k}": v for k, v in report["hiperparametros"].items()}
    final = BUILDERS[name]().set_params(**params)

    _, proba = cross_validate_model(final, X_train, y_train, folds)
    f1_final = f1_macro(y_train, predict_at(proba, cv["umbral"][name]))

    for strategy in ("most_frequent", "stratified"):
        dummy = build_pipeline(DummyClassifier(strategy=strategy, random_state=RANDOM_STATE))
        _, proba_dummy = cross_validate_model(dummy, X_train, y_train, folds)
        assert f1_final > f1_macro(y_train, predict_at(proba_dummy, 0.5)), strategy

    assert f1_final >= MIN_F1_MACRO - MARGIN
