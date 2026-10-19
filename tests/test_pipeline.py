import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import src.features
from src.config import (
    FEATURES_BINARIAS,
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    FEATURES_PROHIBIDAS,
    FEATURES_REDUNDANTES,
    MIN_FREQUENCY,
    TARGET_COL,
)
from src.features import build_preprocessor, model_columns
from src.models import build_pipeline, load_pipeline, save_pipeline

N = 300


@pytest.fixture
def df():
    """Dataset sintético con las 36 columnas reales y el Target."""
    rng = np.random.default_rng(0)
    data = {}
    for c in FEATURES_NOMINALES:
        data[c] = rng.choice([1, 2, 3], size=N)
    for c in FEATURES_NUMERICAS:
        data[c] = rng.normal(100, 15, size=N)
    for c in FEATURES_BINARIAS + FEATURES_REDUNDANTES:
        data[c] = rng.integers(0, 2, size=N)
    for c in FEATURES_PROHIBIDAS:
        data[c] = rng.normal(5, 2, size=N)
    data[TARGET_COL] = rng.choice(["Graduate", "Dropout", "Enrolled"], size=N)
    return pd.DataFrame(data)


@pytest.fixture
def y(df):
    return (df[TARGET_COL] == "Dropout").astype(int)


def test_scaler_learns_only_from_train(df):
    df.loc[200:, "Admission grade"] += 1000  # filas que no son de train
    pre = build_preprocessor().fit(df.iloc[:200])
    scaler = pre.named_transformers_["numericas"][-1]
    i = list(pre.transformers_[1][2]).index("Admission grade")
    assert scaler.mean_[i] == pytest.approx(df["Admission grade"].iloc[:200].mean())


def test_output_excludes_prohibited_redundant_and_target(df):
    names = set(build_preprocessor().fit(df).get_feature_names_out())
    origin = {n for n in df.columns if any(name.startswith(n) for name in names)}
    assert not origin & set(FEATURES_PROHIBIDAS)
    assert not origin & set(FEATURES_REDUNDANTES)
    assert TARGET_COL not in origin


def test_prohibited_values_do_not_change_predictions(df, y):
    pipe = build_pipeline(LogisticRegression()).fit(df, y)
    altered = df.copy()
    altered[list(FEATURES_PROHIBIDAS)] = 999
    assert np.array_equal(pipe.predict_proba(df), pipe.predict_proba(altered))


def test_rare_categories_are_grouped(df):
    df.loc[: MIN_FREQUENCY - 2, "Course"] = 9999  # categoría con menos de MIN_FREQUENCY casos
    pre = build_preprocessor().fit(df)
    names = list(pre.get_feature_names_out())
    assert "Course_9999" not in names
    assert "Course_infrequent_sklearn" in names


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(class_weight="balanced"),
        RandomForestClassifier(n_estimators=10, class_weight="balanced", random_state=0),
        HistGradientBoostingClassifier(max_iter=10, class_weight="balanced", random_state=0),
    ],
)
def test_same_pipeline_works_for_the_three_models(df, y, model):
    pipe = build_pipeline(model).fit(df.iloc[:200], y.iloc[:200])
    assert pipe.predict(df.iloc[200:]).shape == (100,)


def test_sensitivity_adds_six_columns():
    assert len(model_columns(sensibilidad=True)) == len(model_columns()) + 6


def test_preprocessor_rejects_prohibited_in_config(monkeypatch):
    monkeypatch.setattr(src.features, "FEATURES_INGRESO", ("Curricular units 1st sem (grade)",))
    with pytest.raises(ValueError, match="prohibidas"):
        build_preprocessor()


def test_save_and_load_keep_predictions(tmp_path, df, y):
    pipe = build_pipeline(LogisticRegression()).fit(df, y)
    save_pipeline(pipe, "prueba", models_dir=tmp_path)
    loaded = load_pipeline("prueba", models_dir=tmp_path)
    assert np.array_equal(pipe.predict_proba(df), loaded.predict_proba(df))


# Pruebas del pipeline (#16, pruebas 4 y 7 de 8).


def test_pipeline_sin_fuga(df):
    """Prueba 4: transformar test no cambia lo aprendido con train.

    Protege la decisión 004: el preprocesador se ajusta con train y, al
    transformar test, no actualiza ninguna estadística (modas, medianas,
    media y escala, categorías del one-hot). Test lleva valores extremos,
    nulos y una categoría nueva para que cualquier recálculo se note.
    Además, cada fila de test se transforma igual sola que en lote: el
    resultado no depende de las demás filas de test.
    """
    train, test = df.iloc[:200], df.iloc[200:220].copy()
    test["Admission grade"] += 1000
    test.loc[test.index[:5], "Age at enrollment"] = np.nan
    test["Course"] = 123456

    pre = build_preprocessor().fit(train)
    fitted = joblib.hash(pre)
    batch = pre.transform(test)
    assert joblib.hash(pre) == fitted

    scaler = pre.named_transformers_["numericas"][-1]
    i = list(pre.transformers_[1][2]).index("Admission grade")
    assert scaler.mean_[i] == pytest.approx(train["Admission grade"].mean())

    one_by_one = np.vstack([pre.transform(test.iloc[[k]]) for k in range(len(test))])
    assert np.array_equal(batch, one_by_one)


def test_robustez_de_entrada(df, y):
    """Prueba 7: el pipeline acepta categorías nuevas y rechaza registros incompletos.

    Protege el uso con cohortes futuras: un código de carrera que no estaba
    en train no rompe la predicción (va a "infrecuentes", decisión 004), y
    un registro al que le falta una columna del modelo se rechaza con un
    ValueError propio (ValidarColumnas) que nombra la columna, en lugar de
    predecir con datos incompletos.
    """
    pipe = build_pipeline(LogisticRegression()).fit(df, y)

    new = df.head(1).copy()
    new["Course"] = 123456
    proba = pipe.predict_proba(new)
    assert proba.shape == (1, 2)
    assert np.isfinite(proba).all()

    with pytest.raises(ValueError, match="Registro incompleto.*Admission grade"):
        pipe.predict(df.head(1).drop(columns=["Admission grade"]))
