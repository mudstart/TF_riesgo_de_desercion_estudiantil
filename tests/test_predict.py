import json

import numpy as np
import pandas as pd
import pytest

from src.config import DATA_SHA256, METRICS_DIR, RAW_DATA_PATH
from src.data import load_raw
from src.diccionario import ETIQUETAS, etiqueta
from src.errores import NO_CONCLUYENTE, RIESGO, SIN_RIESGO
from src.features import model_columns
from src.predict import CATEGORIAS, PERFILES, cargar_modelo, predecir, version_modelo

# Categoría esperada de cada perfil inventado (decisión 012).
ESPERADO = {
    "Recién egresada de secundaria, Enfermería": "Bajo",
    "Adulto que retoma estudios, Gestión nocturna": "Alto",
    "Ingreso a Ingeniería Informática": "Medio",
}


@pytest.fixture(scope="module")
def raw():
    if not RAW_DATA_PATH.exists():
        pytest.skip("Falta data/raw/data.csv: uv run python -m src.data")
    return load_raw()


@pytest.fixture(scope="module")
def modelo(raw):
    return cargar_modelo()


def test_etiqueta_returns_code_when_unknown():
    assert etiqueta("Course", 9500) == "Nursing"
    assert etiqueta("Gender", 0) == "Mujer"
    assert etiqueta("Course", 1) == "1"


def test_dictionary_covers_every_code_in_the_data(raw):
    for variable, labels in ETIQUETAS.items():
        missing = set(raw[variable].unique()) - set(labels)
        assert not missing, variable


def test_categories_cover_the_three_results():
    assert {nivel for nivel, _ in CATEGORIAS.values()} == {"Bajo", "Medio", "Alto"}
    assert CATEGORIAS[NO_CONCLUYENTE][1] == "No concluyente – requiere revisión humana"
    assert set(CATEGORIAS) == {SIN_RIESGO, NO_CONCLUYENTE, RIESGO}


def test_model_version_depends_on_parameters_threshold_and_data():
    base = version_modelo({"a": 1}, 0.38)
    assert base == version_modelo({"a": 1}, 0.38)
    assert len(base) == 12
    assert base != version_modelo({"a": 2}, 0.38)
    assert base != version_modelo({"a": 1}, 0.39)
    assert base != version_modelo({"a": 1}, 0.38, data_sha256="otro")


def test_traceability_matches_versioned_files(modelo):
    cv = json.loads((METRICS_DIR / "comparacion_cv.json").read_text(encoding="utf-8"))
    test = json.loads((METRICS_DIR / "test_final.json").read_text(encoding="utf-8"))
    t = modelo.trazabilidad
    assert t["modelo"] == cv["seleccion"]["modelo"]
    assert t["umbral"] == cv["umbral"][t["modelo"]]
    assert t["dataset_sha256"] == DATA_SHA256
    assert t["fecha_test"] == test["fecha"]
    assert modelo.desempeno["metricas_test"] == test["metricas"]


@pytest.mark.parametrize("nombre", list(PERFILES))
def test_example_profiles_are_invented_and_fall_in_their_category(raw, modelo, nombre):
    """Los perfiles de ejemplo no están en el dataset y caen en la categoría que dicen."""
    perfil = PERFILES[nombre]
    assert set(perfil) == set(model_columns())
    same = raw[model_columns()].eq(pd.Series(perfil)[model_columns()]).all(axis=1)
    assert not same.any()

    result = predecir(modelo, perfil)
    assert result["estado"] == "prediccion"
    assert result["categoria"] == ESPERADO[nombre]


def test_low_risk_in_alert_subgroup_is_flagged(modelo):
    """E3: un "bajo" en un subgrupo en alerta no debe usarse para excluir."""
    result = predecir(modelo, PERFILES["Recién egresada de secundaria, Enfermería"])
    assert result["resultado"] == SIN_RIESGO
    assert result["e3"]
    assert {a["subgrupo"] for a in result["alertas"]} >= {"Mujer", "Hasta 20", "Nursing"}


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"Admission grade": 250.0}, "fuera del rango"),
        ({"Age at enrollment": None}, "sin valor"),
        ({"Course": 1234}, "no visto"),
    ],
)
def test_invalid_input_abstains_with_reason(modelo, change, reason):
    """A1: fuera de rango, dato faltante o código nunca visto: no se predice."""
    perfil = dict(PERFILES["Ingreso a Ingeniería Informática"], **change)
    result = predecir(modelo, perfil)
    assert result["estado"] == "abstencion"
    assert any(reason in m for m in result["motivos"])
    assert "probabilidad" not in result


def test_local_factors_are_the_change_to_the_typical_value(modelo):
    perfil = PERFILES["Adulto que retoma estudios, Gestión nocturna"]
    factores = predecir(modelo, perfil)["factores"]
    assert len(factores) > 0
    assert factores["efecto"].abs().is_monotonic_decreasing

    top = factores.iloc[0]
    X = pd.DataFrame([perfil])[model_columns()]
    typical = X.assign(**{top["variable"]: modelo.tipicos[top["variable"]]})
    expected = modelo.pipeline.predict_proba(X)[0, 1] - modelo.pipeline.predict_proba(typical)[0, 1]
    assert top["efecto"] == pytest.approx(expected)


def test_typical_profile_has_no_local_factors(modelo):
    result = predecir(modelo, dict(modelo.tipicos))
    assert result["factores"].empty
    assert 0 <= result["probabilidad"] <= 1
    assert np.isfinite(result["probabilidad"])
