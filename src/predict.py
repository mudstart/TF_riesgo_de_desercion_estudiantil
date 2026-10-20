"""Lógica de la demo de predicción (#17), sin interfaz.

Criterio en docs/decisiones/012_demo_prediccion.md. app/app.py solo muestra
lo que devuelven estas funciones:

- cargar_modelo: reconstruye el modelo elegido en #13 con sus hiperparámetros
  versionados y prepara el dominio de train, los valores típicos y la
  trazabilidad.
- predecir: valida un perfil (reglas A1 y E2 de la decisión 009), y si no hay
  que abstenerse, da la probabilidad, la categoría (bajo / no concluyente /
  alto, regla E1), los factores del caso y las alertas de subgrupo (E3).

Solo usa train para ajustar; test solo se lee de test_final.json.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    ANCHO_ABSTENCION,
    BASELINE,
    DATA_SHA256,
    FEATURES_NUMERICAS,
    GRADIENT_BOOSTING,
    METRICS_DIR,
    RANDOM_FOREST,
)
from src.data import build_target, load_raw, load_splits
from src.errores import (
    ETIQUETAS,
    NO_CONCLUYENTE,
    RESULT_PATH,
    RIESGO,
    SIN_RIESGO,
    age_band,
    clasificar,
    dominio_entrenamiento,
    revisar_entrada,
)
from src.features import model_columns, select_features
from src.models.train_baseline import build_baseline
from src.models.train_gb import build_gb
from src.models.train_rf import build_rf

CV_PATH = METRICS_DIR / "comparacion_cv.json"
TEST_PATH = METRICS_DIR / "test_final.json"

BUILDERS = {BASELINE: build_baseline, RANDOM_FOREST: build_rf, GRADIENT_BOOSTING: build_gb}

# Categorías de la decisión 012: la zona de abstención ocupa el lugar de "medio".
CATEGORIAS = {
    SIN_RIESGO: ("Bajo", "Sin riesgo detectado"),
    NO_CONCLUYENTE: ("Medio", "No concluyente – requiere revisión humana"),
    RIESGO: ("Alto", "Riesgo de deserción: priorizar acompañamiento"),
}

AVISO = (
    "Esta predicción es apoyo para priorizar acompañamiento académico. NO debe usarse "
    "para sancionar, excluir ni tomar decisiones automáticas sobre el estudiante."
)

# Qué NO permite concluir el resultado.
NO_PERMITE_CONCLUIR = (
    "Que el estudiante vaya a desertar o no: es una probabilidad estimada con datos del ingreso, "
    "y más de la mitad de los estudiantes marcados como riesgo no desertan.",
    "Por qué desertaría: los factores muestran a qué es sensible el modelo, no causas.",
    "Nada sobre su rendimiento posterior: el modelo solo usa datos disponibles al ingreso (decisión 001).",
)

# Condiciones de detención de la decisión 009 (D1–D3).
ADVERTENCIAS_USO = (
    "D1. Solo vale para la institución, el país y los planes de estudio del dataset; en otro contexto o en "
    "cohortes sin revalidar, no debe usarse.",
    "D2. No debe usarse para sancionar, excluir, admitir, retirar becas ni decidir sin revisión humana.",
    "D3. Si en una cohorte nueva con desenlace conocido el recall cae por debajo de 0.75, hay que detener "
    "su uso y revisarlo.",
)


@dataclass
class Modelo:
    """Modelo elegido en #13, listo para predecir, con todo lo que muestra la demo."""

    nombre: str
    pipeline: Pipeline
    umbral: float
    dominio: dict
    tipicos: dict
    trazabilidad: dict
    desempeno: dict
    importancia: pd.DataFrame


def _read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def valores_tipicos(X: pd.DataFrame) -> dict:
    """Mediana de las numéricas y moda del resto, columna a columna, en train."""
    return {
        c: (float(X[c].median()) if c in FEATURES_NUMERICAS else X[c].mode().iloc[0].item())
        for c in model_columns()
    }


def version_modelo(hiperparametros: dict, umbral: float, data_sha256: str = DATA_SHA256) -> str:
    """12 primeros caracteres del SHA-256 de hiperparámetros, umbral y dataset."""
    payload = json.dumps(
        {"hiperparametros": hiperparametros, "umbral": umbral, "dataset": data_sha256}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def desempeno_conocido(test: dict, errores: dict) -> dict:
    """Métricas de test (#13) y subgrupos donde el modelo falla más (#14)."""
    recall = test["metricas"]["recall_riesgo"]
    peores = sorted(errores["alertas"], key=lambda a: a["recall_riesgo"])
    return {
        "metricas_test": test["metricas"],
        "ic_test": test["ic"],
        "n_test": test["n_test"],
        "detecta_de_cada_10": round(recall * 10),
        "falla_mas_en": [
            {"variable": a["variable"], "subgrupo": a["subgrupo"], "recall": a["recall_riesgo"]} for a in peores
        ],
    }


def cargar_modelo() -> Modelo:
    """Reconstruye el modelo elegido en #13 con todo train (decisión 012)."""
    cv = _read_json(CV_PATH)
    nombre = cv["seleccion"]["modelo"]
    umbral = cv["umbral"][nombre]
    report = _read_json(METRICS_DIR / f"{nombre}.json")
    test = _read_json(TEST_PATH)

    raw, splits = load_raw(), load_splits()
    X = select_features(raw).iloc[splits["train"]]
    y = build_target(raw).iloc[splits["train"]]
    params = {f"modelo__{k}": v for k, v in report["hiperparametros"].items()}
    pipeline = BUILDERS[nombre]().set_params(**params).fit(X, y)

    trazabilidad = {
        "modelo": nombre,
        "version": version_modelo(report["hiperparametros"], umbral),
        "fecha_entrenamiento": report["fecha"],
        "dataset_sha256": DATA_SHA256,
        "n_train": int(len(y)),
        "umbral": umbral,
        "zona_no_concluyente": [umbral - ANCHO_ABSTENCION, umbral + ANCHO_ABSTENCION],
        "fecha_test": test["fecha"],
    }
    importancia = pd.read_csv(METRICS_DIR / f"{nombre}_importancia.csv")
    return Modelo(
        nombre=nombre,
        pipeline=pipeline,
        umbral=umbral,
        dominio=dominio_entrenamiento(X),
        tipicos=valores_tipicos(X),
        trazabilidad=trazabilidad,
        desempeno=desempeno_conocido(test, _read_json(RESULT_PATH)),
        importancia=importancia,
    )


def factores_locales(modelo: Modelo, X: pd.DataFrame) -> pd.DataFrame:
    """Cuánto cambia la probabilidad si cada variable toma su valor típico de train.

    efecto > 0: el valor del perfil sube la probabilidad respecto al típico.
    Es la sensibilidad del modelo para este perfil, no una causa. Se omiten
    las variables cuyo valor ya es el típico.
    """
    columns = [c for c in model_columns() if X[c].iloc[0] != modelo.tipicos[c]]
    if not columns:
        return pd.DataFrame(columns=["variable", "valor", "tipico", "efecto"])
    variants = pd.concat([X] * len(columns), ignore_index=True)
    for i, c in enumerate(columns):
        variants.loc[i, c] = modelo.tipicos[c]
    p = modelo.pipeline.predict_proba(X)[0, 1]
    p_tipico = modelo.pipeline.predict_proba(variants)[:, 1]
    table = pd.DataFrame(
        {
            "variable": columns,
            "valor": [X[c].iloc[0] for c in columns],
            "tipico": [modelo.tipicos[c] for c in columns],
            "efecto": p - p_tipico,
        }
    )
    return table.reindex(table["efecto"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def subgrupos_del_perfil(perfil: dict) -> dict[str, str]:
    """Subgrupo del perfil en cada variable del #14 que se puede deducir del formulario.

    International se deduce de la nacionalidad (1 = portuguesa). Scholarship
    holder no entra al modelo ni al formulario, así que no se evalúa.
    """
    return {
        "Gender": ETIQUETAS["Gender"].get(perfil["Gender"]),
        "Rango de edad": str(age_band([perfil["Age at enrollment"]]).iloc[0]),
        "Estado civil": "Soltero/a" if perfil["Marital Status"] == 1 else "Otro estado civil",
        "Displaced": ETIQUETAS["Displaced"].get(perfil["Displaced"]),
        "International": ETIQUETAS["International"].get(int(perfil["Nacionality"] != 1)),
        "Educational special needs": ETIQUETAS["Educational special needs"].get(perfil["Educational special needs"]),
        "Course": ETIQUETAS["Course"].get(perfil["Course"]),
    }


def alertas_del_perfil(perfil: dict, errores: dict | None = None) -> list[dict]:
    """Subgrupos en alerta del #14 a los que pertenece el perfil."""
    errores = errores if errores is not None else _read_json(RESULT_PATH)
    grupos = subgrupos_del_perfil(perfil)
    return [a for a in errores["alertas"] if grupos.get(a["variable"]) == a["subgrupo"]]


def predecir(modelo: Modelo, perfil: dict) -> dict:
    """Resultado de la demo para un perfil (diccionario con las columnas del modelo).

    estado = "abstencion": no se predice (A1) y motivos explica por qué.
    estado = "prediccion": probabilidad, resultado de clasificar (E1),
    categoría, aviso de revisión si hay un código infrecuente (E2), factores
    del caso y alertas de subgrupo; e3 indica que un "sin riesgo" no debe
    usarse para excluir al estudiante del acompañamiento.
    """
    X = pd.DataFrame([perfil])
    revision = revisar_entrada(X, modelo.dominio).iloc[0]
    if revision["abstenerse"]:
        return {"estado": "abstencion", "motivos": revision["motivos"].split("; ")}

    X = X[model_columns()]
    proba = float(modelo.pipeline.predict_proba(X)[0, 1])
    resultado = clasificar(np.array([proba]), modelo.umbral)[0]
    nivel, texto = CATEGORIAS[resultado]
    alertas = alertas_del_perfil(perfil)
    return {
        "estado": "prediccion",
        "probabilidad": proba,
        "resultado": resultado,
        "categoria": nivel,
        "texto": texto,
        "escalar": bool(revision["escalar"]),
        "motivos": [m for m in revision["motivos"].split("; ") if m],
        "factores": factores_locales(modelo, X),
        "alertas": alertas,
        "e3": resultado == SIN_RIESGO and bool(alertas),
    }


# Perfiles inventados (no tomados del dataset). tests/test_predict.py comprueba
# que ninguno coincide con una fila de data.csv y que cada uno cae en su categoría.
PERFILES: dict[str, dict] = {
    # Bajo, pero en subgrupos en alerta (mujer, hasta 20 años, Nursing): muestra E3.
    "Recién egresada de secundaria, Enfermería": {
        "Marital Status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Course": 9500,
        "Previous qualification": 1,
        "Previous qualification (grade)": 135.4,
        "Nacionality": 1,
        "Mother's qualification": 19,
        "Father's qualification": 37,
        "Mother's occupation": 5,
        "Father's occupation": 7,
        "Admission grade": 131.7,
        "Displaced": 1,
        "Educational special needs": 0,
        "Gender": 0,
        "Age at enrollment": 18,
    },
    # Alto.
    "Adulto que retoma estudios, Gestión nocturna": {
        "Marital Status": 2,
        "Application mode": 39,
        "Application order": 2,
        "Course": 9991,
        "Previous qualification": 1,
        "Previous qualification (grade)": 121.3,
        "Nacionality": 1,
        "Mother's qualification": 37,
        "Father's qualification": 37,
        "Mother's occupation": 9,
        "Father's occupation": 7,
        "Admission grade": 118.4,
        "Displaced": 0,
        "Educational special needs": 0,
        "Gender": 1,
        "Age at enrollment": 34,
    },
    # En la zona de abstención: no concluyente.
    "Ingreso a Ingeniería Informática": {
        "Marital Status": 1,
        "Application mode": 1,
        "Application order": 1,
        "Course": 9119,
        "Previous qualification": 1,
        "Previous qualification (grade)": 141.6,
        "Nacionality": 1,
        "Mother's qualification": 1,
        "Father's qualification": 19,
        "Mother's occupation": 4,
        "Father's occupation": 9,
        "Admission grade": 146.2,
        "Displaced": 1,
        "Educational special needs": 0,
        "Gender": 1,
        "Age at enrollment": 22,
    },
}
