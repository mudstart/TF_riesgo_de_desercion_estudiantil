"""Ajuste ligero de hiperparámetros y reporte estándar de métricas.

Lo usan #9 (baseline), #11 y #12 para que los tres modelos se ajusten y se
reporten igual. El criterio para elegir un hiperparámetro es el mismo con el
que #13 compara los modelos (docs/decisiones/005_criterio_seleccion.md):
mayor recall de riesgo con F1-macro >= MIN_F1_MACRO, a su umbral ajustado.

Todo se calcula solo con train y los folds fijos de cv_folds(splits).
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from src.config import METRICS_DIR, MIN_F1_MACRO
from src.evaluation import (
    choose_threshold,
    classification_metrics,
    cross_validate_model,
    cv_threshold_metrics,
    predict_at,
    summarize_folds,
)
from src.models import build_pipeline

# Claves que escribe #15 en reports/metrics/<nombre>.json: se conservan al reescribirlo.
GREEN_AI_KEYS = ("energia_kwh", "emisiones_gco2eq")


def evaluate_pipeline(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]
) -> dict:
    """Validación cruzada completa de un pipeline sin ajustar.

    Devuelve las métricas por fold al umbral 0.5 (con tiempos) y al umbral
    ajustado sin mirar cada fold, el umbral elegido con todas las
    probabilidades out-of-fold, las métricas out-of-fold a ese umbral y las
    probabilidades out-of-fold.
    """
    fold_05, proba = cross_validate_model(pipeline, X, y, folds)
    threshold = choose_threshold(y, proba)
    return {
        "fold_05": fold_05,
        "fold_ajustado": cv_threshold_metrics(y, proba, folds),
        "umbral": threshold,
        "oof": classification_metrics(y, predict_at(proba, threshold)),
        "proba": proba,
    }


def tune(
    estimator: BaseEstimator,
    param: str,
    grid: list,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[object, pd.DataFrame, dict]:
    """Elige el valor de param (un hiperparámetro del estimador) en grid.

    Para cada valor, validación cruzada con los folds fijos. Gana el de mayor
    recall de riesgo out-of-fold entre los que alcanzan F1-macro >= MIN_F1_MACRO
    a su umbral; si empatan, el de mayor F1-macro y luego el primero de grid
    (ordenar grid del modelo más simple al más complejo).

    Devuelve el valor elegido, una fila por valor probado y la evaluación del
    elegido (salida de evaluate_pipeline).
    """
    rows, evaluations = [], []
    for value in grid:
        pipeline = build_pipeline(clone(estimator).set_params(**{param: value}))
        ev = evaluate_pipeline(pipeline, X, y, folds)
        evaluations.append(ev)
        rows.append({param: value, "umbral": ev["umbral"], **ev["oof"]})
        print(f"  {param}={value:g}: recall {ev['oof']['recall_riesgo']:.3f}, F1-macro {ev['oof']['f1_macro']:.3f}")

    table = pd.DataFrame(rows)
    elegibles = table[table["f1_macro"] >= MIN_F1_MACRO]
    if elegibles.empty:
        raise ValueError(f"Ningún valor de {param} alcanza F1-macro >= {MIN_F1_MACRO}.")
    # sort_values es estable: a igualdad, conserva el orden de grid.
    best = elegibles.sort_values(["recall_riesgo", "f1_macro"], ascending=False, kind="stable").index[0]
    return grid[best], table, evaluations[best]


def confusion(y_true, y_pred) -> dict[str, int]:
    """Matriz de confusión con nombres (1 = Dropout, clase de riesgo)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def model_report(name: str, pipeline: Pipeline, ev: dict, y: pd.Series) -> dict:
    """Reporte estándar de un modelo a partir de su evaluación con CV.

    Mismas claves para baseline, random_forest y gradient_boosting. Las
    métricas son media y desviación entre folds; "oof" y "matriz_confusion"
    usan las predicciones out-of-fold de todo train.
    """
    folds_05 = summarize_folds({name: ev["fold_05"]}).loc[name]
    folds_ajustado = summarize_folds({name: ev["fold_ajustado"]}).loc[name]
    modelo = pipeline.named_steps["modelo"]
    return {
        "modelo": name,
        "estimador": type(modelo).__name__,
        "hiperparametros": {k: v for k, v in modelo.get_params().items() if np.isscalar(v) or v is None},
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(y)),
        "piso_f1_macro": MIN_F1_MACRO,
        "umbral": ev["umbral"],
        "cv_umbral_05": folds_05.to_dict(),
        "cv_umbral_ajustado": folds_ajustado.to_dict(),
        "oof_umbral_05": classification_metrics(y, predict_at(ev["proba"], 0.5)),
        "oof_umbral_ajustado": ev["oof"],
        "matriz_confusion": {
            "umbral_05": confusion(y, predict_at(ev["proba"], 0.5)),
            "umbral_ajustado": confusion(y, predict_at(ev["proba"], ev["umbral"])),
        },
        "tiempos": {
            "entrenamiento_s": folds_05["fit_time_s_media"],
            "inferencia_ms_1000": folds_05["predict_time_ms_1000_media"],
        },
    }


def fit_full_train(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, float]:
    """Entrena con todo train; devuelve el pipeline ajustado y el tiempo (s)."""
    start = time.perf_counter()
    pipeline.fit(X, y)
    return pipeline, time.perf_counter() - start


def save_metrics(report: dict, metrics_dir: Path = METRICS_DIR) -> Path:
    """Escribe reports/metrics/<modelo>.json conservando las claves de #15."""
    path = Path(metrics_dir) / f"{report['modelo']}.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report = {**report, **{k: previous.get(k) for k in GREEN_AI_KEYS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
