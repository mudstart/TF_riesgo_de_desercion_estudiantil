"""Funciones para evaluación y comparación de modelos.

Todas las funciones trabajan con el target binario (1 = Dropout, clase de
riesgo) y con los folds fijos de cv_folds(splits).
"""

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


def f1_macro(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def recall_riesgo(y_true, y_pred) -> float:
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def precision_riesgo(y_true, y_pred) -> float:
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


METRICAS: dict[str, Callable] = {
    "f1_macro": f1_macro,
    "recall_riesgo": recall_riesgo,
    "precision_riesgo": precision_riesgo,
    "balanced_accuracy": balanced_accuracy_score,
}


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Métricas mínimas del proyecto para predicciones 0/1."""
    return {name: float(metric(y_true, y_pred)) for name, metric in METRICAS.items()}


def predict_at(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Clase 1 (riesgo) si la probabilidad alcanza el umbral."""
    return (np.asarray(proba) >= threshold).astype(int)


def cross_validate_model(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Validación cruzada con los folds fijos del proyecto.

    Por cada fold clona el pipeline (mismos hiperparámetros, sin ajustar), lo
    entrena con las filas de entrenamiento y predice las de validación. Devuelve:

    - una fila por fold con las métricas al umbral 0.5, el tiempo de
      entrenamiento (s) y el de inferencia (ms por 1000 registros);
    - las probabilidades out-of-fold de la clase de riesgo, una por fila de X.

    X e y deben ser solo train; folds viene de cv_folds(splits).
    """
    proba = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(folds):
        model = clone(pipeline)

        start = time.perf_counter()
        model.fit(X.iloc[tr], y.iloc[tr])
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        p = model.predict_proba(X.iloc[va])[:, 1]
        predict_time = time.perf_counter() - start

        proba[va] = p
        rows.append(
            {
                "fold": k,
                **classification_metrics(y.iloc[va], predict_at(p, 0.5)),
                "fit_time_s": fit_time,
                "predict_time_ms_1000": predict_time / len(va) * 1e6,
            }
        )

    if np.isnan(proba).any():
        raise ValueError("Los folds no cubren todas las filas de train.")
    return pd.DataFrame(rows), proba


def summarize_folds(fold_metrics: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabla modelo x métrica con media y desviación estándar entre folds."""
    rows = {}
    for name, df in fold_metrics.items():
        cols = df.drop(columns="fold")
        rows[name] = {
            **{f"{c}_media": cols[c].mean() for c in cols},
            **{f"{c}_std": cols[c].std(ddof=1) for c in cols},
        }
    return pd.DataFrame(rows).T
