"""Ajuste ligero de hiperparámetros y reporte estándar de métricas.

Lo usan #9 (baseline), #11 y #12 para que los tres modelos se ajusten y se
reporten igual. El baseline ajusta un hiperparámetro con tune (decisión 005);
los candidatos usan random_search (decisión 006). En ambos casos se elige por
F1-macro medio entre folds al umbral 0.5; el umbral de decisión de cada modelo
se fija después, en la comparación (#13), con el mismo criterio para todos.

Todo se calcula solo con train y los folds fijos de cv_folds(splits).
"""

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import METRICS_DIR, RANDOM_STATE
from src.evaluation import classification_metrics, cross_validate_model, predict_at, summarize_folds
from src.features import model_columns
from src.models import build_pipeline

# Umbral de las métricas de este reporte; el ajustado lo fija #13.
THRESHOLD = 0.5

# Claves que escribe #15 en reports/metrics/<nombre>.json: se conservan al reescribirlo.
GREEN_AI_KEYS = ("energia_kwh", "emisiones_gco2eq")


def tune(
    estimator: BaseEstimator,
    param: str,
    grid: list,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[object, pd.DataFrame, dict]:
    """Elige el valor de param (un hiperparámetro del estimador) en grid.

    Para cada valor, validación cruzada con los folds fijos y F1-macro medio
    entre folds al umbral 0.5. Regla de una desviación estándar: gana el
    primero de grid cuyo F1-macro alcanza el mejor menos la desviación estándar
    del mejor entre folds (no el error estándar std/√k de la regla 1-SE; ver
    docs/decisiones/005_ajuste_y_metricas.md). Diferencias menores no se
    distinguen del ruido y se prefiere el modelo más simple, así que grid va
    del más simple al más complejo.

    Devuelve el valor elegido, una fila por valor probado (media y desviación
    de F1-macro, recall y precisión de riesgo) y la validación cruzada del
    elegido: {"folds": métricas por fold, "proba": probabilidades out-of-fold}.
    """
    rows, evaluations = [], []
    for value in grid:
        pipeline = build_pipeline(clone(estimator).set_params(**{param: value}))
        folds_df, proba = cross_validate_model(pipeline, X, y, folds)
        evaluations.append({"folds": folds_df, "proba": proba})
        row = {param: value}
        for m in ("f1_macro", "recall_riesgo", "precision_riesgo"):
            row[f"{m}_media"] = folds_df[m].mean()
            row[f"{m}_std"] = folds_df[m].std(ddof=1)
        rows.append(row)
        print(f"  {param}={value:g}: F1-macro {row['f1_macro_media']:.3f}, recall {row['recall_riesgo_media']:.3f}")

    table = pd.DataFrame(rows)
    top = table.loc[table["f1_macro_media"].idxmax()]
    table["dentro_1de"] = table["f1_macro_media"] >= top["f1_macro_media"] - top["f1_macro_std"]
    best = int(table["dentro_1de"].idxmax())  # primer True: el más simple
    return grid[best], table, evaluations[best]


def random_search(
    pipeline: Pipeline,
    space: dict,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_iter: int,
) -> tuple[dict, pd.DataFrame, float]:
    """Búsqueda aleatoria por F1-macro medio con los folds fijos (decisión 006).

    space usa nombres del pipeline ("modelo__<parámetro>"). Devuelve los
    hiperparámetros ganadores sin el prefijo "modelo__", una fila por
    combinación probada ordenada de mejor a peor y el tiempo total (s).
    """
    searcher = RandomizedSearchCV(
        pipeline,
        space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=folds,
        refit=False,
        random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    searcher.fit(X, y)
    elapsed = time.perf_counter() - start

    results = pd.DataFrame(searcher.cv_results_)
    table = pd.DataFrame(
        {
            **{p.removeprefix("modelo__"): results[f"param_{p}"] for p in space},
            "f1_macro_media": results["mean_test_score"],
            "f1_macro_std": results["std_test_score"],
            "fit_time_s_media": results["mean_fit_time"],
        }
    ).sort_values("f1_macro_media", ascending=False, kind="stable")
    best = {p.removeprefix("modelo__"): v for p, v in searcher.best_params_.items()}
    return best, table.reset_index(drop=True), elapsed


def permutation_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Permutation importance por columna del modelo (caída de F1-macro).

    En cada fold el pipeline se entrena con las filas de entrenamiento y se
    permuta cada columna en las de validación. Se reporta la media y la
    desviación entre folds, y en cuántos folds la importancia es positiva.
    """
    columns = model_columns()
    per_fold = {}
    for k, (tr, va) in enumerate(folds):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        result = permutation_importance(
            model,
            X.iloc[va][columns],
            y.iloc[va],
            scoring="f1_macro",
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
        )
        per_fold[k] = pd.Series(result.importances_mean, index=columns)
    per_fold = pd.DataFrame(per_fold)
    table = pd.DataFrame(
        {
            "importancia_media": per_fold.mean(axis=1),
            "importancia_std": per_fold.std(axis=1, ddof=1),
            "folds_positiva": (per_fold > 0).sum(axis=1),
        }
    )
    table.index.name = "variable"
    return table.sort_values("importancia_media", ascending=False)


def confusion(y_true, y_pred) -> dict[str, int]:
    """Matriz de confusión con nombres (1 = Dropout, clase de riesgo)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def model_report(name: str, pipeline: Pipeline, ev: dict, y: pd.Series) -> dict:
    """Reporte estándar de un modelo a partir de su validación cruzada.

    Mismas claves para baseline, random_forest y gradient_boosting. "cv" es
    media y desviación entre folds; "oof" y "matriz_confusion" usan las
    predicciones out-of-fold de todo train. Todo al umbral 0.5.
    """
    cv = summarize_folds({name: ev["folds"]}).loc[name]
    pred = predict_at(ev["proba"], THRESHOLD)
    modelo = pipeline.named_steps["modelo"]
    return {
        "modelo": name,
        "estimador": type(modelo).__name__,
        "hiperparametros": {
            k: v
            for k, v in modelo.get_params().items()
            if (np.isscalar(v) or v is None) and v != "deprecated"  # p. ej. penalty en sklearn >= 1.8
        },
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(y)),
        "umbral": THRESHOLD,
        "cv": {k: float(v) for k, v in cv.items()},
        "oof": classification_metrics(y, pred),
        "matriz_confusion": confusion(y, pred),
        "tiempos": {
            "entrenamiento_s": float(cv["fit_time_s_media"]),
            "inferencia_ms_1000": float(cv["predict_time_ms_1000_media"]),
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
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
