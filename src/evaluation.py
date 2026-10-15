"""Funciones para evaluación y comparación de modelos.

Implementa el criterio de docs/decisiones/005_criterio_seleccion.md. Todas las
funciones trabajan con el target binario (1 = Dropout, clase de riesgo).
"""

import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import CONFIDENCE, MIN_F1_MACRO, N_BOOTSTRAP, RANDOM_STATE


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


def threshold_curve(y_true, proba) -> pd.DataFrame:
    """Recall, precisión y F1-macro de la clase de riesgo para cada umbral posible.

    Cada fila corresponde a un valor distinto de proba usado como umbral
    (riesgo si proba >= umbral), de mayor a menor.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    order = np.argsort(-proba, kind="stable")
    p_sorted, y_sorted = proba[order], y_true[order]
    last = np.r_[p_sorted[1:] != p_sorted[:-1], True]  # última fila de cada umbral

    tp = np.cumsum(y_sorted)[last]
    fp = np.cumsum(1 - y_sorted)[last]
    pos = y_true.sum()
    fn, tn = pos - tp, len(y_true) - pos - fp

    f1_pos = 2 * tp / (2 * tp + fp + fn)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros(len(tn)), where=(2 * tn + fn + fp) > 0)
    return pd.DataFrame(
        {
            "umbral": p_sorted[last],
            "recall_riesgo": tp / pos,
            "precision_riesgo": tp / (tp + fp),
            "f1_macro": (f1_pos + f1_neg) / 2,
        }
    )


def choose_threshold(y_true, proba, min_f1_macro: float = MIN_F1_MACRO) -> float:
    """Umbral de mayor recall de riesgo con F1-macro >= min_f1_macro.

    Si ningún umbral alcanza el piso, devuelve el de mayor F1-macro (el modelo
    no será elegible en select_model).
    """
    curve = threshold_curve(y_true, proba)
    feasible = curve[curve["f1_macro"] >= min_f1_macro]
    if feasible.empty:
        return float(curve.loc[curve["f1_macro"].idxmax(), "umbral"])
    # El umbral más alto entre los de recall máximo: igual recall, más precisión.
    best = feasible[feasible["recall_riesgo"] == feasible["recall_riesgo"].max()]
    return float(best["umbral"].max())


def cv_threshold_metrics(
    y: pd.Series,
    proba: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    min_f1_macro: float = MIN_F1_MACRO,
) -> pd.DataFrame:
    """Métricas por fold con umbral ajustado sin mirar ese fold.

    Para el fold k, el umbral se elige con las probabilidades out-of-fold de
    los demás folds y se aplica al fold k. Así la media ± desviación del umbral
    ajustado no se infla por elegir y evaluar el umbral con las mismas filas.
    """
    y = np.asarray(y)
    rows = []
    for k, (_, va) in enumerate(folds):
        others = np.setdiff1d(np.arange(len(y)), va)
        threshold = choose_threshold(y[others], proba[others], min_f1_macro)
        rows.append(
            {"fold": k, "umbral": threshold, **classification_metrics(y[va], predict_at(proba[va], threshold))}
        )
    return pd.DataFrame(rows)


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


def paired_bootstrap(
    y_true,
    pred_a,
    pred_b,
    metric: Callable,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """Intervalo de confianza de metric(a) - metric(b) con bootstrap pareado.

    Ambos modelos se evalúan sobre las mismas filas remuestreadas, de modo que
    el intervalo refleja solo la diferencia entre ellos. Si el intervalo
    contiene 0, la diferencia no se distingue del ruido.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    alpha = (1 - confidence) / 2
    return {
        "diferencia": float(metric(y_true, pred_a) - metric(y_true, pred_b)),
        "ic_inf": float(np.quantile(diffs, alpha)),
        "ic_sup": float(np.quantile(diffs, 1 - alpha)),
    }


def select_model(
    y_true,
    preds: dict[str, np.ndarray],
    min_f1_macro: float = MIN_F1_MACRO,
    **bootstrap_kwargs,
) -> dict:
    """Aplica el criterio de selección de la decisión 005.

    preds tiene las predicciones 0/1 (al umbral ajustado) de cada modelo,
    ordenadas del más simple al más complejo.

    1. Elegibles: F1-macro >= min_f1_macro.
    2. Entre los elegibles, el de mayor recall de riesgo.
    3. Empatan con él los elegibles cuyo intervalo de la diferencia de recall
       (mejor - otro) contiene 0. Entre los empatados gana el más simple.
    """
    metrics = {name: classification_metrics(y_true, p) for name, p in preds.items()}
    elegibles = [name for name in preds if metrics[name]["f1_macro"] >= min_f1_macro]
    if not elegibles:
        raise ValueError(
            f"Ningún modelo alcanza F1-macro >= {min_f1_macro}: "
            "el criterio debe revisarse con una nueva decisión antes de usar test."
        )
    mejor = max(elegibles, key=lambda name: metrics[name]["recall_riesgo"])

    comparaciones = {}
    empatados = []
    for name in elegibles:
        if name == mejor:
            empatados.append(name)
            continue
        ic = paired_bootstrap(y_true, preds[mejor], preds[name], recall_riesgo, **bootstrap_kwargs)
        comparaciones[f"{mejor} - {name}"] = ic
        if ic["ic_inf"] <= 0:
            empatados.append(name)

    return {
        "modelo": empatados[0],
        "mayor_recall": mejor,
        "piso_f1_macro": min_f1_macro,
        "elegibles": elegibles,
        "empatados": empatados,
        "metricas": metrics,
        "ic_recall": comparaciones,
    }


def check_test_unused(path: Path) -> None:
    """Impide evaluar en test más de una vez (decisión 003).

    La evaluación final guarda su resultado en path; si ya existe, el test ya
    se usó y no debe volver a usarse.
    """
    if Path(path).exists():
        raise RuntimeError(
            f"El conjunto de test ya se usó: existe {path}. "
            "La evaluación final se hace una sola vez (docs/decisiones/003)."
        )


def plot_confusion_matrices(y_true, preds: dict[str, np.ndarray], path: Path) -> Path:
    """Una matriz de confusión por modelo, lado a lado."""
    fig, axes = plt.subplots(1, len(preds), figsize=(4.5 * len(preds), 4), squeeze=False)
    for ax, (name, pred) in zip(axes[0], preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, pred, display_labels=["No Dropout", "Dropout"], cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def plot_pr_roc(y_true, probas: dict[str, np.ndarray], thresholds: dict[str, float], path: Path) -> Path:
    """Curvas precision-recall y ROC de todos los modelos, con el umbral elegido."""
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(11, 4.5))
    y_true = np.asarray(y_true)
    for name, proba in probas.items():
        PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=ax_pr)
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax_roc)
        pred = predict_at(proba, thresholds[name])
        color = ax_pr.get_lines()[-1].get_color()
        ax_pr.plot(recall_riesgo(y_true, pred), precision_riesgo(y_true, pred), "o", color=color)
    ax_pr.axhline(y_true.mean(), color="grey", linestyle="--", linewidth=1, label="Azar")
    ax_pr.set_title("Precision-recall (punto = umbral elegido)")
    ax_pr.legend(loc="upper right")
    ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax_roc.set_title("ROC")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)
