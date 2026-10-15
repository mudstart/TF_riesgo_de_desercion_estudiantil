"""Comparación de baseline y candidatos, y evaluación final en test (#13).

Ver docs/decisiones/005_criterio_seleccion.md. Desde la raíz del proyecto:

    # 1. Comparación con validación cruzada (solo train). Se puede repetir.
    uv run python -m src.comparacion

    # 2. Evaluación del modelo elegido en test. UNA SOLA VEZ.
    uv run python -m src.comparacion --test

Los pipelines se leen de models/<nombre>.joblib, guardados por #9, #11 y #12
con save_pipeline. Con --provisional se usan modelos por defecto para los que
falten (solo para probar el código; sus resultados no valen para el informe).
"""

import argparse
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    BASELINE,
    CONFIDENCE,
    FIGURES_DIR,
    METRICS_DIR,
    MIN_F1_MACRO,
    MODELOS_COMPARADOS,
    N_BOOTSTRAP,
    RANDOM_STATE,
)
from src.data import build_target, cv_folds, load_raw, load_splits
from src.evaluation import (
    METRICAS,
    check_test_unused,
    choose_threshold,
    classification_metrics,
    cross_validate_model,
    cv_threshold_metrics,
    paired_bootstrap,
    plot_confusion_matrices,
    plot_pr_roc,
    predict_at,
    select_model,
    summarize_folds,
)
from src.features import select_features
from src.models import build_pipeline, load_pipeline

CV_PATH = METRICS_DIR / "comparacion_cv.json"
CV_TABLES_PATH = METRICS_DIR / "comparacion_cv.md"
TEST_PATH = METRICS_DIR / "test_final.json"

PROVISIONALES = {
    "baseline": LogisticRegression(class_weight="balanced", max_iter=2000),
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
}


def load_models(provisional: bool) -> tuple[dict, list[str]]:
    """Pipelines guardados por #9, #11 y #12, y la lista de los provisionales."""
    models, provisionales = {}, []
    for name in MODELOS_COMPARADOS:
        try:
            models[name] = load_pipeline(name)
        except FileNotFoundError:
            if not provisional:
                raise
            models[name] = build_pipeline(PROVISIONALES[name])
            provisionales.append(name)
    return models, provisionales


def green_ai(name: str) -> dict:
    """Energía y emisiones medidas por #15 en reports/metrics/<nombre>.json, si existen."""
    path = METRICS_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {k: data.get(k) for k in ("energia_kwh", "emisiones_gco2eq")}


def cv_tables(result: dict) -> str:
    """Tablas en Markdown para copiar en reports/comparacion_modelos.md."""

    def pm(row: dict, metric: str, digits: int = 3) -> str:
        return f"{row[metric + '_media']:.{digits}f} ± {row[metric + '_std']:.{digits}f}"

    def opt(value, fmt: str) -> str:
        return "pendiente (#15)" if value is None else format(value, fmt)

    lines = [
        "| Modelo | Umbral | F1-macro | Recall riesgo | Precisión riesgo | Balanced acc. "
        "| Entrenamiento (s) | Inferencia (ms/1000) | Energía (kWh) | Emisiones (gCO2eq) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in result["cv_umbral_ajustado"].items():
        t = result["cv_umbral_05"][name]
        g = result["green_ai"][name]
        lines.append(
            f"| {name} | {pm(row, 'umbral')} | {pm(row, 'f1_macro')} | {pm(row, 'recall_riesgo')} "
            f"| {pm(row, 'precision_riesgo')} | {pm(row, 'balanced_accuracy')} | {pm(t, 'fit_time_s', 2)} "
            f"| {pm(t, 'predict_time_ms_1000', 1)} | {opt(g['energia_kwh'], '.2e')} "
            f"| {opt(g['emisiones_gco2eq'], '.3f')} |"
        )

    lines += ["", "| Modelo | Umbral | F1-macro | Recall riesgo | Precisión riesgo | Balanced acc. |", "|---|---|---|---|---|---|"]
    for name, row in result["cv_umbral_05"].items():
        lines.append(
            f"| {name} | 0.5 | {pm(row, 'f1_macro')} | {pm(row, 'recall_riesgo')} "
            f"| {pm(row, 'precision_riesgo')} | {pm(row, 'balanced_accuracy')} |"
        )

    lines += ["", "| Comparación | Métrica | Diferencia | IC 95 % |", "|---|---|---|---|"]
    for pair, metrics in result["ic_vs_baseline"].items():
        for metric, ic in metrics.items():
            lines.append(f"| {pair} | {metric} | {ic['diferencia']:+.3f} | [{ic['ic_inf']:+.3f}, {ic['ic_sup']:+.3f}] |")
    for pair, ic in result["seleccion"]["ic_recall"].items():
        if pair.endswith(f" - {BASELINE}"):
            continue  # ya está en las comparaciones contra el baseline
        lines.append(f"| {pair} | recall_riesgo | {ic['diferencia']:+.3f} | [{ic['ic_inf']:+.3f}, {ic['ic_sup']:+.3f}] |")

    sel = result["seleccion"]
    lines += [
        "",
        f"- Elegibles (F1-macro ≥ {sel['piso_f1_macro']}): {', '.join(sel['elegibles'])}.",
        f"- Mayor recall de riesgo: {sel['mayor_recall']}.",
        f"- Empatados con él (IC de la diferencia contiene 0): {', '.join(sel['empatados'])}.",
        f"- **Modelo elegido: {sel['modelo']}**, umbral {result['umbral'][sel['modelo']]:.3f}.",
    ]
    if result["provisionales"]:
        lines.insert(0, f"> PROVISIONAL: modelos por defecto para {', '.join(result['provisionales'])}.\n")
    return "\n".join(lines) + "\n"


def load_data():
    df = load_raw()
    splits = load_splits()
    X, y = select_features(df), build_target(df)
    return X, y, splits


def run_cv(provisional: bool) -> dict:
    X, y, splits = load_data()
    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    folds = cv_folds(splits)
    models, provisionales = load_models(provisional)

    fold_05, fold_thr, probas, thresholds, preds = {}, {}, {}, {}, {}
    for name, pipeline in models.items():
        print(f"Validación cruzada: {name}...")
        fold_05[name], probas[name] = cross_validate_model(pipeline, X_train, y_train, folds)
        fold_thr[name] = cv_threshold_metrics(y_train, probas[name], folds)
        thresholds[name] = choose_threshold(y_train, probas[name])
        preds[name] = predict_at(probas[name], thresholds[name])

    seleccion = select_model(y_train, preds)

    # ¿Las diferencias contra el baseline son reales o ruido?
    ic_baseline = {
        f"{name} - {BASELINE}": {
            metric: paired_bootstrap(y_train, preds[name], preds[BASELINE], METRICAS[metric])
            for metric in ("f1_macro", "recall_riesgo")
        }
        for name in models
        if name != BASELINE
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(y_train, preds, FIGURES_DIR / "13_confusion_cv.png")
    plot_pr_roc(y_train, probas, thresholds, FIGURES_DIR / "13_pr_roc_cv.png")

    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "provisionales": provisionales,
        "parametros": {
            "min_f1_macro": MIN_F1_MACRO,
            "n_bootstrap": N_BOOTSTRAP,
            "confidence": CONFIDENCE,
        },
        "cv_umbral_05": summarize_folds(fold_05).to_dict(orient="index"),
        "cv_umbral_ajustado": summarize_folds(fold_thr).to_dict(orient="index"),
        "umbral": thresholds,
        "oof_umbral_ajustado": {name: classification_metrics(y_train, p) for name, p in preds.items()},
        "green_ai": {name: green_ai(name) for name in models},
        "ic_vs_baseline": ic_baseline,
        "seleccion": seleccion,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CV_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    CV_TABLES_PATH.write_text(cv_tables(result), encoding="utf-8")
    print(f"Modelo elegido: {seleccion['modelo']} (umbral {thresholds[seleccion['modelo']]:.3f}).")
    print(f"Resultados en {CV_PATH}.")
    return result


def run_test() -> dict:
    """Entrena el modelo elegido con todo train y lo evalúa una vez en test."""
    check_test_unused(TEST_PATH)
    if not CV_PATH.exists():
        raise FileNotFoundError(f"Falta {CV_PATH}: ejecute antes la comparación con CV.")
    cv = json.loads(CV_PATH.read_text(encoding="utf-8"))
    if cv["provisionales"]:
        raise RuntimeError(
            f"La selección usó modelos provisionales {cv['provisionales']}: "
            "no se gasta el test con ellos."
        )

    name = cv["seleccion"]["modelo"]
    threshold = cv["umbral"][name]
    X, y, splits = load_data()
    pipeline = load_pipeline(name)
    pipeline.fit(X.iloc[splits["train"]], y.iloc[splits["train"]])

    y_test = y.iloc[splits["test"]].to_numpy()
    proba = pipeline.predict_proba(X.iloc[splits["test"]])[:, 1]
    pred = predict_at(proba, threshold)

    # Intervalo de confianza de cada métrica en test con bootstrap.
    rng = np.random.default_rng(RANDOM_STATE)
    alpha = (1 - CONFIDENCE) / 2
    boots = {m: [] for m in METRICAS}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y_test), len(y_test))
        for m, f in METRICAS.items():
            boots[m].append(f(y_test[idx], pred[idx]))

    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": name,
        "umbral": threshold,
        "n_test": int(len(y_test)),
        "metricas": classification_metrics(y_test, pred),
        "ic": {m: [float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha))] for m, v in boots.items()},
    }
    plot_confusion_matrices(y_test, {f"{name} (test)": pred}, FIGURES_DIR / "13_confusion_test.png")
    TEST_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluación final en test guardada en {TEST_PATH}.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="evaluación final en test (una sola vez)")
    parser.add_argument("--provisional", action="store_true", help="usar modelos por defecto si faltan")
    args = parser.parse_args()
    run_test() if args.test else run_cv(args.provisional)
