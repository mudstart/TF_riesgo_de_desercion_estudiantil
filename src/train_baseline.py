"""Baseline: regresión logística con class_weight="balanced" (#9).

Modelo de referencia simple e interpretable contra el que se comparan los
candidatos (#11, #12). Puede ganar la comparación (#13). Desde la raíz:

    uv run python -m src.train_baseline

Solo usa train y los 5 folds fijos (decisión 003). Pasos:

1. Piso absoluto: DummyClassifier(strategy="stratified") con CV.
2. Ajuste ligero de C con CV, mismo criterio que #13 (decisión 005).
   Del C elegido se guardan sus métricas de CV y los tiempos de
   entrenamiento e inferencia.
3. Pipeline final entrenado con todo train -> models/baseline.joblib.
4. Coeficientes con todo train y su estabilidad entre folds.

Salidas: reports/metrics/baseline.json y reports/metrics/baseline_coeficientes.csv.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import BASELINE, METRICS_DIR, RANDOM_STATE
from src.data import build_target, cv_folds, load_raw, load_splits
from src.evaluation import cross_validate_model, summarize_folds
from src.features import select_features
from src.models import build_pipeline, save_pipeline
from src.training import fit_full_train, model_report, save_metrics, tune

# De más regularizado (más simple) a menos: en caso de empate gana el primero.
C_GRID = [float(c) for c in np.logspace(-3, 2, 11)]

COEF_PATH = METRICS_DIR / "baseline_coeficientes.csv"
N_TOP = 10


def build_baseline(C: float = 1.0) -> Pipeline:
    """Pipeline sin ajustar del baseline."""
    return build_pipeline(LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def coefficients(pipeline: Pipeline) -> pd.Series:
    """Coeficientes de un pipeline ajustado, indexados por columna transformada.

    Positivo = aumenta el riesgo de Dropout. Las numéricas están estandarizadas
    (efecto de +1 desviación estándar). Las one-hot no tienen categoría de
    referencia: la regularización L2 reparte el efecto entre las categorías y
    el intercepto, así que su signo y su orden son interpretables, pero no la
    magnitud exacta de cada una. Con C pequeño, todos están encogidos hacia 0.
    """
    names = pipeline.named_steps["preprocesamiento"].get_feature_names_out()
    return pd.Series(pipeline.named_steps["modelo"].coef_[0], index=names)


def coefficient_table(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds) -> pd.DataFrame:
    """Coeficientes con todo train y su estabilidad al reentrenar en cada fold.

    signo_estable es la fracción de folds en que el coeficiente tiene el mismo
    signo que con todo train (una columna que no aparece en un fold cuenta
    como signo distinto).
    """
    full = coefficients(pipeline)
    per_fold = pd.DataFrame(
        {k: coefficients(clone(pipeline).fit(X.iloc[tr], y.iloc[tr])) for k, (tr, _) in enumerate(folds)}
    ).reindex(full.index)
    table = pd.DataFrame(
        {
            "coef": full,
            "odds_ratio": np.exp(full),
            "coef_folds_media": per_fold.mean(axis=1),
            "coef_folds_std": per_fold.std(axis=1, ddof=1),
            "signo_estable": (np.sign(per_fold).eq(np.sign(full), axis=0)).mean(axis=1),
        }
    )
    table.index.name = "variable"
    return table.sort_values("coef", ascending=False)


def top_coefficients(table: pd.DataFrame, n: int = N_TOP) -> dict:
    """Los n coeficientes más positivos y los n más negativos (del más fuerte al más débil)."""

    def records(df: pd.DataFrame) -> list[dict]:
        return df[["coef", "odds_ratio", "signo_estable"]].round(4).reset_index().to_dict(orient="records")

    return {
        "aumentan_riesgo": records(table.head(n)),
        "reducen_riesgo": records(table.tail(n).iloc[::-1]),
    }


def main() -> dict:
    df, splits = load_raw(), load_splits()
    X, y = select_features(df), build_target(df)
    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    folds = cv_folds(splits)

    print("1. Piso absoluto: DummyClassifier(strategy='stratified')")
    dummy = build_pipeline(DummyClassifier(strategy="stratified", random_state=RANDOM_STATE))
    dummy_folds, _ = cross_validate_model(dummy, X_train, y_train, folds)
    dummy_summary = summarize_folds({"dummy": dummy_folds}).loc["dummy"]
    print(f"  F1-macro {dummy_summary['f1_macro_media']:.3f}, recall {dummy_summary['recall_riesgo_media']:.3f}")

    print("2. Ajuste de C (mayor recall con F1-macro >= piso, a su umbral)")
    estimator = build_baseline().named_steps["modelo"]
    best_C, grid_table, ev = tune(estimator, "C", C_GRID, X_train, y_train, folds)
    print(f"  C elegido: {best_C:g}")

    print("3. Pipeline final con todo train")
    pipeline, fit_time = fit_full_train(build_baseline(best_C), X_train, y_train)
    model_path = save_pipeline(pipeline, BASELINE)
    print(f"  Guardado en {model_path}")

    print("4. Coeficientes y estabilidad entre folds")
    coef = coefficient_table(pipeline, X_train, y_train, folds)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    coef.to_csv(COEF_PATH, float_format="%.6f", encoding="utf-8")

    report = model_report(BASELINE, pipeline, ev, y_train)
    report["tiempos"]["entrenamiento_train_completo_s"] = fit_time
    report["ajuste"] = {
        "parametro": "C",
        "rejilla": C_GRID,
        "criterio": "mayor recall de riesgo con F1-macro >= piso, a su umbral (decisión 005)",
        "elegido": best_C,
        "resultados": grid_table.to_dict(orient="records"),
    }
    report["referencia_dummy"] = {"estrategia": "stratified", "cv_umbral_05": dummy_summary.to_dict()}
    report["coeficientes"] = {"archivo": COEF_PATH.name, "n_columnas": len(coef), **top_coefficients(coef)}
    path = save_metrics(report)

    m = report["cv_umbral_ajustado"]
    print(
        f"Baseline (C={best_C:g}, umbral {report['umbral']:.3f}): "
        f"F1-macro {m['f1_macro_media']:.3f} ± {m['f1_macro_std']:.3f}, "
        f"recall {m['recall_riesgo_media']:.3f} ± {m['recall_riesgo_std']:.3f}"
    )
    print(f"Métricas en {path}; coeficientes en {COEF_PATH}.")
    return report


if __name__ == "__main__":
    main()
