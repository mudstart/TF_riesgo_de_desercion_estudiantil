"""Candidato 1: Random Forest con class_weight="balanced" (#11).

Primer modelo no lineal, comparado con el baseline (#9) en #13. Desde la raíz:

    uv run python -m src.models.train_rf

Solo usa train y los 5 folds fijos (decisión 003). Pasos (decisión 006):

1. Búsqueda aleatoria de 25 combinaciones de n_estimators, max_depth,
   min_samples_leaf y max_features, por F1-macro medio entre folds.
2. Validación cruzada de la combinación ganadora con el formato estándar
   (decisión 005): métricas, matriz de confusión y tiempos.
3. Pipeline final entrenado con todo train -> models/random_forest.joblib.
4. Permutation importance de las columnas del modelo en el fold de
   validación de cada fold, promediada entre folds.

Salidas: reports/metrics/random_forest.json y
reports/metrics/random_forest_importancia.csv.
"""

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.config import METRICS_DIR, RANDOM_FOREST, RANDOM_STATE
from src.data import build_target, cv_folds, load_raw, load_splits
from src.evaluation import cross_validate_model
from src.features import model_columns, select_features
from src.models import build_pipeline, save_pipeline
from src.training import fit_full_train, model_report, save_metrics

# Espacio de búsqueda: 4 x 5 x 6 x 4 = 480 combinaciones.
SEARCH_SPACE = {
    "modelo__n_estimators": [100, 200, 300, 500],
    "modelo__max_depth": [5, 10, 15, 20, None],
    "modelo__min_samples_leaf": [1, 2, 5, 10, 20, 50],
    "modelo__max_features": ["sqrt", "log2", 0.3, 0.5],
}
# Combinaciones evaluadas: límite por criterio Green AI (issue #11).
N_ITER = 25

N_REPEATS = 5
IMPORTANCE_PATH = METRICS_DIR / f"{RANDOM_FOREST}_importancia.csv"
N_TOP = 10


def build_rf(**params) -> Pipeline:
    """Pipeline sin ajustar del Random Forest."""
    model = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    return build_pipeline(model.set_params(**params))


def search(
    X: pd.DataFrame, y: pd.Series, folds, n_iter: int = N_ITER, space: dict = SEARCH_SPACE
) -> tuple[dict, pd.DataFrame, float]:
    """Búsqueda aleatoria por F1-macro medio con los folds fijos.

    Devuelve los hiperparámetros ganadores (sin el prefijo "modelo__"), una
    fila por combinación probada ordenada de mejor a peor y el tiempo total (s).
    """
    searcher = RandomizedSearchCV(
        build_rf(),
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


def importance_table(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds) -> pd.DataFrame:
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
            n_repeats=N_REPEATS,
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


def main() -> dict:
    df, splits = load_raw(), load_splits()
    X, y = select_features(df), build_target(df)
    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    folds = cv_folds(splits)

    print(f"1. Búsqueda aleatoria: {N_ITER} combinaciones x {len(folds)} folds (F1-macro)")
    best, search_table, search_time = search(X_train, y_train, folds)
    print(f"  Ganadora: {best} ({search_time:.0f} s)")

    print("2. Validación cruzada de la ganadora (formato estándar, umbral 0.5)")
    pipeline = build_rf(**best)
    folds_df, proba = cross_validate_model(pipeline, X_train, y_train, folds)

    print("3. Pipeline final con todo train")
    pipeline, fit_time = fit_full_train(pipeline, X_train, y_train)
    model_path = save_pipeline(pipeline, RANDOM_FOREST)
    print(f"  Guardado en {model_path}")

    print("4. Permutation importance en los folds de validación")
    importance = importance_table(build_rf(**best), X_train, y_train, folds)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_PATH, float_format="%.6f", encoding="utf-8")

    report = model_report(RANDOM_FOREST, pipeline, {"folds": folds_df, "proba": proba}, y_train)
    report["tiempos"]["entrenamiento_train_completo_s"] = fit_time
    report["tiempos"]["busqueda_total_s"] = search_time
    report["busqueda"] = {
        "metodo": "RandomizedSearchCV",
        "n_iter": N_ITER,
        "n_combinaciones_espacio": int(np.prod([len(v) for v in SEARCH_SPACE.values()])),
        "espacio": {p.removeprefix("modelo__"): v for p, v in SEARCH_SPACE.items()},
        "criterio": "mayor F1-macro medio entre los 5 folds fijos (umbral 0.5); decisión 006",
        "elegido": best,
        "resultados": search_table.to_dict(orient="records"),
    }
    report["importancia"] = {
        "metodo": f"permutation importance (F1-macro, {N_REPEATS} repeticiones) en el fold de validación de cada fold",
        "archivo": IMPORTANCE_PATH.name,
        "top": importance.head(N_TOP).round(4).reset_index().to_dict(orient="records"),
    }
    path = save_metrics(report)

    m = report["cv"]
    print(
        f"Random Forest (umbral 0.5): "
        f"F1-macro {m['f1_macro_media']:.3f} ± {m['f1_macro_std']:.3f}, "
        f"recall {m['recall_riesgo_media']:.3f} ± {m['recall_riesgo_std']:.3f}"
    )
    print(f"Métricas en {path}; importancia en {IMPORTANCE_PATH}.")
    return report


if __name__ == "__main__":
    main()
