"""Candidato 2: Gradient Boosting con class_weight="balanced" (#12).

HistGradientBoostingClassifier de scikit-learn (decisiones 004 y 007),
comparado con el baseline (#9) y el Random Forest (#11) en #13. Desde la raíz:

    uv run python -m src.models.train_gb

Solo usa train y los 5 folds fijos (decisión 003). Pasos (decisiones 006 y 007):

1. Búsqueda aleatoria de 25 combinaciones de learning_rate, max_depth,
   min_samples_leaf, l2_regularization y max_features, por F1-macro medio
   entre folds. El número de árboles lo fija el early stopping con un 10 %
   estratificado del train de cada ajuste (nunca el fold de validación ni test).
2. Validación cruzada de la combinación ganadora con el formato estándar
   (decisión 005), y número de árboles que el early stopping usa en cada fold.
3. Pipeline final entrenado con todo train -> models/gradient_boosting.joblib.
4. Permutation importance de las columnas del modelo en el fold de
   validación de cada fold, promediada entre folds.

Salidas: reports/metrics/gradient_boosting.json y
reports/metrics/gradient_boosting_importancia.csv.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.config import GRADIENT_BOOSTING, METRICS_DIR, RANDOM_STATE
from src.data import build_target, cv_folds, load_raw, load_splits
from src.evaluation import cross_validate_model
from src.features import select_features
from src.models import build_pipeline, save_pipeline
from src.training import fit_full_train, model_report, permutation_importance_table, random_search, save_metrics

# Early stopping: hasta MAX_ITER árboles; se detiene si la pérdida en el 10 %
# de validación interna no mejora durante N_ITER_NO_CHANGE iteraciones.
EARLY_STOPPING = {
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "max_iter": 500,
}

# Espacio de búsqueda: 4 x 4 x 4 x 4 x 3 = 768 combinaciones.
# HistGradientBoosting no tiene subsample de filas; max_features submuestrea columnas.
SEARCH_SPACE = {
    "modelo__learning_rate": [0.02, 0.05, 0.1, 0.2],
    "modelo__max_depth": [3, 5, 8, None],
    "modelo__min_samples_leaf": [10, 20, 50, 100],
    "modelo__l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "modelo__max_features": [0.5, 0.8, 1.0],
}
# Combinaciones evaluadas: límite por criterio Green AI (decisión 006).
N_ITER = 25

N_REPEATS = 5
IMPORTANCE_PATH = METRICS_DIR / f"{GRADIENT_BOOSTING}_importancia.csv"
N_TOP = 10


def build_gb(**params) -> Pipeline:
    """Pipeline sin ajustar del Gradient Boosting, con early stopping."""
    model = HistGradientBoostingClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, **EARLY_STOPPING
    )
    return build_pipeline(model.set_params(**params))


def search(
    X: pd.DataFrame, y: pd.Series, folds, n_iter: int = N_ITER, space: dict = SEARCH_SPACE
) -> tuple[dict, pd.DataFrame, float]:
    """Búsqueda aleatoria del Gradient Boosting (ver random_search)."""
    return random_search(build_gb(), space, X, y, folds, n_iter)


def early_stopping_iterations(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, folds) -> list[int]:
    """Número de árboles que el early stopping deja en cada fold de entrenamiento."""
    return [
        int(clone(pipeline).fit(X.iloc[tr], y.iloc[tr]).named_steps["modelo"].n_iter_)
        for tr, _ in folds
    ]


def main() -> dict:
    df, splits = load_raw(), load_splits()
    X, y = select_features(df), build_target(df)
    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    folds = cv_folds(splits)

    print(f"1. Búsqueda aleatoria: {N_ITER} combinaciones x {len(folds)} folds (F1-macro)")
    best, search_table, search_time = search(X_train, y_train, folds)
    print(f"  Ganadora: {best} ({search_time:.0f} s)")

    print("2. Validación cruzada de la ganadora (formato estándar, umbral 0.5)")
    pipeline = build_gb(**best)
    folds_df, proba = cross_validate_model(pipeline, X_train, y_train, folds)
    iterations = early_stopping_iterations(pipeline, X_train, y_train, folds)
    print(f"  Árboles por fold (early stopping): {iterations}")

    print("3. Pipeline final con todo train")
    pipeline, fit_time = fit_full_train(pipeline, X_train, y_train)
    n_trees = int(pipeline.named_steps["modelo"].n_iter_)
    model_path = save_pipeline(pipeline, GRADIENT_BOOSTING)
    print(f"  {n_trees} árboles. Guardado en {model_path}")

    print("4. Permutation importance en los folds de validación")
    importance = permutation_importance_table(build_gb(**best), X_train, y_train, folds, N_REPEATS)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(IMPORTANCE_PATH, float_format="%.6f", encoding="utf-8")

    report = model_report(GRADIENT_BOOSTING, pipeline, {"folds": folds_df, "proba": proba}, y_train)
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
    report["early_stopping"] = {
        **EARLY_STOPPING,
        "validacion": "10 % estratificado del train de cada ajuste; nunca el fold de validación ni el test",
        "arboles_por_fold": iterations,
        "arboles_modelo_final": n_trees,
    }
    report["importancia"] = {
        "metodo": f"permutation importance (F1-macro, {N_REPEATS} repeticiones) en el fold de validación de cada fold",
        "archivo": IMPORTANCE_PATH.name,
        "top": importance.head(N_TOP).round(4).reset_index().to_dict(orient="records"),
    }
    path = save_metrics(report)

    m = report["cv"]
    print(
        f"Gradient Boosting (umbral 0.5): "
        f"F1-macro {m['f1_macro_media']:.3f} ± {m['f1_macro_std']:.3f}, "
        f"recall {m['recall_riesgo_media']:.3f} ± {m['recall_riesgo_std']:.3f}"
    )
    print(f"Métricas en {path}; importancia en {IMPORTANCE_PATH}.")
    return report


if __name__ == "__main__":
    main()
