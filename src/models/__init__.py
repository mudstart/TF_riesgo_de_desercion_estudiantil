"""Funciones para construcción y entrenamiento de modelos."""

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR
from src.features import ValidarColumnas, build_preprocessor, model_columns


def build_pipeline(model: BaseEstimator, sensibilidad: bool = False) -> Pipeline:
    """Une el preprocesador sin fuga con un modelo (ver decisión 004).

    El mismo preprocesador sirve para cualquier clasificador de scikit-learn.
    El desbalance se maneja en el modelo con class_weight="balanced". Antes
    del preprocesador, ValidarColumnas rechaza registros incompletos.
    Ajustar solo con train o con el fold de entrenamiento:

        pipe = build_pipeline(LogisticRegression(class_weight="balanced"))
        cross_validate(pipe, X_train, y_train, cv=cv_folds(splits))
    """
    return Pipeline(
        [
            ("validacion", ValidarColumnas(model_columns(sensibilidad))),
            ("preprocesamiento", build_preprocessor(sensibilidad)),
            ("modelo", model),
        ]
    )


def save_pipeline(pipeline: Pipeline, name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Guarda el pipeline completo en models/<name>.joblib (no versionado)."""
    path = Path(models_dir) / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(name: str, models_dir: Path = MODELS_DIR) -> Pipeline:
    """Carga un pipeline guardado con save_pipeline."""
    path = Path(models_dir) / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}.")
    return joblib.load(path)
