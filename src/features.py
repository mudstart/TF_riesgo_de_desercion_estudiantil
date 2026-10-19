"""Funciones para transformación e ingeniería de características."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FEATURES_BINARIAS,
    FEATURES_INGRESO,
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    FEATURES_PROHIBIDAS,
    FEATURES_REDUNDANTES,
    FEATURES_SENSIBILIDAD,
    MIN_FREQUENCY,
)


def feature_columns(sensibilidad: bool = False) -> list[str]:
    """Columnas permitidas según docs/decisiones/001_momento_intervencion.md.

    Por defecto, solo las disponibles al ingreso (modelo principal). Con
    sensibilidad=True se añaden las de momento no documentado, únicamente
    para el análisis de sensibilidad. Nunca incluye variables de semestre.
    """
    columns = list(FEATURES_INGRESO)
    if sensibilidad:
        columns += FEATURES_SENSIBILIDAD
    return columns


def select_features(df: pd.DataFrame, sensibilidad: bool = False) -> pd.DataFrame:
    """Devuelve una copia de df solo con las columnas permitidas.

    Lanza ValueError si falta alguna columna esperada.
    """
    columns = feature_columns(sensibilidad)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}.")
    return df[columns].copy()


def model_columns(sensibilidad: bool = False) -> list[str]:
    """Columnas que entran al modelo: las permitidas sin las redundantes.

    Ver docs/decisiones/004_preprocesamiento.md.
    """
    return [c for c in feature_columns(sensibilidad) if c not in FEATURES_REDUNDANTES]


class ValidarColumnas(BaseEstimator, TransformerMixin):
    """Rechaza registros a los que les falta alguna columna que usa el modelo.

    Primer paso del pipeline: deja pasar X sin cambios y, si falta alguna de
    `columns`, lanza un ValueError que las nombra, en lugar del error genérico
    de scikit-learn en el preprocesador.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self._check(X)
        self.columns_ = list(self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check(X)
        return X

    def _check(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Registro incompleto: faltan las columnas {missing} que necesita el modelo.")


def build_preprocessor(sensibilidad: bool = False) -> ColumnTransformer:
    """Preprocesador sin ajustar según docs/decisiones/004_preprocesamiento.md.

    - Nominales: imputación por moda y one-hot; las categorías con menos de
      MIN_FREQUENCY casos en el ajuste, y las desconocidas, van a una sola
      columna de "infrecuentes".
    - Numéricas: imputación por mediana y escalado estándar.
    - Binarias: imputación por moda, sin transformar.
    - Cualquier otra columna (prohibidas, redundantes, Target) se descarta.

    Todas las estadísticas se aprenden en fit, que solo debe llamarse con
    train o con el fold de entrenamiento.
    """
    columns = model_columns(sensibilidad)

    leaked = set(columns) & set(FEATURES_PROHIBIDAS)
    if leaked:
        raise ValueError(f"Variables prohibidas en el modelo: {sorted(leaked)}.")
    unclassified = set(columns) - set(FEATURES_NOMINALES + FEATURES_NUMERICAS + FEATURES_BINARIAS)
    if unclassified:
        raise ValueError(f"Columnas sin tipo asignado: {sorted(unclassified)}.")

    nominales = [c for c in columns if c in FEATURES_NOMINALES]
    numericas = [c for c in columns if c in FEATURES_NUMERICAS]
    binarias = [c for c in columns if c in FEATURES_BINARIAS]

    return ColumnTransformer(
        [
            (
                "nominales",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(
                        min_frequency=MIN_FREQUENCY,
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                ),
                nominales,
            ),
            ("numericas", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numericas),
            ("binarias", SimpleImputer(strategy="most_frequent"), binarias),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
