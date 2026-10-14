"""Funciones para transformación e ingeniería de características."""

import pandas as pd

from src.config import FEATURES_INGRESO, FEATURES_SENSIBILIDAD


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
