"""Funciones para carga, validación y preparación de datos."""

import pandas as pd

from src.config import RISK_CLASS, TARGET_CLASSES, TARGET_COL


def build_target(df: pd.DataFrame, binary: bool = True) -> pd.Series:
    """Construye la variable objetivo según docs/decisiones/002_formulacion_target.md.

    Con binary=True (formulación principal) devuelve 1 para Dropout y 0 para
    Graduate o Enrolled. Con binary=False (formulación secundaria) devuelve las
    tres clases originales. No modifica el DataFrame recibido.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Falta la columna objetivo '{TARGET_COL}'.")

    target = df[TARGET_COL]
    unknown = set(target.unique()) - set(TARGET_CLASSES)
    if unknown:
        raise ValueError(
            f"Clases desconocidas en '{TARGET_COL}': {sorted(map(str, unknown))}. "
            f"Se esperaban: {list(TARGET_CLASSES)}."
        )

    if binary:
        return (target == RISK_CLASS).astype(int).rename("dropout")
    return target.copy()
