"""Funciones para carga, validación y preparación de datos.

Descarga del dataset desde la raíz del proyecto:

    uv run python -m src.data
"""

import hashlib
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_SHA256,
    DATA_URL,
    RAW_DATA_PATH,
    RISK_CLASS,
    TARGET_CLASSES,
    TARGET_COL,
)


def file_sha256(path: Path) -> str:
    """Calcula la huella SHA-256 de un archivo."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    dest: Path = RAW_DATA_PATH,
    url: str = DATA_URL,
    expected_sha256: str = DATA_SHA256,
    force: bool = False,
) -> Path:
    """Descarga el dataset UCI 697 a data/raw/ y verifica su SHA-256.

    Si el archivo ya existe con la huella esperada no se vuelve a descargar,
    salvo con force=True. Si la huella descargada no coincide, no se guarda
    nada y se lanza ValueError: UCI pudo haber publicado otra versión y el
    cambio debe documentarse en docs/ficha_datos.md.
    """
    dest = Path(dest)
    if dest.exists() and not force and file_sha256(dest) == expected_sha256:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
            shutil.copyfileobj(response, f)

        actual = file_sha256(tmp)
        if actual != expected_sha256:
            raise ValueError(
                f"SHA-256 inesperado para {url}: {actual} "
                f"(se esperaba {expected_sha256}). UCI pudo haber actualizado "
                "el dataset: documentar el cambio antes de usarlo."
            )
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lee el CSV original (separador coma, UTF-8 con BOM)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Descárguelo con: uv run python -m src.data"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


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


if __name__ == "__main__":
    path = download_dataset()
    df = load_raw(path)
    print(f"Dataset listo en {path}: {df.shape[0]} filas x {df.shape[1]} columnas.")
