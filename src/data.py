"""Funciones para carga, validación y preparación de datos.

Descarga del dataset y creación de la partición fija (si no existe) desde la
raíz del proyecto:

    uv run python -m src.data
"""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import (
    DATA_SHA256,
    DATA_URL,
    N_FOLDS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    RISK_CLASS,
    SPLITS_PATH,
    TARGET_CLASSES,
    TARGET_COL,
    TEST_SIZE,
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


def make_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE,
    data_sha256: str | None = None,
) -> dict:
    """Crea la partición fija según docs/decisiones/003_particion_validacion.md.

    Train/test estratificado por el Target original de 3 clases, y folds de
    validación cruzada estratificados definidos solo dentro de train. Los
    índices son posiciones de fila (0 a n-1) en el CSV original.
    """
    y = build_target(df, binary=False).to_numpy()
    positions = np.arange(len(df))
    train, test = train_test_split(
        positions, test_size=test_size, stratify=y, random_state=random_state
    )
    train, test = np.sort(train), np.sort(test)

    folds = np.empty(len(train), dtype=int)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for k, (_, val) in enumerate(skf.split(train, y[train])):
        folds[val] = k

    return {
        "data_sha256": data_sha256,
        "n_rows": len(df),
        "stratify": f"{TARGET_COL} (3 clases)",
        "test_size": test_size,
        "n_folds": n_folds,
        "random_state": random_state,
        "train": train.tolist(),
        "test": test.tolist(),
        "folds": folds.tolist(),
    }


def save_splits(splits: dict, path: Path = SPLITS_PATH) -> Path:
    """Guarda la partición en JSON, una clave por línea."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in splits.items()]
    path.write_text("{\n" + ",\n".join(lines) + "\n}\n", encoding="utf-8")
    return path


def load_splits(path: Path = SPLITS_PATH, expected_sha256: str | None = DATA_SHA256) -> dict:
    """Lee y valida la partición fija que usa todo el proyecto.

    Devuelve un dict con "train" y "test" (posiciones de fila) y "folds" (fold
    de cada posición de train) como arrays de numpy. Lanza ValueError si la
    partición no corresponde al dataset esperado o es inconsistente.

    Regla de equipo: "test" se usa una sola vez, en la evaluación final (#13).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Créela con: uv run python -m src.data")
    splits = json.loads(path.read_text(encoding="utf-8"))

    if expected_sha256 is not None and splits["data_sha256"] != expected_sha256:
        raise ValueError(
            "La partición se creó con otra versión del dataset "
            f"(SHA-256 {splits['data_sha256']}, se esperaba {expected_sha256})."
        )

    train = np.array(splits["train"], dtype=int)
    test = np.array(splits["test"], dtype=int)
    folds = np.array(splits["folds"], dtype=int)

    if np.intersect1d(train, test).size:
        raise ValueError("Train y test se solapan.")
    if not np.array_equal(np.sort(np.concatenate([train, test])), np.arange(splits["n_rows"])):
        raise ValueError("Train y test no cubren exactamente todas las filas.")
    if len(folds) != len(train) or set(folds.tolist()) != set(range(splits["n_folds"])):
        raise ValueError("Los folds no corresponden a las filas de train.")

    return {**splits, "train": train, "test": test, "folds": folds}


def cv_folds(splits: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds para el parámetro cv= de scikit-learn.

    Las posiciones son relativas al subconjunto de train, es decir, para usar
    con X.iloc[splits["train"]] e y.iloc[splits["train"]].
    """
    folds = splits["folds"]
    return [
        (np.flatnonzero(folds != k), np.flatnonzero(folds == k))
        for k in range(splits["n_folds"])
    ]


if __name__ == "__main__":
    path = download_dataset()
    df = load_raw(path)
    print(f"Dataset listo en {path}: {df.shape[0]} filas x {df.shape[1]} columnas.")

    if SPLITS_PATH.exists():
        load_splits()
        print(f"Partición válida en {SPLITS_PATH}.")
    else:
        save_splits(make_splits(df, data_sha256=file_sha256(path)))
        print(f"Partición creada en {SPLITS_PATH}.")
