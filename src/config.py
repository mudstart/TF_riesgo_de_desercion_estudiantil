"""Configuración global y reproducible del proyecto."""

from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset UCI 697 (ver docs/ficha_datos.md)
DATA_URL = "https://archive.ics.uci.edu/static/public/697/data.csv"
DATA_SHA256 = "a1b1a6531bbb93a5c7fdf0093b47172776652a4ffb12342fb79655c85b74801b"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"

# Variable objetivo (ver docs/decisiones/002_formulacion_target.md)
TARGET_COL = "Target"
RISK_CLASS = "Dropout"
TARGET_CLASSES = ("Graduate", "Dropout", "Enrolled")
