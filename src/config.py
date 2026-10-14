"""Configuración global y reproducible del proyecto."""

from pathlib import Path

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset UCI 697 (ver docs/ficha_datos.md)
DATA_URL = "https://archive.ics.uci.edu/static/public/697/data.csv"
DATA_SHA256 = "a1b1a6531bbb93a5c7fdf0093b47172776652a4ffb12342fb79655c85b74801b"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"

# Partición fija (ver docs/decisiones/003_particion_validacion.md)
SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits.json"
TEST_SIZE = 0.2
N_FOLDS = 5

# Variable objetivo (ver docs/decisiones/002_formulacion_target.md)
TARGET_COL = "Target"
RISK_CLASS = "Dropout"
TARGET_CLASSES = ("Graduate", "Dropout", "Enrolled")

# Variables según su disponibilidad temporal
# (ver docs/decisiones/001_momento_intervencion.md)

# Disponibles al ingreso: modelo principal.
FEATURES_INGRESO = (
    "Marital Status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Age at enrollment",
    "International",
)

# Momento de medición no documentado: solo en el análisis de sensibilidad.
FEATURES_SENSIBILIDAD = (
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
)

FEATURES_SEM1 = (
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
)

FEATURES_SEM2 = (
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
)

# Posteriores al momento de intervención: nunca entran en un modelo.
FEATURES_PROHIBIDAS = FEATURES_SEM1 + FEATURES_SEM2
