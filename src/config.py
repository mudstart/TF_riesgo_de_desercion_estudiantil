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

# Preprocesamiento (ver docs/decisiones/004_preprocesamiento.md)

# Permitidas pero redundantes de forma exacta (International se deduce de
# Nacionality y Daytime/evening attendance de Course): fuera del modelo,
# disponibles para describir subgrupos.
FEATURES_REDUNDANTES = ("International", "Daytime/evening attendance")

# Códigos enteros sin orden: one-hot.
FEATURES_NOMINALES = (
    "Marital Status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
)

# Magnitudes: escalado estándar.
FEATURES_NUMERICAS = (
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
)

# 0/1: sin transformar.
FEATURES_BINARIAS = (
    "Displaced",
    "Educational special needs",
    "Gender",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
)

# Categorías con menos casos que este mínimo en train se agrupan en una sola.
MIN_FREQUENCY = 10

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

# Nombre con que el baseline (#9) guarda su pipeline (save_pipeline).
BASELINE = "baseline"

# Nombre con que el Random Forest (#11) guarda su pipeline y sus métricas.
RANDOM_FOREST = "random_forest"

# Nombre con que el Gradient Boosting (#12) guarda su pipeline y sus métricas.
GRADIENT_BOOSTING = "gradient_boosting"

# Comparación y selección del modelo final (#13)
# (ver docs/decisiones/008_criterio_seleccion.md)

# Modelos comparados, del más simple al más complejo: este orden desempata.
MODELOS_COMPARADOS = (BASELINE, RANDOM_FOREST, GRADIENT_BOOSTING)

# Piso de F1-macro: cada modelo usa el umbral de mayor recall de riesgo que
# lo respeta, y solo es elegible si lo alcanza.
MIN_F1_MACRO = 0.62

# Remuestreos del bootstrap pareado y nivel de los intervalos.
N_BOOTSTRAP = 2000
CONFIDENCE = 0.95
