# Riesgo de deserción estudiantil

Trabajo final de la asignatura **INF-8239 Ciencia de Datos II** (Modalidad A, Tema 01).

## Descripción

Proyecto reproducible para identificar de forma temprana a los estudiantes con riesgo de deserción, usando el conjunto de datos *Predict Students' Dropout and Academic Success* (UCI ML Repository, id 697, CC BY 4.0).

El decisor es la coordinación académica. El análisis apoya la priorización de acompañamiento temprano con revisión humana; **no** debe usarse para sancionar, excluir ni decidir admisiones. Ver [docs/proyecto.md](docs/proyecto.md).

## Requisitos

- [uv](https://docs.astral.sh/uv/) como gestor de entorno. uv descarga Python **3.12** (fijado en `.python-version`) si no está instalado.
- Git

## Instalación

1. Instalar uv, si no está instalado ([guía oficial](https://docs.astral.sh/uv/getting-started/installation/)):

   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   # Linux / macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clonar el repositorio:

   ```bash
   git clone https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil.git
   cd TF_riesgo_de_desercion_estudiantil
   ```

3. Instalar el entorno y verificar con las pruebas (uv descarga Python 3.12 automáticamente según `.python-version`):

   ```bash
   uv sync
   uv run pytest
   ```

## Obtener los datos

Los datos crudos no se versionan. Para descargarlos a `data/raw/data.csv` y verificar su huella SHA-256:

```bash
uv run python -m src.data
```

Si el archivo ya existe y es válido, no se vuelve a descargar. El mismo comando crea la partición fija `data/processed/splits.json` si todavía no existe. Procedencia, licencia (CC BY 4.0, con atribución obligatoria), cobertura y limitaciones del dataset: [docs/ficha_datos.md](docs/ficha_datos.md).

## Ejecutar el proyecto

Desde la raíz del repositorio, en este orden. Cada paso es reproducible (semilla fija) y usa solo train y los 5 folds fijos; el conjunto de test no se usa hasta la evaluación final (#13).

| Paso | Comando | Resultado | Tiempo aprox. |
|---|---|---|---|
| 1. Datos y partición | `uv run python -m src.data` | `data/raw/data.csv` y `data/processed/splits.json` | segundos |
| 2. Auditoría de datos (EDA, #6) | `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_auditoria_datos.ipynb` | Notebook ejecutado y figuras en `reports/figures/` | < 1 min |
| 3. Baseline: regresión logística (#9) | `uv run python -m src.models.train_baseline` | `models/baseline.joblib`, `reports/metrics/baseline.json` y `baseline_coeficientes.csv` | < 1 min |
| 4. Candidato 1: Random Forest (#11) | `uv run python -m src.models.train_rf` | `models/random_forest.joblib`, `reports/metrics/random_forest.json` y `random_forest_importancia.csv` | ~2 min |
| 5. Candidato 2: Gradient Boosting (#12) | `uv run python -m src.models.train_gb` | `models/gradient_boosting.joblib`, `reports/metrics/gradient_boosting.json` y `gradient_boosting_importancia.csv` | ~1 min |
| 6. Comparación con CV y selección (#13) | `uv run python -m src.comparacion` | `reports/metrics/comparacion_cv.json` y `.md`, `reports/figures/13_*.png` | ~1 min |
| 7. Evaluación final en test (#13), **una sola vez** | `uv run python -m src.comparacion --test` | `reports/metrics/test_final.json` y `reports/figures/13_confusion_test.png` | segundos |
| 8. Análisis de errores y subgrupos (#14) | `uv run python -m src.errores` | `reports/metrics/analisis_errores.json`, `subgrupos.csv` y `reports/figures/14_*.png` | ~1 min |

Los pasos 3 a 5 son independientes entre sí, pero todos requieren el paso 1. El paso 6 requiere los pasos 3 a 5. El paso 8 solo requiere los pasos 1 y 6 (lee los hiperparámetros y el umbral de `reports/metrics/`) y no usa el test. El paso 7 ya se ejecutó y su resultado está versionado: el script se niega a repetirlo mientras exista `test_final.json` ([decisión 008](docs/decisiones/008_criterio_seleccion.md)). Los modelos (`models/`) no se versionan; las métricas (`reports/metrics/`) sí.

Para comprobar que todo funciona:

```bash
uv run pytest
```

## Estructura

```text
data/
  raw/         Datos originales (no versionados)
  processed/   Datos procesados
docs/          Documentación del proyecto y enunciado (docs/enunciado/)
models/        Modelos entrenados (no versionados)
notebooks/     Exploración, numerados: 01_, 02_, ...
reports/
  figures/     Figuras del informe
  metrics/     Métricas de evaluación
src/
  config.py      Configuración global (semilla)
  data.py        Carga, validación y preparación de datos
  features.py    Transformación de características
  models/        Construcción de modelos (__init__.py) y un script de entrenamiento por modelo
    train_baseline.py  Baseline de regresión logística (#9)
    train_rf.py        Candidato 1: Random Forest (#11)
    train_gb.py        Candidato 2: Gradient Boosting (#12)
  training.py    Ajuste de hiperparámetros y reporte estándar de métricas
  evaluation.py  Evaluación y comparación de modelos
  comparacion.py Comparación final, selección y evaluación única en test (#13)
  errores.py     Análisis de errores, subgrupos, zona de abstención y condiciones de uso (#14)
tests/         Pruebas automáticas con pytest
```

## Reproducibilidad

La semilla global está centralizada en `src/config.py`:

```python
RANDOM_STATE = 42
```

Todo código con componentes aleatorios (particiones, validación cruzada, modelos) debe importarla en lugar de escribir el valor a mano:

```python
from src.config import RANDOM_STATE
```

La partición train/test y los 5 folds de validación cruzada están fijados en `data/processed/splits.json` (se crea con `uv run python -m src.data` si no existe). Todos los modelos deben usarlos, sin crear particiones propias:

```python
from src.data import cv_folds, load_splits

splits = load_splits()  # splits["train"], splits["test"], cv_folds(splits)
```

**El conjunto de test se usa una sola vez, en la evaluación final (#13).** Ver [docs/decisiones/003_particion_validacion.md](docs/decisiones/003_particion_validacion.md).

Todo modelo se construye con `build_pipeline`, que añade el preprocesamiento sin fuga y descarta las variables prohibidas. El pipeline se ajusta solo con train (o con el fold de entrenamiento) y el desbalance se maneja con `class_weight="balanced"`:

```python
from sklearn.linear_model import LogisticRegression
from src.models import build_pipeline

pipe = build_pipeline(LogisticRegression(class_weight="balanced", max_iter=5000))
```

Ver [docs/decisiones/004_preprocesamiento.md](docs/decisiones/004_preprocesamiento.md).

## Convenciones de trabajo

- **Ramas:** `feature/<num-issue>-descripcion-corta` (ej. `feature/2-estructura`).
- **Pull Requests:** todo cambio entra a `main` por PR, con al menos 1 revisión de otro integrante.
- **Commits:** descriptivos y con referencia al issue (ej. `feat: agrega baseline logístico #9`).
- **Cierre de issues:** la descripción del PR incluye `Closes #N`.
- **Decisiones (obligatorio):** todo PR que decida sobre datos, variables, target, métricas, validación o preprocesamiento, o que afecte el trabajo de otro integrante, incluye su documento en `docs/decisiones/`. Sin él, el revisor no aprueba el PR. Ver [docs/decisiones/README.md](docs/decisiones/README.md).
- **Notebooks:** numerados en orden de ejecución (`01_eda.ipynb`, `02_...`).
- **Datos y modelos:** los datos crudos y los modelos pesados no se suben al repositorio.
