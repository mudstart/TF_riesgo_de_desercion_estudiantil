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
  training.py    Ajuste de hiperparámetros y reporte estándar de métricas
  evaluation.py  Evaluación y comparación de modelos
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
