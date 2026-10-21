# Riesgo de deserción estudiantil

[![Pruebas](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/actions/workflows/pruebas.yml/badge.svg?branch=main)](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/actions/workflows/pruebas.yml) [![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/) [![uv](https://img.shields.io/badge/entorno-uv-DE5FE9.svg)](https://docs.astral.sh/uv/) [![Demo Streamlit](https://img.shields.io/badge/demo-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](#demo-de-predicción) [![Datos CC BY 4.0](https://img.shields.io/badge/datos-CC%20BY%204.0-lightgrey.svg)](https://doi.org/10.24432/C5MC89)

Trabajo final de la asignatura **INF-8239 Ciencia de Datos II** (Modalidad A, Tema 01).

## Descripción

Proyecto reproducible para identificar de forma temprana a los estudiantes con riesgo de deserción, usando el conjunto de datos *Predict Students' Dropout and Academic Success* (UCI ML Repository, id 697, CC BY 4.0).

El decisor es la coordinación académica. El análisis apoya la priorización de acompañamiento temprano con revisión humana; **no** debe usarse para sancionar, excluir ni decidir admisiones. Ver [docs/proyecto.md](docs/proyecto.md).

## Resultado principal

Con solo las 18 variables disponibles al ingreso (sin rendimiento semestral, para evitar la fuga), un **Random Forest con umbral 0.382** detecta en datos no vistos al **80 % de los estudiantes que abandonan** (recall 0.803, IC 95 % [0.755, 0.848]), con una precisión de 0.45 y un F1-macro de 0.61. Mejora al baseline en unos 3 desertores detectados por cada 100. Falla más con las mujeres, los menores de 21 años y tres carreras, así que su salida tiene tres niveles (riesgo, **no concluyente con revisión humana** y sin riesgo) y reglas explícitas para abstenerse, escalar o detener su uso. **Sirve para ordenar a quién contactar primero, nunca para descartar a nadie.**

Informe completo, legible sin abrir el código: [reports/informe_tecnico.md](reports/informe_tecnico.md).

## Versión 1.0

La versión entregada es el tag **`v1.0`**, con su [release en GitHub](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/releases/tag/v1.0). Para obtener exactamente esa versión:

```bash
git clone --branch v1.0 https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil.git
```

**Verificación de reproducibilidad (2026-10-20).** Se clonó el repositorio en una carpeta vacía y se siguieron los pasos de este README: `uv sync`, descarga de los datos (mismo SHA-256), auditoría, entrenamiento de los tres modelos, comparación en validación cruzada y análisis de errores. Las métricas, los hiperparámetros, la selección y las alertas salieron idénticos a los versionados, y las figuras, byte a byte iguales. La evaluación en test se negó a repetirse, como exige la decisión 008. Pasaron las 110 pruebas, y la demo funcionó con un perfil inventado en vivo.

## Integrantes y roles

| Integrante | GitHub | Responsabilidades (issues) |
|---|---|---|
| Roberto Byas | [RByas](https://github.com/RByas) | Registro del equipo, decisor y pregunta (#1); auditoría de datos (#6); pruebas de contrato y anti-fuga 1–3 (#10); análisis de errores, subgrupos y abstención (#14) |
| Manicaotex Mueses | [mannixmueses-cloud](https://github.com/mannixmueses-cloud) | Estructura y entorno (#2); baseline de regresión logística (#9); comparación y selección del modelo final (#13); medición Green AI (#15) |
| Jose M. Maduro | [mudstart](https://github.com/mudstart) | Dataset y ficha (#3); momento de intervención (#4); target (#5); partición (#7); pipeline sin fuga (#8); Random Forest (#11); Gradient Boosting (#12); pruebas 4–8 y CI (#16); demo (#17); informe, README y declaración de IA (#18); ensayo y defensa (#19) |

El detalle con enlaces a cada PR y a sus revisiones está en la [sección 14 del informe](reports/informe_tecnico.md#14-contribución-de-cada-integrante).

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
| 8. Análisis de errores y subgrupos (#14) | `uv run python -m src.errores` | `reports/metrics/analisis_errores.json` y `.md`, `reports/figures/14_*.png` | ~1.5 min |
| 9. Medición Green AI (#15) | `uv run python -m src.green_ai` | `reports/metrics/green_ai.json` y `.md`, `reports/figures/15_desempeno_vs_costo.png`; energía y emisiones en `reports/metrics/<modelo>.json` | ~15 min (3 repeticiones) |

Los pasos 3 a 5 son independientes entre sí, pero todos requieren el paso 1. El paso 6 requiere los pasos 3 a 5, y los pasos 8 y 9 requieren el 6 (solo usan train). Después del paso 9 se repite el paso 6 para que la tabla de #13 muestre la energía y las emisiones. El paso 7 ya se ejecutó y su resultado está versionado: el script se niega a repetirlo mientras exista `test_final.json` ([decisión 008](docs/decisiones/008_criterio_seleccion.md)). Los modelos (`models/`) no se versionan; las métricas (`reports/metrics/`) sí.

## Pruebas

```bash
uv run pytest
```

110 pruebas automáticas en unos 30 s. GitHub Actions las ejecuta en cada PR y en cada push a `main` (badge al inicio), después de descargar y verificar el dataset. Las 8 pruebas mínimas del proyecto:

| N.º | Qué protege | Archivo |
|---|---|---|
| 1 | Contrato del dataset: esquema, tipos, rangos y huella SHA-256 | `tests/test_data.py` |
| 2 | Anti-fuga: ninguna variable posterior al ingreso llega al modelo | `tests/test_data.py` |
| 3 | Partición válida, sin solapamiento y reproducible | `tests/test_data.py` |
| 4 | Pipeline sin fuga: transformar test no cambia lo aprendido con train | `tests/test_pipeline.py` |
| 5 | Reproducibilidad: misma semilla, mismas predicciones | `tests/test_model.py` |
| 6 | Salida válida: probabilidades en [0, 1] que suman 1 y clases 0/1 | `tests/test_model.py` |
| 7 | Robustez: categorías nuevas y rechazo claro de registros incompletos | `tests/test_pipeline.py` |
| 8 | Desempeño mínimo: el modelo final supera al `DummyClassifier` sin usar test | `tests/test_model.py` |

Criterios en las decisiones [010](docs/decisiones/010_pruebas_y_ci.md) y [003](docs/decisiones/003_particion_validacion.md).

## Demo de predicción

Una aplicación Streamlit (#17) que estima el riesgo de deserción de un estudiante al ingreso. Solo necesita el paso 1 (`data/raw/data.csv`); no necesita `models/`, porque reconstruye el modelo elegido en #13 con los hiperparámetros versionados (~1 s):

```bash
uv run streamlit run app/app.py
```

Se abre en el navegador (http://localhost:8501). Incluye tres perfiles ficticios, que no están en el dataset: uno de riesgo bajo, uno alto y uno no concluyente; también se pueden ingresar datos propios. Para cada perfil muestra:

- la probabilidad y el nivel: **bajo**, **no concluyente – requiere revisión humana** (zona de abstención del #14) o **alto**;
- una abstención, sin resultado, si un dato falta o está fuera de lo visto en train;
- las variables que más pesan en el caso y qué **no** permite concluir el resultado;
- avisos de revisión humana para los subgrupos donde el modelo falla más;
- el desempeño en test y la trazabilidad: versión del modelo, fecha de entrenamiento, hash del dataset y umbral.

Criterios en la [decisión 012](docs/decisiones/012_demo_prediccion.md).

## Documentación

| Documento | Contenido |
|---|---|
| [reports/informe_tecnico.md](reports/informe_tecnico.md) | **Informe técnico final**: problema, datos, metodología, resultados, subgrupos, Green AI, conclusión, uso responsable, trazabilidad y contribuciones |
| [docs/proyecto.md](docs/proyecto.md) | Decisor, pregunta de análisis, usos excluidos y registro inicial de riesgos (#1) |
| [docs/ficha_datos.md](docs/ficha_datos.md) | Procedencia, licencia, versión y limitaciones del dataset (#3) |
| [docs/decisiones/](docs/decisiones/README.md) | Registro de decisiones del proyecto (001–012) |
| [docs/declaracion_ia.md](docs/declaracion_ia.md) | Declaración de uso de herramientas de IA |
| [docs/guia_defensa.md](docs/guia_defensa.md) | Evidencia para la defensa individual, aporte de cada integrante y guion del ensayo de la demo |
| Informes parciales | [Auditoría de datos](reports/auditoria_datos.md) · [baseline](reports/baseline.md) · [comparación de modelos](reports/comparacion_modelos.md) · [errores y subgrupos](reports/analisis_errores.md) · [Green AI](reports/green_ai.md) |

## Estructura

```text
app/
  app.py       Demo de predicción con Streamlit (#17)
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
  errores.py     Análisis de errores y subgrupos, zona de abstención y revisión de la entrada (#14)
  green_ai.py    Costo computacional y energético de los tres modelos (#15)
  predict.py     Lógica de la demo: validación, predicción, categorías, factores y trazabilidad (#17)
  diccionario.py Etiquetas de los códigos del dataset UCI para el formulario de la demo (#17)
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

## Datos: licencia y cita

El dataset se distribuye bajo **CC BY 4.0**, que exige atribución:

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
