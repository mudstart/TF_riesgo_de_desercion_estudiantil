# Decisión 004: preprocesamiento sin fuga

- Issue: [#8](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/8) Pipeline de preprocesamiento sin fuga.
- Rama: `feature/8-pipeline-sin-fuga`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-14.
- Estado: Propuesta.
- Reemplaza a: No aplica.

## 1. Contexto

Una fuga habitual es calcular estadísticas de preprocesamiento (medias para escalar, categorías para codificar) con todo el dataset antes de partir: el modelo recibe así información del test. Además, los tres modelos (#9, #11, #12) deben compararse con el mismo preprocesamiento.

La auditoría de datos (#6, [reports/auditoria_datos.md](../../reports/auditoria_datos.md), sección 8) dejó cuatro propuestas para este issue: excluir dos columnas redundantes (7), usar one-hot para las nominales (9), agrupar categorías raras aprendiendo solo con train (10) y preferir `class_weight` y umbral antes que remuestreo (4). También recomendó conservar un imputador para datos nuevos (1).

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Nominales (códigos enteros) | Usar el código como número | Simple | Inventa un orden y distancias inexistentes (curso 33 vs. 9991) |
| | **One-hot** | Respeta que son categorías | Más columnas (112 en total) |
| Categorías raras | Una columna por categoría | Sin pérdida | Columnas con 1–9 casos: ruido y sobreajuste |
| | **Agrupar en "infrecuentes" con train** | Menos ruido; categorías nuevas tienen destino | Pierde el detalle de categorías muy pequeñas |
| Categorías desconocidas | `handle_unknown="ignore"` (propuesto en el issue) | Simple | Una categoría nueva queda como "ninguna categoría" |
| | **`"infrequent_if_exist"`** | Una categoría nueva se trata como infrecuente | Ninguna relevante |
| Desbalance | SMOTE (con `imblearn`) | Genera ejemplos sintéticos | Nueva dependencia; ejemplos artificiales |
| | **`class_weight="balanced"`** | Sin dependencias; combinable con ajuste de umbral (#13) | `GradientBoostingClassifier` no lo admite |

## 3. Decisión

El pipeline es un `ColumnTransformer` seguido del modelo (`build_pipeline` en `src/models.py`, `build_preprocessor` en `src/features.py`):

| Grupo | Columnas (modelo principal) | Transformación |
|---|---|---|
| Nominales | Marital Status, Application mode, Course, Previous qualification, Nacionality, Mother's/Father's qualification, Mother's/Father's occupation | Imputación por moda + `OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist")` |
| Numéricas | Application order, Previous qualification (grade), Admission grade, Age at enrollment | Imputación por mediana + `StandardScaler` |
| Binarias | Displaced, Educational special needs, Gender | Imputación por moda, sin transformar |
| Descartadas | `International` y `Daytime/evening attendance` (redundantes), las 12 de semestre (prohibidas), `Target` y cualquier otra | `remainder="drop"` |

En el análisis de sensibilidad (`sensibilidad=True`) se añaden Debtor, Tuition fees up to date y Scholarship holder como binarias, y Unemployment rate, Inflation rate y GDP como numéricas.

- **Desbalance:** cada modelo se crea con `class_weight="balanced"`. Para el Gradient Boosting (#12) se usa `HistGradientBoostingClassifier`, que admite ese parámetro. No se usa SMOTE.
- **Ajuste:** `fit` solo se llama con `X.iloc[splits["train"]]` o, en validación cruzada, con el fold de entrenamiento que entrega `cv_folds()`.
- **Serialización:** `save_pipeline()` y `load_pipeline()` guardan el pipeline completo con `joblib` en `models/` (no versionado).
- La clasificación por tipo y la lista de redundantes están en `src/config.py`.

## 4. Justificación

1. **Sin fuga por construcción.** Imputadores, escalador y codificador aprenden sus estadísticas solo en `fit`. Al estar dentro de un `Pipeline`, `cross_validate` los reajusta en cada fold de entrenamiento y el fold de validación solo pasa por `transform`.
2. **Columnas prohibidas: se ignoran y se rechazan.** El `ColumnTransformer` solo toma las columnas listadas, así que el pipeline puede recibir el DataFrame completo y descarta lo demás. Además, `build_preprocessor` lanza error si una variable prohibida llegara a la lista del modelo (por ejemplo, por un cambio en `src/config.py`).
3. **Redundantes fuera del modelo.** `International` vale 1 si y solo si `Nacionality` ≠ 1, y `Daytime/evening attendance` queda determinada por `Course` (#6). Con one-hot son combinaciones lineales exactas de otras columnas y no aportan información. Se conservan en `FEATURES_INGRESO` para describir subgrupos (#14).
4. **Umbral de 10 casos.** Es el criterio usado en la auditoría (#6, sección 6), donde agrupar reduce, por ejemplo, la ocupación del padre de 46 a 15 categorías. Se aprende con train para no filtrar información del test.
5. **Imputador sin nulos.** El dataset no tiene nulos, por lo que no cambia ningún resultado; protege la demo (#17) ante entradas incompletas.
6. **Un preprocesador para los 3 modelos.** El escalado no es necesario para los árboles, pero no les afecta, y usar el mismo preprocesamiento hace la comparación justa.

## 5. Evidencia

Validación cruzada con los 5 folds fijos de la [decisión 003](003_particion_validacion.md), solo sobre train (3,539 filas), formulación binaria, umbral 0.5, `class_weight="balanced"` e hiperparámetros por defecto. Verifica que el pipeline funciona; **no** es la comparación de modelos (#13).

| Escenario | Modelo | F1-macro | Recall Dropout | Precisión Dropout |
|---|---|---|---|---|
| Principal (16 columnas) | LogisticRegression | 0.673 | 0.659 | 0.523 |
| Principal | RandomForestClassifier | 0.662 | 0.527 | 0.550 |
| Principal | HistGradientBoostingClassifier | 0.667 | 0.593 | 0.533 |
| Sensibilidad (22 columnas) | LogisticRegression | 0.734 | 0.704 | 0.609 |
| Sensibilidad | RandomForestClassifier | 0.742 | 0.612 | 0.677 |
| Sensibilidad | HistGradientBoostingClassifier | 0.733 | 0.636 | 0.639 |

Con el one-hot, la regresión logística del escenario principal sube de F1-macro 0.639 (códigos como números, [decisión 001](001_momento_intervencion.md)) a 0.673. Tras el ajuste, el preprocesador produce 112 columnas.

## 6. Consecuencias

- Los modelos #9, #11 y #12 se construyen con `build_pipeline(modelo)` y `class_weight="balanced"`; #12 usa `HistGradientBoostingClassifier`.
- El análisis de sensibilidad usa `build_pipeline(modelo, sensibilidad=True)`.
- Las pruebas de pipeline (#16) pueden apoyarse en `tests/test_pipeline.py`.
- El umbral de decisión se fija en #13 con validación cruzada, no aquí.

Uso típico:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src.data import build_target, cv_folds, load_raw, load_splits
from src.models import build_pipeline

df, splits = load_raw(), load_splits()
y = build_target(df)
X_train, y_train = df.iloc[splits["train"]], y.iloc[splits["train"]]

pipe = build_pipeline(LogisticRegression(class_weight="balanced", max_iter=5000))
cross_validate(pipe, X_train, y_train, cv=cv_folds(splits), scoring="f1_macro")
```

## 7. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | POR COMPLETAR: enlace al PR |
