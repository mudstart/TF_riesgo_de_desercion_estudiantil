# Decisión 003: partición train/test y validación cruzada

- Issue: [#7](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/7) Partición estratificada train/test fija y validación cruzada.
- Rama: `feature/7-particion-train-test`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-13.
- Estado: Aprobada (PR #26).
- Reemplaza a: No aplica.

## 1. Contexto

Para comparar de forma justa el baseline y los dos modelos candidatos (#9, #11, #12), todos deben entrenarse, ajustarse y evaluarse con exactamente las mismas filas. El enunciado exige además que "la partición y las métricas correspondan al tipo de dato y al riesgo del dominio".

El dataset tiene 4,424 estudiantes con clases desbalanceadas (Graduate 49.9 %, Dropout 32.1 %, Enrolled 17.9 %) y no incluye año ni cohorte ([ficha_datos.md](../ficha_datos.md), limitación 3).

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Tipo de partición | Aleatoria estratificada | Conserva las proporciones de clase | No simula el paso del tiempo |
| | Temporal (cohortes antiguas → recientes) | Simula el uso real | **No es posible:** no hay columna de año o cohorte |
| Variable de estratificación | Target de 3 clases | Conserva las proporciones en la formulación binaria y en la multiclase | Ninguna relevante |
| | Target binario | Conserva Dropout / no-Dropout | Enrolled puede quedar desbalanceada en la formulación secundaria |
| Folds de validación cruzada | Guardados en archivo | Mismos folds para todos aunque cambie la versión de scikit-learn | Archivo de 35 KB versionado |
| | Regenerados con la semilla | Más simple | Depende de que scikit-learn no cambie su algoritmo |

## 3. Decisión

- **Partición aleatoria estratificada 80 % train / 20 % test**, estratificada por el `Target` original de 3 clases, con `RANDOM_STATE` = 42.
- **Validación cruzada estratificada de 5 folds definida solo dentro de train**, para elegir hiperparámetros y comparar modelos.
- Los índices de train y test y el fold de cada fila de train se guardan en `data/processed/splits.json`, versionado en Git, junto con el SHA-256 del `data.csv` del que proceden.
- Todo el proyecto obtiene la partición con `load_splits()` y los folds con `cv_folds()` de `src/data.py`. Nadie crea particiones propias.
- **Regla de equipo: el conjunto de test se usa una sola vez, en la evaluación final (#13).** Antes de eso, toda decisión (hiperparámetros, umbral, selección de variables o de modelo) se toma solo con la validación cruzada sobre train.

## 4. Justificación

1. **80/20.** Con 4,424 filas deja 885 estudiantes de test, de los cuales 284 son Dropout: suficientes para estimar el recall de la clase de riesgo con una precisión razonable, y 3,539 filas para entrenar y validar.
2. **Estratificar por 3 clases.** La formulación principal es binaria y la secundaria multiclase ([decisión 002](002_formulacion_target.md)). Estratificar por las 3 clases conserva las proporciones de ambas con una sola partición, de modo que sus resultados son comparables.
3. **Sin partición temporal.** El dataset no trae año ni cohorte, así que no puede simularse "entrenar con el pasado y evaluar con el futuro". Las variables macroeconómicas actúan como indicador indirecto de cohorte, pero no tienen año documentado. Por eso los resultados miden el desempeño dentro de las mismas cohortes (2008/09–2018/19) y no garantizan el desempeño en cohortes futuras: debe declararse como limitación en el informe (#18).
4. **Validación cruzada dentro de train.** Elegir hiperparámetros o modelo mirando el test lo convierte en datos de entrenamiento indirectos y produce resultados optimistas. Con 5 folds cada modelo se valida con 5 subconjuntos de unos 708 estudiantes.
5. **Archivo con hash.** `load_splits()` rechaza la partición si el dataset cambió (otro SHA-256), si train y test se solapan, si no cubren todas las filas o si los folds no corresponden a train.

## 5. Proporciones de clase verificadas

Porcentajes por clase en cada subconjunto, con la partición guardada en `data/processed/splits.json`:

| Subconjunto | N | Graduate | Dropout | Enrolled | N Dropout |
|---|---|---|---|---|---|
| Total | 4,424 | 49.9 % | 32.1 % | 17.9 % | 1,421 |
| Train | 3,539 | 49.9 % | 32.1 % | 17.9 % | 1,137 |
| Test | 885 | 49.9 % | 32.1 % | 18.0 % | 284 |
| Fold 0 (validación) | 708 | 49.9 % | 32.2 % | 17.9 % | 228 |
| Fold 1 (validación) | 708 | 49.9 % | 32.2 % | 17.9 % | 228 |
| Fold 2 (validación) | 708 | 50.0 % | 32.1 % | 17.9 % | 227 |
| Fold 3 (validación) | 708 | 50.0 % | 32.1 % | 17.9 % | 227 |
| Fold 4 (validación) | 707 | 49.9 % | 32.1 % | 18.0 % | 227 |

La diferencia máxima respecto del total es de 0.1 puntos porcentuales.

## 6. Consecuencias

- Los modelos #9, #11 y #12 usan `load_splits()` y `cv_folds()`; ningún notebook ni script crea su propia partición.
- El pipeline (#8) se ajusta solo con las filas de train, o con las de entrenamiento de cada fold.
- La evaluación final (#13) es el único lugar donde se usa `splits["test"]`.
- Las pruebas de partición del #10 deben comprobar el archivo real `data/processed/splits.json`.
- Si UCI publica otra versión del dataset, la partición debe regenerarse y esta decisión reemplazarse.

Uso típico:

```python
from src.data import build_target, cv_folds, load_raw, load_splits
from src.features import select_features

df = load_raw()
splits = load_splits()
X, y = select_features(df), build_target(df)
X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
# cross_validate(modelo, X_train, y_train, cv=cv_folds(splits), ...)
```

## 7. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | POR COMPLETAR: enlace al PR |
