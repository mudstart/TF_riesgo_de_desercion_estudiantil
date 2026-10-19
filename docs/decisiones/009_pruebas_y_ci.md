# Decisión 009: prueba de desempeño mínimo sin test y pruebas en cada PR

- Issue: [#16](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/16) Pruebas de pipeline y modelos (pruebas 4–8 de 8).
- Rama: `feature/16-pruebas-pipeline-modelos`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-19.
- Estado: Propuesta.
- Reemplaza a: No aplica.

## 1. Contexto

El issue #16 completa las 8 pruebas mínimas. La prueba 8 pide que el modelo final supere al `DummyClassifier` en F1-macro como prueba de regresión, y el issue propone como opcional un flujo de GitHub Actions que ejecute `pytest` en cada PR. Hay que decidir tres cosas:

1. **Con qué datos se mide la prueba 8.** El test ya se usó una vez en #13, y la [decisión 008](008_criterio_seleccion.md) y la [decisión 003](003_particion_validacion.md) prohíben volver a usarlo.
2. **Qué exige la prueba 8.** Solo superar al Dummy, o también que el modelo no empeore respecto a lo que se aprobó en #13.
3. **Si las pruebas se ejecutan automáticamente en cada PR.** Eso afecta el trabajo de los tres integrantes (criterio 5 del registro).

Además, `models/*.joblib` y `data/raw/` no se versionan, así que la prueba 8 no puede depender de un modelo guardado en el equipo de alguien.

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Datos de la prueba 8 | Conjunto de test | Es el desempeño "real" | Usa test más de una vez: invalida la evaluación final (decisiones 003 y 008) |
| | **Out-of-fold en los 5 folds de train** | Mismos folds que #13; reproduce su cifra exacta | No mide generalización nueva, solo regresión |
| Modelo de la prueba 8 | Cargar `models/<modelo>.joblib` | Rápido | No versionado: falla en otro equipo y en CI |
| | **Reconstruirlo con los hiperparámetros y el umbral de `reports/metrics/`** | Versionado y reproducible en cualquier equipo | Unos 3 s de entrenamiento (5 folds) |
| Exigencia | Solo superar al Dummy | Es lo que pide el issue | El modelo podría empeorar mucho sin que la prueba falle |
| | **Superar al Dummy y F1-macro ≥ `MIN_F1_MACRO` − 0.02** | Detecta una regresión real | Hay que fijar un margen |
| Ejecución | Solo en local, a mano | Sin configuración | Depende de que cada integrante lo recuerde |
| | **GitHub Actions en cada PR y push a `main`** | Evidencia en el PR; mismo entorno (`uv.lock`) | Descarga el dataset de UCI en cada ejecución |

## 3. Decisión

- **Prueba 8 sin test.** Se mide el F1-macro out-of-fold del modelo elegido en #13 sobre los 5 folds fijos de train. El modelo y su umbral se leen de `reports/metrics/comparacion_cv.json` (`seleccion.modelo` y `umbral`) y sus hiperparámetros de `reports/metrics/<modelo>.json`.
- **Exigencia doble.** El modelo final debe superar a dos `DummyClassifier` (clase mayoritaria y aleatorio estratificado) **y** mantener un F1-macro de al menos `MIN_F1_MACRO` − 0.02 = 0.60. El margen está en `MARGIN` de `tests/test_model.py`.
- **Pruebas en cada PR.** `.github/workflows/pruebas.yml` ejecuta `uv run pytest` en cada PR y en cada push a `main`. Instala con `uv sync --locked` y descarga el dataset con `uv run python -m src.data`, que verifica su SHA-256.

## 4. Justificación

1. **El test sigue intacto.** La prueba 8 no lo usa, así que la evaluación única de #13 sigue siendo válida. Al reutilizar los folds de #13, la prueba reproduce exactamente la cifra aprobada (0.6201), lo que la convierte en una prueba de regresión: cualquier cambio en los datos, el preprocesamiento o el modelo que la mueva hace fallar la prueba.
2. **Superar al Dummy es una condición muy débil.** Los Dummy quedan en 0.404 (clase mayoritaria) y 0.508 (estratificado). Un modelo que cayera a 0.55 seguiría "superándolos" pese a haber perdido gran parte de lo que se aprobó. El piso de 0.60 lo detecta.
3. **Por qué 0.02 de margen.** Con los mismos folds y la misma semilla la cifra es determinista, pero puede variar levemente con otra versión de scikit-learn o de NumPy. 0.02 es menor que la desviación entre folds del F1-macro en #13 (0.021 para el Random Forest), así que no oculta un empeoramiento real.
4. **CI.** Con el dataset descargado, las pruebas 1–3 y 8 se ejecutan en vez de saltarse. Si UCI no responde, el flujo falla en lugar de dar un verde engañoso.

**Supuestos:** UCI mantiene el archivo con el mismo SHA-256; `comparacion_cv.json` solo cambia si se repite #13.

## 5. Evidencia

- `uv run pytest`: 79 pruebas en verde en unos 20 s (el issue pide menos de 1 minuto). La prueba 8 tarda unos 3 s.
- Prueba 8 sobre los datos reales:

| Modelo | F1-macro out-of-fold (train, 5 folds) |
|---|---|
| Random Forest, umbral 0.382 (modelo final de #13) | 0.620 |
| Dummy, aleatorio estratificado | 0.508 |
| Dummy, clase mayoritaria | 0.404 |
| Piso de la prueba (`MIN_F1_MACRO` − 0.02) | 0.600 |

## 6. Consecuencias

- Cada PR muestra el resultado de `pytest` en la pestaña Checks. Un PR en rojo no debería aprobarse.
- Si una decisión futura cambia el modelo final o el piso de F1-macro, la prueba 8 se adapta sola, porque lee `comparacion_cv.json` y `MIN_F1_MACRO`.
- Para que la prueba 7 dé un error claro, `build_pipeline` antepone el paso `validacion` (`ValidarColumnas` en `src/features.py`), que rechaza un registro incompleto con "Registro incompleto: faltan las columnas [...]". No transforma los datos, así que no cambia ninguna métrica; los `models/*.joblib` guardados antes de este cambio no lo incluyen hasta volver a entrenarlos.
- #14 y #15 pueden apoyarse en la misma CI. Si un cambio baja el F1-macro de 0.60, hay que documentar una decisión nueva en lugar de ajustar el margen.

## 7. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | POR COMPLETAR: enlace al PR |
