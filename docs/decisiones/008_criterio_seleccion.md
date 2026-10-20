# Decisión 008: umbral de decisión y criterio de selección del modelo final

- Issue: [#13](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/13) Evaluación comparativa final de modelos.
- Rama: `feature/13-comparacion-modelos`.
- Responsable: Manicaotex Mueses.
- Fecha: 2026-10-18 (primer borrador: 2026-10-14).
- Estado: Aprobada (PR #37).
- Reemplaza a: No aplica.

> **Este criterio se fija ANTES de evaluar en test.** El commit que añade este documento es anterior a la primera ejecución de `uv run python -m src.comparacion --test`, y el historial de Git es coherente con ello: el commit del criterio es de las 10:07 del 2026-10-18 y `reports/metrics/test_final.json` registra la evaluación en test a las 10:19:43 del mismo día. Cambiarlo después de ver test invalidaría la evaluación final.
>
> El criterio se redactó por primera vez el 2026-10-14 en una rama anterior, con modelos provisionales y numerado 005, número que después ocupó la decisión de ajuste y métricas. Esa rama nunca evaluó en test. Esta versión mantiene el mismo criterio y los mismos parámetros, y solo cambia la numeración y las referencias a los modelos finales de #9, #11 y #12.

## 1. Contexto

El baseline (#9, regresión logística) y los dos candidatos (#11, Random Forest; #12, Gradient Boosting) deben compararse de forma justa y hay que elegir uno con un criterio explícito. Las métricas mínimas exigidas son **F1-macro** y **recall de la clase de riesgo** (Dropout), y el baseline puede ganar.

El coste de los errores no es simétrico para la coordinación académica. Un desertor no detectado (falso negativo) pierde el acompañamiento. Un falso positivo recibe una entrevista o tutoría que no necesitaba, que es un coste bajo porque siempre hay revisión humana ([proyecto.md](../proyecto.md)). Por eso se prioriza el recall, pero sin marcar a casi todos: si todos son "riesgo", la lista no sirve para priorizar.

Al umbral 0.5, los tres modelos quedan a menos de 0.01 de F1-macro entre sí (0.670–0.676) y con recall de Dropout entre 0.60 y 0.65 ([decisión 007](007_gradient_boosting.md)): se pierde uno de cada tres desertores. Al ingreso la señal es moderada, así que bajar el umbral sube el recall muy rápido a costa de la precisión. En una prueba con el umbral que maximiza F2, el recall llegó a 0.92, pero el F1-macro cayó a 0.51 y la precisión bajó a 0.39, cerca del 0.32 de marcar al azar.

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Umbral | 0.5 por defecto | Simple | Con `class_weight="balanced"` deja recall ≈ 0.60–0.65 |
| | Maximizar F-beta (β=2) por modelo | Estándar | Cada modelo elige su propio compromiso, así que comparar recall no es justo. En estos datos marca a casi todos |
| | Recall fijo (p. ej. 0.80) y comparar precisión | Justo | Contradice la prioridad del issue (comparar por recall) |
| | **Mayor recall con F1-macro ≥ piso común** | Justo (mismo piso para todos), prioriza el recall y evita la lista inútil | Hay que fijar el piso |
| Piso de F1-macro | Relativo al baseline (baseline − 0.02) | Se adapta a los datos | Depende de cómo se ajustó el baseline |
| | **Absoluto: 0.62** | Fijo y verificable antes de test | Si ningún modelo lo alcanza, hay que revisarlo |
| ¿Diferencia real o ruido? | Desviación estándar de CV | Ya calculada | Con 5 folds es poco precisa |
| | **Bootstrap pareado sobre las predicciones out-of-fold** | Compara los modelos sobre las mismas filas; intervalo al 95 % | Más cómputo (segundos) |
| Desempate | Mejor F1-macro | Desempeño | Ignora el coste |
| | **El más simple** (baseline < RF < GB) | Interpretable, más barato (Green AI, #15) | Ninguna relevante si la diferencia es ruido |

## 3. Decisión

Todo se calcula **solo con train**, con los 5 folds fijos ([decisión 003](003_particion_validacion.md)) y con los pipelines que guardan #9, #11 y #12 (`models/baseline.joblib`, `models/random_forest.joblib`, `models/gradient_boosting.joblib`), con sus hiperparámetros ya elegidos.

1. **Umbral por modelo.** Sobre las probabilidades out-of-fold, se usa el umbral de **mayor recall de Dropout que mantiene F1-macro ≥ 0.62** (`MIN_F1_MACRO`).
2. **Elegibles.** Los modelos con F1-macro out-of-fold ≥ 0.62 a su umbral.
3. **Ganador provisional.** El elegible con mayor recall de Dropout.
4. **¿Real o ruido?** Bootstrap pareado (2000 remuestreos, IC 95 %) de la diferencia de recall entre el ganador y cada otro elegible. Si el intervalo contiene 0, se consideran **empatados**.
5. **Desempate.** Entre los empatados gana el más simple, en este orden: baseline, Random Forest, Gradient Boosting.
6. **Test, una sola vez.** El modelo elegido se entrena con todo train y se evalúa en test con su umbral de CV (`--test`), con intervalos bootstrap al 95 %. El script se niega a ejecutarse si `reports/metrics/test_final.json` ya existe.

Si ningún modelo alcanza el piso, no se usa el test: se documenta una decisión nueva que reemplace a esta.

## 4. Justificación

1. **Mismo piso, mismo trato.** Con un umbral de 0.5, o uno propio de cada modelo, el modelo que "gana en recall" puede ser simplemente el que se calibró más bajo. Con el mismo piso de F1-macro, todos compiten con la misma restricción de calidad global.
2. **0.62 como piso.** Al umbral 0.5 el F1-macro está en ≈ 0.67. Un piso de 0.62 deja ceder unos 5 puntos de F1-macro para ganar recall (hasta ≈ 0.78–0.80), con una precisión de ≈ 0.45: menos de 1 de cada 2 marcados es desertor, pero se detectan 4 de cada 5. Por debajo de ese piso, la precisión se acerca a la de marcar al azar.
3. **Umbral sin fuga.** El umbral se elige con las predicciones out-of-fold de train, nunca con test. La tabla de media ± desviación usa, para cada fold, un umbral elegido con los demás folds, de modo que no se infla.
4. **Bootstrap pareado.** Los tres modelos predicen las mismas 3,539 filas de train. Remuestrear filas y recalcular la diferencia en cada remuestreo aísla la variación debida al modelo.
5. **El baseline puede ganar.** Si un candidato complejo no mejora el recall más allá del ruido, se elige el baseline: es interpretable (coeficientes, #9) y gasta menos energía (#15).

**Supuestos:** el coste de un falso positivo es bajo porque hay revisión humana; las cohortes futuras se parecen a las del dataset (limitación de la decisión 003).

## 5. Consecuencias

- `src/evaluation.py` implementa el criterio: `threshold_curve`, `choose_threshold`, `cv_threshold_metrics`, `paired_bootstrap`, `bootstrap_ci`, `select_model` y `check_test_unused`. `src/comparacion.py` lo ejecuta.
- Los parámetros están en `src/config.py`: `MIN_F1_MACRO`, `N_BOOTSTRAP`, `CONFIDENCE` y `MODELOS_COMPARADOS` (orden de simplicidad).
- #15 escribe `energia_kwh` y `emisiones_gco2eq` en `reports/metrics/<nombre>.json`; mientras falten, la tabla los marca como pendientes.
- #14 (análisis de errores y subgrupos) usa el modelo y el umbral elegidos aquí.
- Salidas: `reports/metrics/comparacion_cv.json` y `.md`, `reports/figures/13_*.png`, `reports/metrics/test_final.json` y `reports/comparacion_modelos.md`.

## 6. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | Jose M. Maduro |
| Fecha | 2026-10-18 |
| Evidencia | [PR #37](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/37): aprobado tras volver a ejecutar la comparación con CV y la evaluación en test, con los mismos resultados |