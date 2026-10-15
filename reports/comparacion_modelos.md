# Comparación de modelos y selección final (#13)

> **Estado: pendiente.** Se completa cuando estén los pipelines finales de #9, #11 y #12 y las mediciones de #15. Las tablas salen de `reports/metrics/comparacion_cv.md`, que genera `uv run python -m src.comparacion`.

## 1. Criterio de selección (fijado antes de ver test)

Ver [decisión 005](../docs/decisiones/005_criterio_seleccion.md). En resumen:

1. Cada modelo usa el umbral de **mayor recall de Dropout con F1-macro ≥ 0.62**, elegido con las predicciones out-of-fold de train.
2. Gana el elegible con mayor recall de Dropout.
3. Si el intervalo bootstrap al 95 % de la diferencia de recall contiene 0, es un empate y gana el más simple (baseline < Random Forest < Gradient Boosting).
4. El modelo elegido se evalúa **una sola vez** en test.

## 2. Tabla comparativa (validación cruzada, 5 folds, media ± desviación)

<!-- Pegar aquí las tablas de reports/metrics/comparacion_cv.md -->

## 3. ¿Diferencias reales o ruido?

<!-- Intervalos del bootstrap pareado: interpretar cuáles contienen 0. -->

## 4. Ajuste del umbral

![Curvas PR y ROC](figures/13_pr_roc_cv.png)

<!-- Comentar el compromiso recall/precisión en el umbral elegido frente a 0.5. -->

## 5. Matrices de confusión (out-of-fold, umbral ajustado)

![Matrices de confusión](figures/13_confusion_cv.png)

## 6. Evaluación final en test (una sola vez)

<!-- Copiar reports/metrics/test_final.json: métricas con IC 95 % y matriz figures/13_confusion_test.png. -->

## 7. Conclusión

<!-- Qué modelo se elige y por qué, en 3-5 líneas, con referencia al coste Green AI (#15). -->
