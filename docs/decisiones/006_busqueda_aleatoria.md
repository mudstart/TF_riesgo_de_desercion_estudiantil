# Decisión 006: búsqueda aleatoria de hiperparámetros para los candidatos

- Issue: [#11](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/11) Modelo candidato 1: Random Forest.
- Rama: `feature/11-random-forest`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-17.
- Estado: Aprobada (PR #34).
- Reemplaza a: [005](005_ajuste_y_metricas.md), **solo el criterio de ajuste para #11 y #12**. El formato estándar de métricas de la 005 sigue vigente, y el baseline (#9) conserva su ajuste.

## 1. Contexto

La [decisión 005](005_ajuste_y_metricas.md) fijó para los tres modelos un hiperparámetro principal, una rejilla ordenada de lo más simple a lo más complejo y la regla de una desviación estándar. El issue #11 pide en cambio `RandomizedSearchCV` sobre cuatro hiperparámetros del Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`), limitado a 20–30 iteraciones por criterio Green AI. Ambas cosas no son compatibles, y la 005 exige una decisión nueva para cambiar la regla.

## 2. Opciones evaluadas

| Opción | Ventaja | Desventaja |
|---|---|---|
| Seguir la 005: un parámetro con `tune()` y regla de una desviación | Mismo criterio que el baseline | Fija a mano tres de los cuatro hiperparámetros, que en un Random Forest interactúan |
| Rejilla completa de los 4 parámetros | Explora todo | 480 combinaciones × 5 folds = 2,400 entrenamientos: contrario al criterio Green AI |
| **Búsqueda aleatoria de 25 combinaciones, gana el mayor F1-macro** | Explora los 4 parámetros con coste acotado (125 entrenamientos); es lo que pide el issue | La regla de una desviación no tiene un orden claro de "más simple" con 4 parámetros; criterio distinto del baseline |

## 3. Decisión

Para los candidatos (#11 Random Forest y #12 Gradient Boosting):

- `RandomizedSearchCV` con **25 combinaciones** de un espacio discreto definido en el script de cada modelo, `scoring="f1_macro"`, los **5 folds fijos** de `cv_folds(splits)`, solo train y `random_state=42`.
- Gana la combinación con **mayor F1-macro medio entre folds** (umbral 0.5).
- La ganadora se vuelve a validar con `cross_validate_model` y se guarda con `model_report` + `save_metrics`, en el formato estándar de la 005. El JSON añade la clave `busqueda` con el espacio, las 25 combinaciones y el tiempo total.
- El umbral de decisión sigue fijándose en #13 para los tres modelos con el mismo criterio.

Espacio del Random Forest (`src/models/train_rf.py`, 480 combinaciones):

| Parámetro | Valores |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | 5, 10, 15, 20, sin límite |
| `min_samples_leaf` | 1, 2, 5, 10, 20, 50 |
| `max_features` | "sqrt", "log2", 0.3, 0.5 |

Fijos: `class_weight="balanced"` (decisión 004) y `random_state=42`.

## 4. Justificación

1. **Los hiperparámetros del Random Forest interactúan.** La profundidad, el tamaño mínimo de hoja y el número de variables por división regulan juntos la complejidad del árbol; fijar tres a mano y ajustar uno sesga el resultado hacia esos valores fijos.
2. **Coste acotado (Green AI).** 25 de 480 combinaciones (5 %) cuestan 125 entrenamientos y unos 61 s en la máquina de desarrollo. La búsqueda aleatoria cubre mejor un espacio de varios parámetros que una rejilla del mismo tamaño (Bergstra y Bengio, 2012).
3. **Mayor F1-macro en lugar de la regla de una desviación.** Con cuatro parámetros no hay un orden total de "más simple a más complejo" que permita aplicar la regla de forma inequívoca.

Limitaciones aceptadas:

- **Criterio distinto del baseline.** Con la regla de "el mayor F1-macro", el baseline habría elegido `C` = 3.16 con F1-macro 0.673 en lugar de 0.670 (005, sección 5). La diferencia (0.003) es menor que la desviación entre folds de ese mejor valor (0.005), así que no cambia la comparación de forma relevante, pero #13 debe mencionarla.
- **Sesgo optimista de selección.** Elegir la mejor de 25 combinaciones con los mismos folds con que se reporta favorece levemente al Random Forest. Las 8 mejores combinaciones están entre 0.668 y 0.676, dentro de una desviación de la ganadora (0.015): su ventaja sobre ellas no se distingue del ruido. El test sigue sin usarse hasta #13.

## 5. Evidencia: Random Forest

Hiperparámetros finales: **`n_estimators` = 300, `max_depth` = sin límite, `min_samples_leaf` = 2, `max_features` = "log2"**.

Las 25 combinaciones obtuvieron F1-macro medio entre 0.650 y 0.676. Detalle en `reports/metrics/random_forest.json` (`busqueda.resultados`).

Validación cruzada de la ganadora (5 folds, solo train, umbral 0.5), frente al baseline:

| Métrica | Baseline (#9) | Random Forest |
|---|---|---|
| F1-macro | 0.670 ± 0.018 | 0.676 ± 0.016 |
| Recall Dropout | 0.631 ± 0.031 | 0.602 ± 0.018 |
| Precisión Dropout | 0.527 ± 0.030 | 0.545 ± 0.029 |
| Balanced accuracy | 0.681 ± 0.015 | 0.681 ± 0.014 |
| Entrenamiento por fold | 0.07 s | 0.48 s |
| Inferencia por 1,000 registros | 24 ms | 92 ms |
| Búsqueda de hiperparámetros | — | 61 s |

Al umbral 0.5 el Random Forest gana 0.006 de F1-macro, muy por debajo de la desviación entre folds de ambos modelos (≈ 0.017), con menor recall y mayor precisión, a unas 7 veces el coste de entrenamiento. La comparación definitiva, con el umbral ajustado, corresponde a #13.

Permutation importance (caída media de F1-macro al permutar la columna en el fold de validación, 5 folds): las más importantes son `Age at enrollment` (0.025, positiva en los 5 folds), `Application mode` (0.017), `Course` (0.014), `Gender` (0.010) y `Mother's occupation` (0.006). `Admission grade`, `Application order`, `Father's occupation`, `Displaced`, `Nacionality` y `Educational special needs` tienen importancia media ≤ 0. Detalle en `reports/metrics/random_forest_importancia.csv`.

## 6. Consecuencias

- #12 usa el mismo procedimiento con su propio espacio de búsqueda y 25 combinaciones.
- #13 compara los tres modelos con las claves comunes del formato estándar y debe mencionar que el baseline se ajustó con la regla de una desviación.
- `Gender` aparece entre las variables más importantes: #14 debe revisar errores por género y comparar un modelo sin esa variable.
- El tiempo total de búsqueda (`tiempos.busqueda_total_s`) queda disponible para la medición Green AI (#15).

Referencia: Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.

## 7. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | POR COMPLETAR: enlace al PR |
