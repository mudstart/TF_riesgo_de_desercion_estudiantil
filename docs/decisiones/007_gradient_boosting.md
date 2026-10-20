# Decisión 007: Gradient Boosting con early stopping

- Issue: [#12](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/12) Modelo candidato 2: Gradient Boosting.
- Rama: `feature/12-gradient-boosting`.
- Responsable: Jose M. Maduro.
- Fecha: 2026-10-17.
- Estado: Aprobada (PR #35).
- Reemplaza a: No aplica.

## 1. Contexto

El issue #12 pide un segundo candidato de Gradient Boosting, elegir la librería, manejar el desbalance, una búsqueda acotada de `learning_rate`, `max_depth`, `n_estimators` y `subsample`, y early stopping con una porción de validación separada dentro de train, nunca con test. El procedimiento de búsqueda ya está fijado por la [decisión 006](006_busqueda_aleatoria.md) (25 combinaciones, mayor F1-macro, folds fijos) y el desbalance por la [decisión 004](004_preprocesamiento.md) (`class_weight="balanced"`).

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Librería | XGBoost / LightGBM | Muy usadas; `subsample` de filas | Nueva dependencia; `scale_pos_weight` distinto de `class_weight` de los otros dos modelos |
| | **`HistGradientBoostingClassifier`** (scikit-learn) | Sin dependencias; admite `class_weight`; early stopping nativo | Sin `subsample` de filas |
| Número de árboles | Buscarlo como hiperparámetro | Explícito | Gasta combinaciones de la búsqueda en algo que el early stopping resuelve |
| | **Early stopping** | Ajusta los árboles a cada combinación con coste mínimo | Usa un 10 % del train de cada ajuste para validar |

## 3. Decisión

- **Modelo:** `HistGradientBoostingClassifier(class_weight="balanced", random_state=42)` dentro de `build_pipeline` (#8).
- **Early stopping:** hasta 500 árboles; se detiene si la pérdida en un **10 % estratificado del train que recibe cada ajuste** no mejora durante 10 iteraciones. En validación cruzada, ese 10 % sale del fold de entrenamiento: nunca del fold de validación ni del test.
- **Búsqueda (decisión 006):** 25 combinaciones de un espacio de 768:

| Parámetro | Valores |
|---|---|
| `learning_rate` | 0.02, 0.05, 0.1, 0.2 |
| `max_depth` | 3, 5, 8, sin límite |
| `min_samples_leaf` | 10, 20, 50, 100 |
| `l2_regularization` | 0, 0.1, 1, 10 |
| `max_features` | 0.5, 0.8, 1.0 |

- **Sustituciones respecto al issue:** `n_estimators` lo determina el early stopping; `subsample` (no disponible) se sustituye por `max_features`, que submuestrea columnas en cada división y también reduce el sobreajuste. Se añaden `min_samples_leaf` y `l2_regularization`, los principales reguladores de este modelo.
- **Interpretabilidad:** permutation importance, igual que el Random Forest, para que sean comparables. No se usa SHAP, que añadiría una dependencia; queda como posible mejora.

## 4. Justificación

1. **Mismo manejo del desbalance en los tres modelos.** `class_weight="balanced"` en el baseline, el Random Forest y este modelo hace que las diferencias se deban al algoritmo y no a cómo se pondera cada clase.
2. **Sin dependencias nuevas.** El proyecto usa solo scikit-learn para modelar; con 3,539 filas de train, una implementación más rápida no aporta una ventaja práctica.
3. **Early stopping sin fuga.** La validación interna se separa del conjunto que recibe `fit`, que en la búsqueda y en la validación cruzada es siempre el fold de entrenamiento, y con todo train en el modelo final. El fold de validación sigue sirviendo solo para medir, y el test no se toca hasta #13.

## 5. Evidencia

Hiperparámetros finales: **`learning_rate` = 0.05, `max_depth` = 8, `min_samples_leaf` = 20, `l2_regularization` = 1, `max_features` = 0.5**. El early stopping dejó entre 67 y 95 árboles por fold y **83 árboles** en el modelo final.

Las 25 combinaciones obtuvieron F1-macro medio entre 0.649 y 0.674; 12 de ellas quedan dentro de una desviación de la ganadora (0.006), así que su ventaja sobre ellas no se distingue del ruido. Detalle en `reports/metrics/gradient_boosting.json` (`busqueda.resultados`).

Validación cruzada (5 folds, solo train, umbral 0.5):

| Métrica | Baseline (#9) | Random Forest (#11) | Gradient Boosting |
|---|---|---|---|
| F1-macro | 0.670 ± 0.018 | 0.676 ± 0.016 | 0.674 ± 0.006 |
| Recall Dropout | 0.631 ± 0.031 | 0.602 ± 0.018 | 0.653 ± 0.015 |
| Precisión Dropout | 0.527 ± 0.030 | 0.545 ± 0.029 | 0.526 ± 0.012 |
| Balanced accuracy | 0.681 ± 0.015 | 0.681 ± 0.014 | 0.687 ± 0.004 |
| Entrenamiento por fold | 0.07 s | 0.45 s | 0.32 s |
| Inferencia por 1,000 registros | 24 ms | 90 ms | 14 ms |
| Búsqueda de hiperparámetros | — | 61 s | 41 s |

Los tres modelos quedan a menos de 0.01 de F1-macro, por debajo de la variación entre folds. Al umbral 0.5, el Gradient Boosting tiene el mayor recall de Dropout y la menor variación entre folds, con una inferencia más rápida que el baseline. La elección final, con el umbral ajustado, corresponde a #13.

Permutation importance (5 folds): `Age at enrollment` (0.065, positiva en los 5 folds), `Course` (0.028), `Gender` (0.019), `Application mode` (0.017) y `Father's occupation` (0.011). Coinciden con el Random Forest en las cuatro primeras. Detalle en `reports/metrics/gradient_boosting_importancia.csv`.

## 6. Consecuencias

- #13 compara los tres modelos con las claves comunes del formato estándar.
- `Gender` vuelve a estar entre las tres variables más importantes: refuerza que #14 revise errores por género y compare un modelo sin esa variable.
- Los tiempos (`tiempos`) y el de búsqueda quedan disponibles para Green AI (#15).

## 7. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | Manicaotex Mueses |
| Fecha | 2026-10-18 |
| Evidencia | [PR #35](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/35): `uv run pytest` en verde (61 pruebas) y `train_gb` reproduce las métricas (solo cambian los tiempos) |
