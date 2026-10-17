# Decisión 005: ajuste de hiperparámetros y formato estándar de métricas

- Issue: [#9](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/9) Baseline de regresión logística.
- Rama: `feature/9-baseline-logistica`.
- Responsable: Manicaotex Mueses.
- Fecha: 2026-10-16.
- Estado: Propuesta.
- Reemplaza a: No aplica.

## 1. Contexto

El baseline (#9) es el primero de los tres modelos que se ajustan y se comparan. Random Forest (#11) y Gradient Boosting (#12) usarán las mismas funciones de `src/training.py`, así que lo que se decida aquí lo heredan: cómo se elige el hiperparámetro y en qué formato se guardan las métricas que lee la comparación (#13) y completa Green AI (#15).

Hay que decidir dos cosas (criterios 2 y 5 de [README.md](README.md)):

1. **Criterio de ajuste:** con qué métrica y con qué regla se elige el valor del hiperparámetro.
2. **Formato de métricas:** qué claves tiene `reports/metrics/<modelo>.json`, para que los tres modelos se comparen sin adaptar código.

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Métrica de ajuste | Recall de Dropout | Es la métrica prioritaria | Al umbral 0.5 premia marcar a casi todos; el umbral se fija después (#13) |
| | ROC-AUC | Independiente del umbral | No mide el equilibrio entre clases que se reporta |
| | **F1-macro al umbral 0.5** | Equilibra las dos clases; es la métrica principal del proyecto | Depende del umbral, que todavía no está fijado |
| Regla de elección | Máximo F1-macro medio | Simple | Elige diferencias menores que el ruido entre folds; con C grande, coeficientes apoyados en grupos de 22–31 estudiantes |
| | Regla 1-SE (corte = mejor − std/√k) | Regla clásica (Breiman et al., 1984; Hastie et al., 2009) | Supone folds independientes, y no lo son (comparten ~75 % del train): el SE subestima el ruido. En el baseline elige C = 1, casi sin regularización |
| | **Regla de una desviación estándar (corte = mejor − std)** | Más conservadora: solo prefiere un modelo más complejo si mejora más que la variación entre folds | No es la regla de los libros; puede regularizar de más |
| Formato | Cada modelo guarda lo suyo | Flexible | #13 y #15 tendrían que adaptarse a cada modelo |
| | **`model_report` + `save_metrics` comunes** | Mismas claves; una prueba protege el contrato | Cambiar el formato obliga a acordarlo entre los tres |

## 3. Decisión

**Criterio de ajuste** (`tune` en `src/training.py`):

- Un hiperparámetro principal por modelo, con una rejilla corta **ordenada del valor más simple al más complejo** (en el baseline, `C` de 0.001 a 100, 11 valores en escala logarítmica).
- Para cada valor, validación cruzada con los 5 folds fijos de la [decisión 003](003_particion_validacion.md), solo sobre train, y F1-macro medio entre folds al umbral 0.5.
- **Regla de una desviación estándar:** gana el primer valor de la rejilla cuyo F1-macro medio es mayor o igual que el del mejor menos la desviación estándar (ddof = 1) del mejor entre folds. La columna `dentro_1de` de la tabla de resultados marca los valores que pasan el corte.
- El umbral de decisión no se ajusta aquí: lo fija #13 para los tres modelos con el mismo criterio.

**Formato estándar** (`model_report` y `save_metrics`): `reports/metrics/<modelo>.json` (`baseline`, `random_forest`, `gradient_boosting`) con estas claves:

| Clave | Contenido |
|---|---|
| `modelo`, `estimador` | Nombre del modelo y clase de sklearn |
| `hiperparametros` | Parámetros escalares del estimador (sin los marcados como `deprecated`) |
| `fecha`, `n_train`, `umbral` | Cuándo se generó, filas de train y umbral de las métricas (0.5) |
| `cv` | Media y desviación entre folds de F1-macro, recall y precisión de Dropout, balanced accuracy y tiempos |
| `oof` | Las mismas métricas con las predicciones out-of-fold de todo train |
| `matriz_confusion` | `tn`, `fp`, `fn`, `tp` out-of-fold (1 = Dropout) |
| `tiempos` | Entrenamiento por fold (s) e inferencia por 1,000 registros (ms) |
| `energia_kwh`, `emisiones_gco2eq` | Los escribe #15; `save_metrics` los conserva al regenerar el archivo |

Cada modelo puede añadir claves propias (el baseline añade `ajuste`, `referencia_dummy` y `coeficientes`), pero no quitar ni renombrar las comunes.

## 4. Justificación

1. **F1-macro al umbral 0.5 para ajustar.** Ajustar por recall al umbral 0.5 confundiría dos cosas: la capacidad del modelo y dónde se pone el corte. Con F1-macro se elige un modelo equilibrado, y el umbral que sube el recall se decide aparte y por igual para todos (#13).
2. **Desviación y no error estándar.** Las 5 estimaciones de F1-macro no son independientes: cada par de folds de entrenamiento comparte el 75 % de las filas. Por eso std/√5 subestima la incertidumbre real de la media (Bengio y Grandvalet, 2004), y la regla 1-SE pasa a distinguir diferencias que siguen siendo ruido. Restar la desviación completa es una cota conservadora de ese ruido.
3. **Se prefiere lo simple porque el proyecto lo necesita.** El modelo debe ser explicable a la universidad y revisarse por equidad (#14). En el baseline, C = 1 o 3.16 deja coeficientes grandes en categorías con muy pocos casos (por ejemplo, una ocupación de la madre con 22 estudiantes y ningún desertor como factor más protector, OR 0.11). Con C = 0.01 el modelo se apoya en variables con mucha evidencia y todos los coeficientes que cambian de signo entre folds son despreciables (|coef| < 0.04).
4. **El coste en rendimiento es menor que el ruido.** El F1-macro de C = 0.01 (0.670) queda a 0.004 del máximo (0.673 ± 0.005).
5. **Un formato con contrato.** `tests/test_training.py::test_report_has_the_standard_format` falla si cambian las claves comunes, así que un cambio en el formato no pasa sin que lo vean los tres modelos.

Supuesto explícito: elegir el hiperparámetro con los mismos folds con que se reportan las métricas da un sesgo optimista pequeño. Se acepta porque se ajusta un único parámetro con una rejilla corta, y el test sigue sin tocarse hasta #13.

## 5. Evidencia

Ajuste de `C` del baseline (validación cruzada, 5 folds, solo train, umbral 0.5). Detalle en [reports/baseline.md](../../reports/baseline.md) y `reports/metrics/baseline.json` (`ajuste.resultados`).

| C | 0.001 | 0.003 | 0.01 | 0.032 | 0.1 | 0.32 | 1 | 3.16 | 10 | 32 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1-macro medio | 0.657 | 0.664 | 0.670 | 0.670 | 0.670 | 0.671 | 0.673 | 0.673 | 0.672 | 0.672 | 0.673 |

| Regla | Corte | C elegido |
|---|---|---|
| Máximo | — | 3.16 |
| 1-SE (mejor − 0.0051/√5) | 0.6711 | 1 |
| **Una desviación (mejor − 0.0051)** | **0.6683** | **0.01** |

## 6. Consecuencias

- #11 y #12 llaman a `tune(estimador, parametro, rejilla, X_train, y_train, cv_folds(splits))` con la rejilla ordenada de lo más simple a lo más complejo (por ejemplo, de menos a más profundidad u hojas), y guardan con `save_metrics(model_report(...))`.
- #13 lee las claves comunes de los tres `reports/metrics/*.json` y fija el umbral de cada modelo; no vuelve a ajustar hiperparámetros.
- #15 escribe `energia_kwh` y `emisiones_gco2eq` en esos archivos; reentrenar no los borra.
- Cambiar la regla o el formato requiere una decisión nueva que reemplace a esta.

Referencias:

- Breiman, L., Friedman, J., Olshen, R. y Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
- Hastie, T., Tibshirani, R. y Friedman, J. (2009). *The Elements of Statistical Learning* (2.ª ed.), sección 7.10. Springer.
- Bengio, Y. y Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *Journal of Machine Learning Research*, 5, 1089–1105.

## 7. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | [PR #33](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/33) |
