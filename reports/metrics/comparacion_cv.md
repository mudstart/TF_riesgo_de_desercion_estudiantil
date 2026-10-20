Umbral ajustado (decisión 008; el de cada fold se elige sin ese fold):

| Modelo | Umbral | F1-macro | Recall riesgo | Precisión riesgo | Balanced acc. | Entrenamiento (s) | Inferencia (ms/1000) | Energía (kWh) | Emisiones (gCO2eq) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.431 ± 0.005 | 0.620 ± 0.023 | 0.773 ± 0.014 | 0.453 ± 0.021 | 0.665 ± 0.016 | 0.04 ± 0.02 | 12.1 ± 1.1 | 7.40e-05 | 0.043 |
| random_forest | 0.382 ± 0.007 | 0.620 ± 0.021 | 0.808 ± 0.021 | 0.454 ± 0.018 | 0.673 ± 0.014 | 0.47 ± 0.01 | 95.1 ± 3.2 | 1.51e-03 | 0.876 |
| gradient_boosting | 0.385 ± 0.007 | 0.618 ± 0.023 | 0.792 ± 0.012 | 0.452 ± 0.020 | 0.668 ± 0.017 | 0.38 ± 0.03 | 16.0 ± 1.4 | 1.02e-03 | 0.595 |

Umbral 0.5 (referencia):

| Modelo | Umbral | F1-macro | Recall riesgo | Precisión riesgo | Balanced acc. |
|---|---|---|---|---|---|
| baseline | 0.5 | 0.670 ± 0.018 | 0.631 ± 0.031 | 0.527 ± 0.030 | 0.681 ± 0.015 |
| random_forest | 0.5 | 0.676 ± 0.016 | 0.602 ± 0.018 | 0.545 ± 0.029 | 0.681 ± 0.014 |
| gradient_boosting | 0.5 | 0.674 ± 0.006 | 0.653 ± 0.015 | 0.526 ± 0.012 | 0.687 ± 0.004 |

Bootstrap pareado sobre las predicciones out-of-fold al umbral ajustado:

| Comparación | Métrica | Diferencia | IC 95 % |
|---|---|---|---|
| random_forest - baseline | f1_macro | -0.000 | [-0.011, +0.011] |
| random_forest - baseline | recall_riesgo | +0.032 | [+0.014, +0.051] |
| gradient_boosting - baseline | f1_macro | +0.000 | [-0.012, +0.013] |
| gradient_boosting - baseline | recall_riesgo | +0.018 | [-0.001, +0.037] |
| random_forest - gradient_boosting | recall_riesgo | +0.014 | [-0.001, +0.030] |

- Elegibles (F1-macro ≥ 0.62): baseline, random_forest, gradient_boosting.
- Mayor recall de riesgo: random_forest.
- Empatados con él (IC de la diferencia contiene 0): random_forest, gradient_boosting.
- **Modelo elegido: random_forest**, umbral 0.382.
