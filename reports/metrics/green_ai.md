Equipo: Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz, 4 núcleos lógicos, 7.9 GB de RAM, GPU: ninguna. Windows-11-10.0.22000-SP0; Python 3.12.14, scikit-learn 1.9.1, codecarbon 3.3.1 (modo process, intensidad de carbono de Dominican Republic).

Entrenamiento (búsqueda o ajuste de hiperparámetros + modelo final), media ± desviación de 3 repeticiones:

| Modelo | Tiempo (s) | Energía (kWh) | Emisiones (gCO2eq) | Inferencia (ms/1000) | Energía inferencia (kWh/1000) | Tamaño (MB) | F1-macro | Recall riesgo |
|---|---|---|---|---|---|---|---|---|
| baseline | 7.5 ± 1.2 | 4.41e-05 ± 1.28e-05 | 0.026 ± 0.007 | 13.3 ± 4.5 | 7.43e-08 | 0.01 | 0.620 ± 0.023 | 0.773 ± 0.014 |
| random_forest | 141.8 ± 18.4 | 9.15e-04 ± 1.81e-04 | 0.532 ± 0.105 | 115.2 ± 16.1 | 6.12e-07 | 18.58 | 0.620 ± 0.021 | 0.808 ± 0.021 |
| gradient_boosting | 92.2 ± 6.7 | 5.46e-04 ± 3.74e-05 | 0.317 ± 0.022 | 17.9 ± 1.9 | 1.04e-07 | 0.28 | 0.618 ± 0.023 | 0.792 ± 0.012 |

Costo relativo al baseline (veces):

| Modelo | Energía entrenamiento | Inferencia | Tamaño |
|---|---|---|---|
| baseline | 1.0 | 1.0 | 1.0 |
| random_forest | 20.8 | 8.7 | 1657.7 |
| gradient_boosting | 12.4 | 1.3 | 24.9 |
