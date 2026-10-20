Equipo: AMD Ryzen 5 5600X 6-Core Processor, 12 núcleos lógicos, 31.9 GB de RAM, GPU: ninguna. Windows-11-10.0.26300-SP0; Python 3.12.14, scikit-learn 1.9.1, codecarbon 3.3.1 (modo process, intensidad de carbono de Dominican Republic).

Entrenamiento (búsqueda o ajuste de hiperparámetros + modelo final), media ± desviación de 3 repeticiones:

| Modelo | Tiempo (s) | Energía (kWh) | Emisiones (gCO2eq) | Inferencia (ms/1000) | Energía inferencia (kWh/1000) | Tamaño (MB) | F1-macro | Recall riesgo |
|---|---|---|---|---|---|---|---|---|
| baseline | 3.5 ± 0.3 | 7.40e-05 ± 8.63e-06 | 0.043 ± 0.005 | 7.5 ± 2.5 | 1.58e-07 | 0.01 | 0.620 ± 0.023 | 0.773 ± 0.014 |
| random_forest | 66.6 ± 1.9 | 1.51e-03 ± 5.88e-05 | 0.876 ± 0.034 | 70.3 ± 8.2 | 1.50e-06 | 18.58 | 0.620 ± 0.021 | 0.808 ± 0.021 |
| gradient_boosting | 44.0 ± 2.3 | 1.02e-03 ± 5.75e-05 | 0.595 ± 0.033 | 10.5 ± 2.0 | 2.48e-07 | 0.28 | 0.618 ± 0.023 | 0.792 ± 0.012 |

Costo relativo al baseline (veces):

| Modelo | Energía entrenamiento | Inferencia | Tamaño |
|---|---|---|---|
| baseline | 1.0 | 1.0 | 1.0 |
| random_forest | 20.4 | 9.3 | 1657.7 |
| gradient_boosting | 13.9 | 1.4 | 24.9 |
