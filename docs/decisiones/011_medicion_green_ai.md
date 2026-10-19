# Decisión 011: cómo se mide el costo computacional y energético (Green AI)

- Issue: [#15](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/15) Medición Green AI (costo computacional y energético).
- Rama: `feature/15-green-ai`.
- Responsable: Manicaotex Mueses.
- Fecha: 2026-10-19.
- Estado: Propuesta.
- Reemplaza a: No aplica.

## 1. Contexto

El eje del tema es "Modelos tabulares y Green AI". El issue #15 pide medir para los tres modelos el tiempo de entrenamiento (con la búsqueda de hiperparámetros), el tiempo de inferencia por 1,000 registros, la energía (kWh), las emisiones (gCO2eq) y el tamaño del modelo serializado, con mediciones comparables (mismo equipo y condiciones). La comparación (#13) ya reserva las columnas `energia_kwh` y `emisiones_gco2eq` en `reports/metrics/<modelo>.json` ([decisiones 005](005_ajuste_y_metricas.md) y [008](008_criterio_seleccion.md)).

El equipo de medición es un portátil sin GPU y sin medidor de potencia por hardware accesible desde Python en Windows, así que la energía es una estimación.

## 2. Opciones evaluadas

| Opción | Ventaja | Desventaja |
|---|---|---|
| Solo tiempos (`time.perf_counter`) | Simple, sin dependencias | No da energía ni emisiones |
| `codecarbon` en línea (`EmissionsTracker`) | Detecta el país por IP | Necesita internet; resultado depende de la red usada |
| **`codecarbon` sin conexión (`OfflineEmissionsTracker`, país fijo `DOM`)** | Reproducible, sin red, intensidad de carbono de República Dominicana (580.78 gCO2/kWh, datos de codecarbon) | La potencia de CPU se estima (TDP y carga), no se mide |
| Medidor físico (vatímetro) | Medición real | No reproducible por otros integrantes |
| Medir cada modelo en ejecuciones separadas | Simple | Condiciones distintas (carga, temperatura) entre modelos |

## 3. Decisión

- `OfflineEmissionsTracker(country_iso_code="DOM", tracking_mode="process", measure_power_secs=1)`, sin archivo `emissions.csv`.
- **Entrenamiento** = lo que hace cada script de #9, #11 y #12 para obtener el modelo: ajuste de `C` (baseline) o búsqueda aleatoria de 25 combinaciones (candidatos) con los 5 folds fijos, más el entrenamiento final con todo train. No incluye la permutation importance ni los coeficientes, que son análisis y no entrenamiento.
- **3 repeticiones**, alternando los tres modelos en cada repetición dentro de un mismo proceso; se reporta media ± desviación.
- **Inferencia**: 200 lotes de 1,000 filas de train con `predict_proba` del pipeline completo (preprocesamiento + modelo), tras un lote de calentamiento.
- **Tamaño**: bytes del pipeline serializado con `joblib`, igual que `save_pipeline`.
- `energia_kwh` y `emisiones_gco2eq` de `reports/metrics/<modelo>.json` son la **media del entrenamiento**.

## 4. Justificación

- Sin conexión y con país fijo, cualquier integrante obtiene emisiones calculadas con la misma intensidad de carbono, y la ejecución no depende de internet.
- `tracking_mode="process"` atribuye al proceso de Python su parte de la carga de CPU, lo que reduce el efecto de otros programas abiertos.
- Alternar los modelos y repetir hace que los tres compartan las mismas condiciones del equipo; la desviación muestra cuánto varía la medición.
- Medir en un mismo script el entrenamiento completo evita comparar tiempos tomados en días y cargas distintos (ver la nota de `reports/comparacion_modelos.md` sobre tiempos medidos con el equipo bajo más carga).
- Supuesto: la estimación de potencia de codecarbon es igual de imprecisa para los tres modelos, así que las **proporciones** entre modelos son más fiables que los valores absolutos.

## 5. Consecuencias

- Nueva dependencia `codecarbon` en `pyproject.toml`.
- Nuevo script `src/green_ai.py` y pruebas en `tests/test_green_ai.py`.
- Salidas: `reports/metrics/green_ai.json` y `green_ai.md`, `reports/figures/15_desempeno_vs_costo.png` y `reports/green_ai.md`.
- Al volver a ejecutar `uv run python -m src.comparacion` (solo CV), la tabla de #13 muestra energía y emisiones en lugar de "pendiente (#15)". No cambia el modelo elegido ni toca test.
- No se reentrena ni se sobrescribe nada en `models/`.

## 6. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | |
| Fecha | |
| Evidencia | Enlace al PR |
