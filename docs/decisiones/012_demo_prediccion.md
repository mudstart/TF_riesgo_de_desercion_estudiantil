# Decisión 012: categorías de riesgo, explicación y modelo de la demo

- Issue: [#17](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/17) Artefacto práctico: demo de predicción.
- Rama: `feature/17-demo-prediccion`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-19.
- Estado: Propuesta.
- Reemplaza a: No aplica.

## 1. Contexto

El issue #17 pide una demo (`streamlit run app/app.py`) que, para un estudiante inventado, muestre un nivel de riesgo con probabilidad y categoría **bajo / medio / alto**, "no concluyente" en la zona de abstención del #14, los principales factores y la trazabilidad del modelo. Hay que decidir tres cosas que el issue no fija:

1. **Cómo se pasa de una probabilidad a bajo / medio / alto.** El modelo elegido en #13 tiene un único umbral (0.382, [decisión 008](008_criterio_seleccion.md)) y una zona de abstención alrededor de él ([decisión 009](009_abstencion_y_subgrupos.md)). El test ya se usó, así que un corte nuevo no se puede validar con datos no vistos.
2. **Cómo se muestran los factores.** El modelo elegido es un Random Forest, que no tiene coeficientes. SHAP no es dependencia del proyecto ([decisión 007](007_gradient_boosting.md)).
3. **De dónde sale el modelo.** `models/*.joblib` no se versiona, y el issue exige que la demo se ejecute con un solo comando.

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Categorías | Bajo / medio / alto con un corte nuevo, además de la zona | Tres niveles de riesgo "puros" | El corte nuevo no se puede validar sin usar test otra vez |
| | **La zona de abstención ocupa el lugar de "medio"** | Usa solo el umbral (008) y la zona (009) ya aprobados | "Medio" no es un nivel de riesgo, sino "no se puede concluir" |
| Factores | Solo importancia global | Ya está versionada | No explica el caso concreto |
| | SHAP | Explicación local estándar | Dependencia nueva, descartada en la decisión 007 |
| | **Importancia global + sensibilidad local** | Explica el caso sin dependencias nuevas | La sensibilidad local es una aproximación, no una atribución exacta |
| Modelo | Cargar `models/random_forest.joblib` | Rápido | No versionado: hay que ejecutar `train_rf` antes |
| | **Reconstruirlo con los hiperparámetros versionados** | Un solo comando en cualquier equipo con `data.csv` | ~0.6 s al abrir la app (en caché después) |

## 3. Decisión

1. **Categorías**, con el umbral `u` de `comparacion_cv.json` (0.382) y el ancho `ANCHO_ABSTENCION` (0.05):

| Probabilidad | Categoría | Qué se muestra |
|---|---|---|
| `p < u − 0.05` (< 0.332) | **Bajo** | "Sin riesgo detectado" |
| `u − 0.05 ≤ p < u + 0.05` (0.332–0.432) | **Medio: no concluyente** | "No concluyente – requiere revisión humana" (regla E1) |
| `p ≥ u + 0.05` (≥ 0.432) | **Alto** | "Riesgo de deserción: priorizar acompañamiento" |

   Se usa `clasificar()` de `src/errores.py`, de modo que la demo y el análisis del #14 aplican exactamente la misma regla.

2. **Factores.**
   - **Globales:** las variables con mayor permutation importance del Random Forest (`reports/metrics/random_forest_importancia.csv`, #11).
   - **Del caso:** para cada variable del modelo, cuánto cambia la probabilidad si esa variable toma el valor típico de train (mediana en las numéricas, moda en las nominales y binarias), manteniendo el resto. Se muestran las que más suben y más bajan la probabilidad. La interfaz advierte que **no son causas**, sino la sensibilidad del modelo para ese perfil.

3. **Modelo.** Al abrir, la app entrena el modelo elegido en `comparacion_cv.json` con todo train y los hiperparámetros de `reports/metrics/<modelo>.json`, y lo guarda en caché. Con la misma semilla reproduce el modelo de #13: sus probabilidades difieren del pipeline guardado en menos de 1e-15 (redondeo del orden en que los hilos suman los árboles) y ninguna fila cambia de clase.

4. **Reglas de uso del #14** ([decisión 009](009_abstencion_y_subgrupos.md)):
   - **A1:** la app no predice si `revisar_entrada()` indica abstención (valor fuera del rango de train, código nunca visto o dato faltante) y explica el motivo.
   - **E1:** zona de abstención, como en la tabla anterior.
   - **E2:** con un código infrecuente en train, se da el resultado con aviso de revisión humana.
   - **E3:** si el perfil pertenece a un subgrupo en alerta y el resultado es "bajo", se avisa de que ese "sin riesgo" no debe usarse para excluirlo del acompañamiento.
   - **D1–D3:** se muestran siempre como advertencias de uso.

5. **Trazabilidad visible:**
   - modelo y versión, con la versión como los 12 primeros caracteres del SHA-256 de hiperparámetros, umbral y hash del dataset;
   - fecha de entrenamiento de `<modelo>.json`;
   - SHA-256 del dataset (`DATA_SHA256`, #3);
   - métricas de test con IC 95 % (`test_final.json`, #13).

6. **Perfiles de ejemplo:** 3 perfiles inventados (bajo, alto y no concluyente). Una prueba comprueba que ninguno coincide con una fila del dataset y que cada uno cae en la categoría que dice.

## 4. Justificación

1. **No se inventan umbrales después de usar test.** La zona de abstención ya está justificada en la decisión 009: en [0.332, 0.432] la tasa real de deserción (0.24) es inferior a la global, así que ahí el modelo no informa. Mostrarla como "medio" es honesto: es justo la franja donde el modelo no distingue.
2. **Explicación sin dependencias nuevas.** La importancia global es la que se usó para comparar modelos (#11, #12). La sensibilidad local responde a la pregunta del usuario ("¿qué pesa en este caso?") con el mismo pipeline y sin aproximar el modelo con otro.
3. **Un solo comando y la misma regla en todo el proyecto.** Reconstruir el modelo a partir de archivos versionados evita depender de un `.joblib` local, y reutilizar `clasificar()` y `revisar_entrada()` garantiza que la demo aplica las reglas aprobadas en el #14 y no una copia que pueda divergir.

**Supuestos:** `data/raw/data.csv` está descargado (`uv run python -m src.data`); el valor típico de train es una referencia razonable para la sensibilidad local.

## 5. Consecuencias

- Dependencia nueva `streamlit`.
- `src/predict.py` contiene la lógica sin interfaz (carga, validación, predicción, categorías, factores y trazabilidad) y se prueba en `tests/test_predict.py`. `app/app.py` es solo la interfaz.
- `src/diccionario.py` añade las etiquetas de los códigos UCI para que el formulario muestre texto.
- El README documenta `streamlit run app/app.py`.
- Si una decisión futura cambia el umbral, la zona o el modelo elegido, la demo los toma automáticamente de los archivos versionados.

## 6. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | POR COMPLETAR: enlace al PR |
