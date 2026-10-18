# Decisión 009: análisis de subgrupos, zona de abstención y condiciones de detención

- Issue: [#14](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/14) Análisis de errores y de subgrupos.
- Rama: `feature/14-errores-subgrupos`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-18.
- Estado: Propuesta.
- Reemplaza a: No aplica.

## 1. Contexto

#13 eligió el Random Forest con umbral 0.382 ([decisión 008](008_criterio_seleccion.md)). Falta saber dónde y con quién se equivoca, y fijar por escrito cuándo el resultado no debe usarse. La demo (#17) aplica estas reglas y la sección de uso responsable del informe (#18) las cita, así que se deciden aquí y no en cada issue.

Hay que decidir cuatro cosas: con qué datos se analiza, cuándo una brecha entre subgrupos es relevante, qué casos son "no concluyentes" y qué condiciones detienen, anulan o escalan el resultado.

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Datos del análisis | Predicciones en test | Datos no vistos | El test ya se usó una vez (decisión 003); elegir la zona de abstención con él lo contaminaría. Con 885 filas, los subgrupos quedan muy pequeños |
| | **Probabilidades out-of-fold de train** | 3,539 filas, cada una predicha por un modelo que no la vio; mismas cifras que #13 | Hiperparámetros y umbral se eligieron con esos folds (optimismo leve, ya presente en #13) |
| Brecha relevante | Cualquier diferencia significativa | Simple | Con 3,539 filas, diferencias pequeñas e irrelevantes salen significativas |
| | Ratio de recall (regla de los 4/5) | Conocida | Pensada para tasas de selección, no para recall |
| | **Recall del subgrupo más de 0.10 por debajo del resto, con IC 95 % que excluye 0** | Exige tamaño y evidencia a la vez; 0.10 ≈ 10 desertores de cada 100 sin detectar | El 0.10 es una convención del equipo |
| Tamaño mínimo | Ninguno | Reporta todo | Intervalos de ±0.30 en grupos de 13 desertores |
| | **30 desertores en el subgrupo** | El IC del recall queda en torno a ±0.15 | Grupos pequeños (internacionales, necesidades especiales) quedan sin evaluar |
| Zona de abstención | 0.40–0.60 (la del issue) | Centrada en 0.5 | El umbral es 0.382: la zona quedaría casi toda del lado "riesgo" |
| | **Simétrica alrededor del umbral, ±0.05: [0.332, 0.432]** | Recoge los casos que un cambio pequeño de probabilidad cambiaría de clase | 21 % de los casos va a revisión humana |
| | ±0.10 | Más errores evitados | 40 % de los casos a revisión: la lista deja de priorizar |

## 3. Decisión

1. **Datos.** Todo el análisis usa las probabilidades out-of-fold del modelo elegido en train, con su umbral de #13. El test no se vuelve a usar.
2. **Subgrupos.** Género, rango de edad (hasta 20, 21–25, más de 25), estado civil (soltero/a u otro), becario, desplazado, internacional, necesidades educativas especiales y curso (`SUBGRUPOS`, `LIMITES_EDAD`). Becario no entra al modelo e internacional es redundante, pero ambos describen grupos.
3. **Métricas por subgrupo.** Recall de Dropout, tasa de falsos positivos, precisión y tasa de marcados, con IC bootstrap al 95 %, y la brecha de recall y de falsos positivos frente al resto de la población.
4. **Alerta de subgrupo.** Un subgrupo con al menos 30 desertores (`MIN_RIESGO_SUBGRUPO`) queda en alerta si su recall está más de 0.10 por debajo del resto (`MAX_BRECHA_RECALL`) y el IC de la brecha excluye 0. Con menos de 30 desertores, se reporta como "sin evidencia suficiente".
5. **Zona de abstención.** Si la probabilidad está a menos de 0.05 del umbral (`ANCHO_ABSTENCION`), es decir, entre 0.332 y 0.432, el resultado es **"no concluyente: requiere revisión humana"**.
6. **Condiciones de detención, abstención y escalamiento**, en este orden de prioridad:

| N.º | Condición | Acción | Cómo se comprueba |
|---|---|---|---|
| D1 | Uso en otra institución, país o plan de estudios, o en cohortes sin revalidar | **Detener**: no usar el modelo | Responsabilidad de quien lo aplica; el modelo solo vale para el contexto del dataset |
| D2 | Uso para sancionar, excluir, admitir, retirar becas o decidir sin revisión humana | **Detener** | Usos excluidos en [proyecto.md](../proyecto.md), sección 5 |
| D3 | Al aplicarlo a una cohorte nueva con desenlace conocido, el recall global cae por debajo de 0.75 (límite inferior del IC en test) | **Detener** y reentrenar o revisar | Repetir `uv run python -m src.errores` con los datos nuevos |
| A1 | Falta un valor, una numérica está fuera del rango de train o un código no se vio nunca en train | **Abstenerse**: no dar resultado | `revisar_entrada(...)["abstenerse"]` |
| E1 | La probabilidad cae en la zona [0.332, 0.432] | **Escalar**: "no concluyente, requiere revisión humana" | `clasificar(...) == "no concluyente"` |
| E2 | Un código nominal es infrecuente en train (< 10 casos; el modelo lo trata como "otros") | **Escalar**: se da el resultado con aviso y revisión humana | `revisar_entrada(...)["escalar"]` |
| E3 | El estudiante pertenece a un subgrupo en alerta y el resultado es "sin riesgo" | **No descartar**: el "sin riesgo" no se usa para excluirlo del acompañamiento; se muestra el recall del subgrupo | `alertas` en `reports/metrics/analisis_errores.json` |

7. **Variables sensibles.** El modelo elegido conserva `Gender`. Quitarla después de usar el test exigiría una nueva decisión y una nueva evaluación con datos no vistos. La comparación con y sin variables sensibles se reporta como evidencia para una versión futura (ver justificación).

## 4. Justificación

Resultados en [reports/analisis_errores.md](../../reports/analisis_errores.md).

1. **Por qué ±0.05.** En la zona [0.332, 0.432] la tasa real de deserción es 0.24, por debajo de la tasa global (0.32): marcar a esos estudiantes como "riesgo" no es mejor que elegir al azar, y dejarlos como "sin riesgo" pierde a 1 de cada 4. La zona recoge el 21 % de los casos, pero también el 36 % de los falsos negativos y el 24 % de los falsos positivos. Fuera de ella, el recall sube de 0.81 a 0.86 y la precisión de 0.45 a 0.49. Con ±0.10 se evitarían más errores, pero el 40 % de los estudiantes iría a revisión manual.
2. **Por qué una alerta y no excluir al subgrupo.** El error costoso es el falso negativo (proyecto.md). En los subgrupos con alerta, el modelo deja sin detectar a muchos más desertores, así que su "sin riesgo" no es fiable; su "riesgo", en cambio, sigue siendo útil para priorizar. Por eso E3 solo anula el uso del "sin riesgo".
3. **Por qué 0.10 y 30.** Las brechas encontradas son de 0.15 a 0.40 con intervalos lejos de 0, así que la conclusión no depende del valor exacto. Con 32 subgrupos hay comparaciones múltiples: las alertas tienen un límite superior del IC de −0.07 o menos, y no desaparecerían con una corrección conservadora.
4. **Por qué no se quita `Gender` ahora.** Sin `Gender`, la brecha de recall entre mujeres y hombres baja de −0.22 a −0.11 y el recall global cambia −0.011 (IC [−0.026, +0.004], ruido). Es un argumento fuerte para una versión futura, pero la brecha por edad (−0.40) no cambia y el modelo ya se evaluó en test con `Gender`.

**Supuestos:** el coste de un falso positivo es bajo porque hay revisión humana; las cohortes futuras se parecen a las del dataset; los rangos de train son una aproximación mínima del dominio (no detectan combinaciones raras de valores individualmente normales).

## 5. Consecuencias

- `src/errores.py` implementa el análisis y las reglas: `subgroup_metrics`, `error_profile`, `clasificar`, `abstention_table`, `dominio_entrenamiento`, `revisar_entrada` y `neutralize`. Se ejecuta con `uv run python -m src.errores`.
- Los parámetros están en `src/config.py`: `ANCHO_ABSTENCION`, `MAX_BRECHA_RECALL`, `MIN_RIESGO_SUBGRUPO`, `LIMITES_EDAD` y `SUBGRUPOS`.
- #17 (demo) debe aplicar A1, E1, E2 y E3 con esas funciones y mostrar D1–D3 como advertencias de uso.
- #18 (informe) usa los hallazgos y las limitaciones éticas de `reports/analisis_errores.md` en la sección de uso responsable.
- Salidas: `reports/metrics/analisis_errores.json` y `.md`, `reports/figures/14_subgrupos.png` y `14_zona_abstencion.png`.

## 6. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | POR COMPLETAR: enlace al PR |
