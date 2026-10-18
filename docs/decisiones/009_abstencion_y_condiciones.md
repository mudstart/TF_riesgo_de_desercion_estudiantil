# Decisión 009: zona de abstención y condiciones de detención y escalamiento

- Issue: [#14](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/14) Análisis de errores y de subgrupos (equidad).
- Rama: `feature/14-errores-subgrupos`.
- Responsable: Roberto Byas.
- Fecha: 2026-10-18.
- Estado: Propuesta.
- Reemplaza a: No aplica.

## 1. Contexto

El modelo elegido en #13 (Random Forest, umbral 0.382; [decisión 008](008_criterio_seleccion.md)) detecta 4 de cada 5 desertores, pero más de la mitad de los marcados no deserta y la señal al ingreso es moderada (AUC ≈ 0.75). El issue #14 pide definir por escrito cuándo el modelo debe abstenerse, escalar a revisión humana o dejar de usarse. La demo (#17) aplica estas reglas y la defensa pregunta por ellas.

El análisis de errores ([reports/analisis_errores.md](../../reports/analisis_errores.md)), con las predicciones out-of-fold de train, muestra que:

- alrededor del umbral la fracción real de desertores (20–31 %) no supera la tasa base (32 %);
- el recall es relevantemente menor en mujeres (0.70 frente a 0.92), en estudiantes de 17–19 años (0.50), en becarios (0.63) y en desplazados (0.73);
- en los mayores de 25 años el modelo marca al 99 %, de modo que la marca no distingue a nadie dentro del grupo.

## 2. Opciones evaluadas

| Aspecto | Opción | Ventaja | Desventaja |
|---|---|---|---|
| Datos del análisis | Test | Datos no vistos | Ya se usó una vez (decisión 003); fijar reglas mirándolo lo contamina; subgrupos 4 veces más pequeños |
| | **Out-of-fold de train, mismos folds y umbral** | Sin fuga, reproduce #13, más filas por subgrupo | Las cifras de subgrupos no se confirman en test |
| Zona de abstención | Fija 0.40–0.60 (ejemplo del issue) | Simple | Con umbral 0.382, todo ese tramo ya es "riesgo": 1,143 casos con 65 % de error, sin relación con la incertidumbre del modelo |
| | **Umbral ± ancho** | Se centra donde el modelo no aporta información | Hay que fijar el ancho |
| Ancho | ±0.04 | 17 % de casos a revisión | Pierde 149 desertores como "sin riesgo" |
| | **±0.06** | 25 % a revisión; pierde 116 desertores (10 %) frente a 216 (19 %) sin zona | Más carga de revisión |
| | ±0.08 | Pierde 87 desertores | 33 % de los casos sin recomendación |
| Brechas | Solo comparar puntos | Simple | Confunde ruido con brecha en grupos pequeños |
| | **Bootstrap por grupo + umbral de relevancia 0.10** | Separa brechas reales de ruido y descarta las triviales | El umbral 0.10 es una convención del equipo |
| Respuesta a una brecha | Umbral distinto por grupo | Iguala el recall | Decide explícitamente con el género o la edad: trato diferenciado difícil de justificar |
| | **Escalar a revisión humana los "sin riesgo" del grupo desfavorecido** | No usa el atributo para perjudicar; corrige el error más costoso | Más carga de revisión |

## 3. Decisión

1. **Zona de abstención:** probabilidades en [umbral − 0.06, umbral + 0.06) = **[0.32, 0.44)** se reportan como "no concluyente, requiere revisión humana" (`ANCHO_ABSTENCION`, `decide`).
2. **Brecha relevante:** diferencia de recall de Dropout (o de tasa de falsos positivos) entre niveles de una variable mayor que **0.10** con intervalo bootstrap al 95 % que no contiene 0 (`MAX_BRECHA`).
3. **Grupo con evidencia insuficiente:** menos de **30 desertores** en train (`MIN_DESERTORES_SUBGRUPO`).
4. **Marca no informativa:** el modelo marca a más del **95 %** del grupo (`MAX_TASA_MARCADOS`).
5. **Condiciones de uso**, en este orden (tabla completa en la sección 6 del informe):
   1. Otra institución, país, cohorte o plan de estudios → detener.
   2. Propósito de sancionar, excluir, admitir o asignar becas → detener.
   3. Entrada fuera del contrato aprendido de train (rango numérico, categoría con menos de 10 casos o desconocida, faltante) → abstenerse (`input_ranges`, `out_of_range`).
   4. Probabilidad en la zona → escalar a revisión humana.
   5. "Sin riesgo" en un grupo con recall relevantemente menor (hoy: mujeres, 17–19 años, becarios, desplazados) → escalar; no basta para descartar (`restricciones`).
   6. "Riesgo" en un grupo con marca no informativa (hoy: 25–34 y 35+) → no usar la marca para priorizar dentro del grupo.
   7. Grupo con evidencia insuficiente (hoy: nacionalidad extranjera) → no concluyente.
   8. En una reevaluación con una cohorte nueva, una brecha relevante de recall o un recall global < 0.70 → detener el uso para ese grupo (o en general) hasta reentrenar.

El modelo y el umbral elegidos en la decisión 008 **no cambian**.

## 4. Justificación

1. **La zona está donde el modelo no informa.** En [0.32, 0.44) la tasa real de deserción es del 20–31 %, igual o menor que la base. Escalar esos casos convierte 100 desertores que se perderían en casos revisados por una persona.
2. **±0.06 equilibra pérdida y carga.** Reduce a la mitad los desertores descartados (216 → 116) y deja la recomendación del modelo en el 75 % de los casos.
3. **Escalar en vez de usar umbrales por grupo.** Un umbral más bajo para mujeres igualaría el recall, pero usaría el género para decidir de forma explícita. Escalar a revisión humana corrige el error más costoso (el desertor no detectado) sin tratar distinto a nadie de forma automática.
4. **El test no se reutiliza.** Las reglas se fijan con train para no ajustarlas al test, que ya se usó en #13.
5. **El modelo sin `Gender` es una opción, no un cambio de este issue.** Reduce la brecha de género a la mitad sin pérdida de recall distinguible del ruido, pero cambiar el modelo después de usar el test requiere una decisión que reemplace a la 008.

**Supuestos:** la revisión humana existe y tiene capacidad para ≈ 25 % de casos adicionales; el acompañamiento es de bajo coste y voluntario; las proporciones de los subgrupos en cohortes futuras se parecen a las del dataset.

## 5. Consecuencias

- `src/config.py`: `MODELO_FINAL`, `ANCHO_ABSTENCION`, `MAX_BRECHA`, `MIN_DESERTORES_SUBGRUPO` y `MAX_TASA_MARCADOS`.
- `src/errores.py` implementa el análisis y las funciones que usa la demo (#17): `decide`, `abstention_zone`, `input_ranges`, `out_of_range` y `restricciones`. `tests/test_errores.py` las prueba.
- Salidas: `reports/analisis_errores.md`, `reports/metrics/analisis_errores.json`, `reports/metrics/subgrupos.csv` y `reports/figures/14_*.png`.
- #17 debe mostrar "no concluyente" y los motivos de abstención o escalamiento, no solo "riesgo / sin riesgo".
- #18 usa la sección 8 del informe (limitaciones éticas) en la sección de uso responsable.
- Queda abierta para el equipo la propuesta de retirar `Gender` del modelo (decisión nueva).

## 6. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
| Evidencia | POR COMPLETAR: enlace al PR |
