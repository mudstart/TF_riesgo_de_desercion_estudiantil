# Decisión 002: formulación de la variable objetivo

- Issue: [#5](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/5) Decidir formulación del target: 3 clases o binario.
- Rama: `feature/5-formulacion-target`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-10.
- Estado: aprobada por el equipo.

## 1. Contexto

La variable `Target` del conjunto UCI 697 tiene tres clases, medidas al final de la duración normal del curso:

| Clase | Estudiantes | Proporción |
|---|---|---|
| Graduate | 2,209 | 49.9 % |
| Dropout | 1,421 | 32.1 % |
| Enrolled | 794 | 17.9 % |
| Total | 4,424 | 100 % |

El enunciado exige como evaluación mínima **F1-macro** y **recall de la clase de riesgo**. La formulación del target cambia cómo se interpretan ambas métricas.

`Enrolled` no equivale a graduación: indica que el estudiante seguía matriculado al cierre del periodo observado y no garantiza que no abandone después.

## 2. Opciones evaluadas

| Opción | Ventaja | Desventaja |
|---|---|---|
| Multiclase (3 clases) | Respeta los datos originales | `Enrolled` es ambigua y, al ser la clase minoritaria y más difícil de separar, baja el F1-macro sin aportar información sobre el riesgo |
| Binario: Dropout vs. No-Dropout | Responde directamente al problema de riesgo | Pierde el matiz entre Graduate y Enrolled |
| Binario excluyendo Enrolled | Clases más separables | Elimina 794 estudiantes y sesga la muestra |

## 3. Decisión

- **Formulación principal: binaria.** `Dropout` = 1; `Graduate` y `Enrolled` = 0. No se elimina ninguna fila.
- **Formulación secundaria: multiclase** con las tres clases originales, como comparación.
- **Clase de riesgo:** `Dropout` en ambas formulaciones.
- La formulación principal es la que se usa en la comparación final de modelos (#13) y en la conclusión para el decisor.

## 4. Justificación

1. **La decisión apoyada es binaria.** La coordinación académica valora si prioriza o no el acompañamiento temprano de un estudiante. Un modelo binario responde exactamente esa pregunta y permite ajustar el umbral de clasificación para favorecer el recall de Dropout.
2. **F1-macro es interpretable.** En binario promedia el desempeño en Dropout y No-Dropout. En multiclase lo arrastra `Enrolled`, cuyo error no corresponde a la decisión que se apoya.
3. **No se excluye a Enrolled.** Son estudiantes que no se graduaron en el tiempo normal y probablemente los más parecidos a Dropout. Eliminarlos haría la tarea artificialmente fácil y el desempeño reportado sería optimista respecto de una población real.
4. **La secundaria conserva el matiz.** La formulación multiclase permite comprobar si los modelos distinguen `Enrolled` de `Dropout` y si la conclusión cambia al respetar las clases originales.

## 5. Cálculo de las métricas

| Formulación | F1-macro | Recall de riesgo |
|---|---|---|
| Principal (binaria) | Promedio simple del F1 de las clases 0 y 1 | Recall de la clase 1 (Dropout) |
| Secundaria (multiclase) | Promedio simple del F1 de Graduate, Dropout y Enrolled | Recall de la clase Dropout |

El recall de riesgo se reporta siempre junto a la precisión de Dropout y la matriz de confusión, para no favorecer un modelo que clasifique a todos como riesgo (ver `docs/proyecto.md`, sección 4).

## 6. Consecuencias

- `build_target(df)` en `src/` aplica esta decisión y genera ambas formulaciones.
- La columna `Target` original se conserva para el análisis de errores (#14): por ejemplo, cuántos estudiantes `Enrolled` marca el modelo como riesgo.
- Los modelos del baseline y candidatos (#9, #11, #12) se entrenan y evalúan en ambas formulaciones, lo que duplica esa parte del trabajo.
- La partición estratificada (#7) se estratifica por la clase original de tres valores, de modo que ambas formulaciones comparten la misma partición.
- `docs/proyecto.md`, sección 6, se actualiza para reflejar esta decisión.

## 7. Aprobación

| Campo | Valor |
|---|---|
| Acordado por | Roberto Byas, Mannix Mueses y Jose Maduro |
| Medio y fecha del acuerdo | POR COMPLETAR |
| Evidencia verificable | POR COMPLETAR: comentario en el issue #5 o revisión del PR |
