# Análisis de errores y de subgrupos (#14)

Dónde y con quién se equivoca el modelo elegido en #13 (Random Forest, umbral 0.382), si perjudica de forma sistemática a algún grupo y cuándo no debe usarse su resultado. Criterios en la [decisión 009](../docs/decisiones/009_abstencion_y_subgrupos.md).

Se reproduce con:

```bash
uv run python -m src.errores
```

Salidas: `reports/metrics/analisis_errores.json`, `reports/metrics/analisis_errores.md` (todas las tablas, incluidas las de cada curso) y las figuras `reports/figures/14_subgrupos.png` y `14_zona_abstencion.png`.

**Datos.** Probabilidades out-of-fold del Random Forest en train (3,539 estudiantes, 1,137 desertores): cada estudiante lo predice un modelo que no lo vio al entrenar. Son las mismas predicciones de la matriz de confusión de #13 (921 desertores detectados, 216 no detectados, 1,114 falsas alarmas). **El test no se vuelve a usar** (decisión 003). Los intervalos son bootstrap al 95 % (2,000 remuestreos).

## 1. Resumen

| N.º | Hallazgo | Consecuencia |
|---|---|---|
| H1 | **La edad decide casi todo.** El 88 % de los desertores no detectados (190 de 216) tiene 20 años o menos. En ese grupo el modelo detecta a 56 de cada 100 desertores; con más de 25 años marca al 99 % de los estudiantes | Para los menores de 21 un "sin riesgo" no es fiable; para los mayores de 25 el modelo no prioriza: los marca a todos |
| H2 | **Las mujeres desertoras se detectan menos.** Recall 0.70 frente a 0.92 en hombres; el 81 % de los no detectados son mujeres (174 de 216), y el 71 % son mujeres de 20 años o menos | Subgrupo en alerta (E3). Sin `Gender` la brecha se reduce a la mitad sin perder recall global |
| H3 | **Los desertores no detectados "parecen buenos estudiantes" al ingreso.** Frente a los detectados: 8.5 años más jóvenes, nota de admisión 5 puntos más alta, más becarios (20 % frente a 8 %), casi todos en horario diurno y más desplazados | Al ingreso no hay señal para ellos; su riesgo aparece después (rendimiento del primer semestre, prohibido por la decisión 001) |
| H4 | **Un tercio de las falsas alarmas son estudiantes Enrolled.** El modelo marca al 60 % de los Enrolled y al 42 % de los Graduate | Parte de las "falsas alarmas" son estudiantes que no se graduaron a tiempo, para quienes el acompañamiento también tiene sentido |
| H5 | **Tres cursos concentran los no detectados.** Nursing (recall 0.53), Social Service (0.45) y Journalism and Communication (0.65) suman 101 de los 216 | Subgrupos en alerta. En cambio, los cursos nocturnos se marcan casi enteros (falsos positivos 0.93–0.95) |
| H6 | **La zona cercana al umbral no informa.** Entre 0.332 y 0.432 está el 21 % de los estudiantes, con una tasa real de deserción de 0.24, por debajo de la global (0.32) | Zona de abstención: resultado "no concluyente" y revisión humana (E1) |

## 2. Métricas por subgrupo

![Recall y falsos positivos por subgrupo](figures/14_subgrupos.png)

Recall = proporción de desertores del subgrupo que el modelo detecta. FPR (tasa de falsos positivos) = proporción de no desertores del subgrupo que el modelo marca. La brecha compara el subgrupo con el resto de la población. Global: recall 0.81, FPR 0.46, precisión 0.45.

| Variable | Subgrupo | n | Desertores | Prevalencia | Marcados | Recall (IC 95 %) | FPR (IC 95 %) | Precisión | Brecha de recall (IC 95 %) | Alerta |
|---|---|---|---|---|---|---|---|---|---|---|
| Género | Mujer | 2,298 | 581 | 0.25 | 0.44 | 0.70 [0.66, 0.74] | 0.36 [0.34, 0.38] | 0.40 | −0.22 [−0.27, −0.18] | **sí** |
| | Hombre | 1,241 | 556 | 0.45 | 0.82 | 0.92 [0.90, 0.95] | 0.73 [0.69, 0.76] | 0.51 | +0.22 | no |
| Edad | Hasta 20 | 2,061 | 435 | 0.21 | 0.34 | 0.56 [0.52, 0.61] | 0.29 [0.26, 0.31] | 0.35 | −0.40 [−0.45, −0.35] | **sí** |
| | 21 a 25 | 648 | 237 | 0.37 | 0.78 | 0.90 [0.86, 0.94] | 0.71 [0.66, 0.75] | 0.42 | +0.11 | no |
| | Más de 25 | 830 | 465 | 0.56 | 0.99 | 1.00 [0.99, 1.00] | 0.98 [0.97, 0.99] | 0.56 | +0.31 | no |
| Estado civil | Soltero/a | 3,144 | 948 | 0.30 | 0.53 | 0.77 [0.75, 0.80] | 0.42 [0.40, 0.44] | 0.44 | −0.22 [−0.25, −0.18] | **sí** |
| | Otro | 395 | 189 | 0.48 | 0.96 | 0.99 [0.97, 1.00] | 0.93 [0.89, 0.96] | 0.49 | +0.22 | no |
| Becario | No | 2,645 | 1,022 | 0.39 | 0.65 | 0.83 [0.81, 0.85] | 0.53 [0.51, 0.56] | 0.50 | +0.20 | no |
| | Sí | 894 | 115 | 0.13 | 0.36 | 0.63 [0.54, 0.71] | 0.32 [0.29, 0.35] | 0.23 | −0.20 [−0.30, −0.11] | **sí** |
| Desplazado | Sí | 1,942 | 534 | 0.27 | 0.47 | 0.73 [0.69, 0.77] | 0.37 [0.35, 0.40] | 0.43 | −0.15 [−0.19, −0.10] | **sí** |
| | No | 1,597 | 603 | 0.38 | 0.70 | 0.88 [0.85, 0.91] | 0.59 [0.56, 0.62] | 0.47 | +0.15 | no |
| Nacionalidad (internacional) | No (portuguesa) | 3,453 | 1,114 | 0.32 | 0.57 | 0.81 [0.79, 0.83] | 0.46 [0.44, 0.48] | 0.45 | +0.03 | no |
| | Sí (otra nacionalidad) | 86 | 23 | 0.27 | 0.58 | 0.78 [0.61, 0.96] | 0.51 [0.40, 0.63] | 0.36 | −0.03 [−0.21, +0.14] | sin evidencia |
| Necesidades especiales | No | 3,498 | 1,124 | 0.32 | 0.58 | 0.81 [0.79, 0.84] | 0.47 [0.45, 0.49] | 0.45 | +0.27 | no |
| | Sí | 41 | 13 | 0.32 | 0.41 | 0.54 [0.23, 0.77] | 0.36 [0.18, 0.54] | 0.41 | −0.27 [−0.57, −0.03] | sin evidencia |
| Curso | Nursing | 607 | 90 | 0.15 | 0.17 | 0.53 [0.43, 0.63] | 0.11 [0.08, 0.14] | 0.46 | −0.30 [−0.41, −0.20] | **sí** |
| | Social Service | 291 | 55 | 0.19 | 0.27 | 0.45 [0.33, 0.58] | 0.23 [0.18, 0.29] | 0.31 | −0.37 [−0.51, −0.25] | **sí** |
| | Journalism and Communication | 267 | 82 | 0.31 | 0.48 | 0.65 [0.55, 0.74] | 0.41 [0.34, 0.48] | 0.41 | −0.18 [−0.28, −0.07] | **sí** |

Los 17 cursos están en `reports/metrics/analisis_errores.md`. "Sin evidencia" = menos de 30 desertores en el subgrupo; sus intervalos son demasiado anchos para concluir (decisión 009).

**Nacionalidad y estudiante internacional son el mismo subgrupo.** En los datos, `International` = 1 si y solo si `Nacionality` no es portuguesa ([auditoría](auditoria_datos.md)), así que la fila "Nacionalidad (internacional)" cubre las dos variables del issue. Desglosar por país no es posible: en train hay 18 nacionalidades extranjeras y la que más desertores tiene suma 7, muy por debajo del mínimo de 30.

**Brechas significativas.** Ocho subgrupos quedan en alerta: mujeres, 20 años o menos, solteros/as, becarios, desplazados y los cursos Nursing, Social Service y Journalism and Communication. En todos, el límite superior del intervalo de la brecha está en −0.07 o menos, así que no son ruido aunque se hagan 32 comparaciones.

**¿Perjudica sistemáticamente a algún grupo?** Depende de qué error se mire, y por eso la respuesta tiene dos partes:

- **Recall (el error costoso).** Las mujeres, los menores de 21, los becarios y los desplazados que van a desertar reciben menos atención: el modelo los detecta menos. Con una intervención de apoyo, el perjuicio es **quedarse sin acompañamiento**, y recae en esos grupos.
- **Falsos positivos (el error barato).** Los hombres (0.73), los mayores de 25 (0.98) y los cursos nocturnos se marcan mucho más sin necesidad. Con acompañamiento voluntario y revisión humana el coste es bajo, pero si la etiqueta se usara para algo más que apoyo (estigma, sanción), el perjuicio caería sobre ellos. Por eso los usos excluidos de [proyecto.md](../docs/proyecto.md) son una condición de detención (D2).

**Por qué ocurre.** Con un único umbral, los grupos con menos deserción (mujeres 25 %, menores de 21 21 %, becarios 13 %) reciben probabilidades más bajas y caen más por debajo del umbral. La edad es la variable que más pesa: las brechas de estado civil y de los cursos nocturnos son en buena parte de edad (mediana de edad: 36 años en quienes no son solteros y 32 en horario nocturno, frente a 20 en el resto). La brecha de género puede mezclarse con el curso (V de Cramér 0.42 entre ambos, [auditoría](auditoria_datos.md)), pero no se explica solo por él: sin `Gender` se reduce a la mitad (sección 6).

## 3. Falsos negativos: desertores no detectados

Es el error más costoso: el estudiante abandona sin recibir acompañamiento. Son 216 de 1,137 desertores (19 %).

| Variable | Detectados (TP) | No detectados (FN) | FN − TP |
|---|---|---|---|
| Edad al ingreso (media) | 27.7 | 19.2 | −8.5 |
| Nota de admisión (media, 0–200) | 123.8 | 129.0 | +5.1 |
| Nota de la titulación previa (media) | 130.1 | 135.3 | +5.2 |
| Orden de preferencia de la solicitud (media) | 1.45 | 2.07 | +0.62 |
| Hombres | 56 % | 19 % | −36 puntos |
| Desplazados | 42 % | 66 % | +24 puntos |
| Horario diurno | 82 % | 99 % | +16 puntos |
| Becarios (no está en el modelo) | 8 % | 20 % | +12 puntos |
| Matrícula al día (no está en el modelo) | 66 % | 75 % | +8 puntos |
| Deudores (no está en el modelo) | 22 % | 22 % | 0 |

El perfil típico del desertor no detectado es una estudiante de 18 a 20 años, en horario diurno, desplazada, con buena nota de admisión y que no entró en su primera opción. Por edad, el 88 % tiene 20 años o menos; por curso, 101 de los 216 son de Nursing (42), Social Service (30) y Journalism and Communication (29).

A diferencia de los desertores detectados, la mayoría tiene la matrícula al día y una nota de admisión alta. Con las variables disponibles al ingreso, no se distinguen de quienes terminan. Su riesgo probablemente aparece en el primer semestre, cuyo rendimiento está prohibido en este escenario (decisión 001). Esta es la limitación principal del enfoque de intervención al ingreso.

## 4. Falsos positivos: acompañamiento innecesario

Son 1,114 de 2,402 no desertores (46 %). El coste es una entrevista o tutoría que el estudiante no necesitaba.

| Variable | Marcados sin desertar (FP) | Bien clasificados (TN) | FP − TN |
|---|---|---|---|
| Edad al ingreso (media) | 25.2 | 19.1 | +6.2 |
| Nota de admisión (media) | 125.7 | 129.7 | −4.0 |
| Hombres | 45 % | 14 % | +30 puntos |
| Horario diurno | 82 % | 99 % | −17 puntos |
| Becarios (no está en el modelo) | 22 % | 41 % | −19 puntos |

Los falsos positivos son el reflejo de los falsos negativos: estudiantes mayores, hombres, de horario nocturno y con notas algo más bajas, que se parecen a los desertores típicos pero terminan. **379 de las 1,114 falsas alarmas (34 %) son estudiantes Enrolled**, es decir, que no se graduaron en el plazo normal. El modelo marca al 60 % de los Enrolled frente al 42 % de los Graduate. Para la coordinación, esa parte de las "falsas alarmas" son estudiantes con retraso a los que el acompañamiento también puede servir. Enrolled no es lo mismo que graduado (decisión 002).

## 5. Casos de baja confianza y zona de abstención

![Zona de abstención](figures/14_zona_abstencion.png)

La probabilidad del modelo crece de forma ordenada con la tasa real de deserción (panel derecho), pero cerca del umbral las dos clases se solapan mucho (panel izquierdo). Un cambio pequeño en la probabilidad cambia la clase, así que la etiqueta de esos casos no tiene respaldo.

| Ancho alrededor del umbral | Zona | Casos en zona | Tasa real de deserción en zona | % de los FN en zona | % de los FP en zona | Recall fuera de la zona | Precisión fuera de la zona |
|---|---|---|---|---|---|---|---|
| Sin zona | — | 0 | — | — | — | 0.810 | 0.453 |
| ±0.025 | [0.357, 0.407] | 374 (10.6 %) | 0.26 | 22.7 % | 11.3 % | 0.839 | 0.469 |
| **±0.05 (elegido)** | **[0.332, 0.432]** | **747 (21.1 %)** | **0.24** | **36.1 %** | **23.5 %** | **0.856** | **0.491** |
| ±0.075 | [0.307, 0.457] | 1,093 (30.9 %) | 0.26 | 58.3 % | 33.0 % | 0.895 | 0.507 |
| ±0.10 | [0.282, 0.482] | 1,433 (40.5 %) | 0.24 | 67.6 % | 42.2 % | 0.912 | 0.528 |

La zona sugerida en el issue (0.40–0.60) no sirve aquí porque el umbral es 0.382, no 0.5: quedaría casi toda del lado "riesgo". Se usa una zona simétrica alrededor del umbral, de ±0.05, porque en ella la tasa real de deserción (0.24) está por debajo de la global (0.32): marcarlos no es mejor que elegir al azar y descartarlos pierde a 1 de cada 4. Con zonas más anchas se evitan más errores, pero a partir de ±0.075 un tercio de los estudiantes iría a revisión manual y la lista deja de priorizar.

Con la zona, la salida tiene tres niveles:

| Resultado | Estudiantes | Desertores entre ellos |
|---|---|---|
| Riesgo (≥ 0.432) | 1,673 (47 %) | 821 (49 %) |
| No concluyente (0.332–0.432) | 747 (21 %) | 178 (24 %) |
| Sin riesgo (< 0.332) | 1,119 (32 %) | 138 (12 %) |

La zona no es igual para todos: va a revisión el 28 % de los menores de 21, el 22 % de los de 21 a 25 y el 3 % de los mayores de 25. Por tanto, los grupos con menor recall son también los que más se benefician de la revisión humana.

## 6. ¿Excluir las variables sensibles?

Se repitió la validación cruzada del mismo Random Forest (mismos hiperparámetros) con las variables fijadas en un valor constante, para que el modelo no pueda usarlas. Cada variante usa su propio umbral con el criterio de la decisión 008.

| Variante | Umbral | F1-macro | Recall | Δ recall frente al elegido (IC 95 %) | Precisión | Recall mujer / hombre | Brecha mujer (IC 95 %) | Recall hasta 20 |
|---|---|---|---|---|---|---|---|---|
| Modelo elegido | 0.382 | 0.620 | 0.810 | — | 0.453 | 0.70 / 0.92 | −0.22 [−0.27, −0.18] | 0.56 |
| Sin `Gender` | 0.388 | 0.621 | 0.799 | −0.011 [−0.026, +0.004] | 0.453 | 0.75 / 0.85 | −0.11 [−0.15, −0.06] | 0.55 |
| Sin demográficas sensibles¹ | 0.401 | 0.620 | 0.790 | −0.020 [−0.039, −0.001] | 0.453 | 0.75 / 0.84 | −0.09 [−0.14, −0.04] | 0.60 |

¹ `Gender`, `Age at enrollment`, `Marital Status`, `Nacionality`, `Displaced` y `Educational special needs`.

- **Sin `Gender`**, la brecha de recall entre mujeres y hombres baja a la mitad (−0.22 → −0.11) y el recall global cambia menos de lo que se distingue del ruido. Quitar el género casi no cuesta y reduce la desigualdad.
- La brecha **no desaparece**: el curso, el horario y la titulación previa siguen funcionando como proxies del género. Excluir una variable no garantiza la equidad.
- **Sin las demográficas**, el recall global baja 0.02 (diferencia real, aunque pequeña). La brecha por edad baja de −0.40 a −0.31 pero sigue siendo grande, porque la edad también se refleja en el modo de acceso, el curso y la titulación previa.

**Recomendación (no aplicada).** El modelo elegido conserva `Gender`, porque ya se evaluó en test con esa variable y cambiarlo ahora requeriría una evaluación nueva con datos no vistos (decisiones 003 y 009). Para una versión futura, o antes de cualquier aplicación local, se recomienda el modelo sin `Gender`: rinde igual y es menos desigual. La edad no se recomienda quitarla, porque la brecha seguiría por los proxies y se perdería recall. Para la edad sirven mejor la zona de abstención y la regla E3.

## 7. Condiciones de detención, abstención y escalamiento

Las define la [decisión 009](../docs/decisiones/009_abstencion_y_subgrupos.md) y las aplica la demo (#17) con las funciones de `src/errores.py`:

| N.º | Condición | Acción |
|---|---|---|
| D1 | Uso en otra institución, país o plan de estudios, o en cohortes nuevas sin revalidar | **Detener** |
| D2 | Uso para sancionar, excluir, decidir admisiones o becas, o sin revisión humana | **Detener** |
| D3 | En una cohorte nueva con desenlace conocido, recall global < 0.75 | **Detener** y revisar o reentrenar |
| A1 | Falta un valor, numérica fuera del rango de train o código nunca visto (`revisar_entrada`) | **Abstenerse**: sin resultado |
| E1 | Probabilidad en [0.332, 0.432] (`clasificar`) | **Escalar**: "no concluyente, requiere revisión humana" |
| E2 | Código nominal infrecuente en train (< 10 casos) | **Escalar**: resultado con aviso |
| E3 | Subgrupo en alerta y resultado "sin riesgo" | **No descartar**: el "sin riesgo" no excluye del acompañamiento |

**Revisión de la entrada (A1 y E2) en validación cruzada.** Al comparar cada fold con el dominio de los otros cuatro, el 1.9 % de los casos activaría A1 (sobre todo por códigos de ocupación o titulación de los padres que no aparecen) y el 6.2 % activaría E2 (nacionalidades y titulaciones poco frecuentes). En esos casos el modelo no se equivocó más que en el resto (recall 0.87–0.89 frente a 0.80), así que A1 y E2 son una **precaución**, no una corrección de errores observados: el modelo no tiene datos para respaldar el resultado en esos casos. Con todo train como referencia, los porcentajes serán algo menores.

**Relación con los ejemplos del issue.**

- Entrada fuera de rango → A1.
- Probabilidad en la zona → E1.
- Otra institución o país → D1.
- **Brecha de recall por encima del umbral → E3, no "no usar el modelo para ese subgrupo".** Excluir a las mujeres o a los menores de 21 dejaría sin priorizar a la mayoría del alumnado, y su "riesgo" sigue siendo útil. Lo que falla es su "sin riesgo", y E3 anula solo eso (decisión 009, justificación 2).
- **Perfil muy distinto al de entrenamiento: cubierto solo en parte.** `revisar_entrada` revisa cada variable por separado. No detecta combinaciones raras de valores que son normales uno a uno, por ejemplo una edad y un curso que casi nunca aparecen juntos. Una comprobación multivariante (distancia al vecino más cercano en train) queda como mejora para la demo (#17).

**E3 en la práctica.** Las alertas cubren a la mayoría del alumnado (el 65 % son mujeres y el 58 % tiene 20 años o menos). La conclusión honesta es que, al ingreso, **el modelo sirve para priorizar a quién contactar primero, pero no para descartar a nadie**. Un "sin riesgo" en una estudiante joven no significa que no vaya a abandonar.

## 8. Limitaciones éticas

Insumo para la sección de uso responsable del informe (#18):

1. **Desigualdad en el error costoso.** El modelo detecta peor a las desertoras, a los menores de 21, a los becarios y a los desplazados. Usado sin las reglas E1 y E3, reforzaría que estos grupos reciban menos apoyo.
2. **Etiqueta sobre grupos, no sobre personas.** Marca al 99 % de los mayores de 25 y al 96 % de los estudiantes de horario nocturno: para ellos la etiqueta refleja la pertenencia al grupo, no algo individual. No debe comunicarse al estudiante como un juicio sobre él.
3. **Variables sensibles y proxies.** Género, edad, estado civil y nacionalidad entran en el modelo. Quitarlas reduce poco las brechas, porque otras variables las sustituyen. Cualquier aplicación local debe revisar la normativa sobre el uso de estos datos (RGPD en el contexto original).
4. **Datos de otra realidad.** Una sola institución portuguesa (Portalegre), cohortes sin fecha conocida y sin validación temporal (decisión 003). Las brechas y la zona de abstención pueden ser distintas en otra población (D1).
5. **Subgrupos pequeños sin evaluar.** Estudiantes internacionales (23 desertores) y con necesidades educativas especiales (13): no hay evidencia para afirmar que el modelo sea justo con ellos, tampoco que no lo sea. Requieren revisión humana siempre.
6. **Optimismo leve.** El análisis usa predicciones out-of-fold de train, con hiperparámetros y umbral elegidos con los mismos folds. Las cifras de test de #13 (recall 0.80) muestran que el sesgo es pequeño, pero los subgrupos no se verificaron en test.
7. **Correlación, no causa.** Que la edad o el género predigan la deserción no significa que la causen, ni que el acompañamiento la reduzca (proyecto.md).
8. **Revisión humana obligatoria.** El modelo prioriza; la decisión de contactar, y cómo hacerlo, es de la coordinación académica. Nunca se usa para sancionar, excluir ni decidir admisiones o becas (D2).
