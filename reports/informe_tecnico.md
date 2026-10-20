# Informe técnico: riesgo de deserción estudiantil al ingreso

INF-8239 Ciencia de Datos II · Trabajo final · Modalidad A, Tema 01 (Modelos tabulares y Green AI)

Equipo: Roberto Byas, Manicaotex Mueses y Jose M. Maduro. Repositorio: https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil

> Este informe se puede leer sin abrir el código. Cada afirmación remite al documento de decisión, al informe parcial o al archivo de métricas versionado del que sale.

## Resumen

El proyecto evalúa si la coordinación académica puede identificar, **al ingreso**, a los estudiantes con riesgo de abandonar para priorizar su acompañamiento. Usa el dataset UCI 697 (4,424 estudiantes de una institución portuguesa) **solo con las 18 variables disponibles al matricularse**, excluyendo el rendimiento semestral, que inflaría los resultados con información futura. Se compararon una regresión logística y dos modelos candidatos con un criterio fijado antes de usar el test. Ganó un **Random Forest con umbral 0.382**, que en datos no vistos **detecta al 80 % de los desertores** (IC 95 % [0.755, 0.848]) con una **precisión de 0.45** y un F1-macro de 0.61. La mejora sobre el baseline es real pero pequeña, y su costo energético, despreciable. El modelo **falla más con las mujeres, los menores de 21 años y tres carreras**, cuyos desertores se parecen al ingreso a quienes terminan. Por eso la salida tiene tres niveles (riesgo, **no concluyente con revisión humana** y sin riesgo) y reglas explícitas para abstenerse, escalar o detener su uso. **Conclusión:** el modelo sirve para **ordenar a quién contactar primero**, nunca para descartar, sancionar ni excluir a nadie, y antes de usarlo en otra institución debe revalidarse con datos locales.

## 1. Problema, pregunta delimitada y decisor

**Decisor.** La coordinación académica de una institución de educación superior.

**Problema.** Una parte importante de los estudiantes abandona sus estudios, y cuando el abandono se hace visible (malas notas, unidades no aprobadas) suele ser tarde para acompañarlos. La coordinación necesita saber si, con lo que ya conoce de un estudiante **al ingreso**, puede identificar a quienes conviene ofrecer acompañamiento temprano.

**Decisión apoyada.** Priorizar el contacto, las tutorías y el acompañamiento académico temprano, siempre con revisión humana. El proyecto **no** apoya sancionar, excluir, admitir o rechazar solicitudes, retirar becas, etiquetar a una persona como desertora ni automatizar decisiones individuales ([proyecto.md](../docs/proyecto.md), sección 5).

**Pregunta delimitada** ([proyecto.md](../docs/proyecto.md), sección 4):

> ¿Con qué nivel de recall de la clase Dropout y de F1-macro, acompañados de la precisión de Dropout, pueden identificarse estudiantes con riesgo de deserción en el conjunto de datos UCI utilizando exclusivamente variables cuya disponibilidad al ingreso esté documentada, para orientar la priorización de acompañamiento por la coordinación académica?

El **recall de Dropout** es la proporción de estudiantes que abandonaron y que el modelo marca como riesgo: es la métrica principal, porque dejar sin detectar a un desertor le quita el acompañamiento. La **precisión** evita premiar a un modelo que marque a todos, y el **F1-macro** resume el desempeño en ambas clases. No se fijó un valor objetivo antes de evaluar.

Es un proyecto de **Modalidad A** (análisis reproducible): no se presenta como un sistema validado para decidir sobre estudiantes reales. El análisis tampoco demuestra que el acompañamiento reduzca la deserción.

## 2. Datos: origen, licencia y limitaciones

Fuente: *Predict Students' Dropout and Academic Success*, UCI Machine Learning Repository, conjunto 697 ([ficha de datos](../docs/ficha_datos.md), #3).

| Elemento | Valor |
|---|---|
| Contexto | Instituto Politécnico de Portalegre (Portugal), 17 titulaciones de pregrado, cohortes 2008/09 a 2018/19 |
| Tamaño | 4,424 estudiantes × 37 columnas (36 características y `Target`), sin nulos ni duplicados |
| Clases | Graduate 2,209 (49.9 %) · **Dropout 1,421 (32.1 %)** · Enrolled 794 (17.9 %) |
| Licencia | CC BY 4.0, con atribución obligatoria (ver [Referencias](#referencias)) |
| Versión | DOI 10.24432/C5MC89; última actualización en UCI 2024-02-26 |
| Huella | SHA-256 `a1b1a653…85b74801b`, fijada en `src/config.py`: si UCI cambia el archivo, la descarga falla |

**Limitaciones que condicionan las conclusiones** (ficha, sección 6; [auditoría de datos](auditoria_datos.md), #6):

1. **Una sola institución, en Portugal, y cohortes antiguas.** Nada de lo que se concluye vale para otra institución (por ejemplo, la UASD) ni para cohortes recientes sin revalidar.
2. **Sin año ni cohorte.** No se puede validar "con el pasado para predecir el futuro": la partición es aleatoria (sección 4).
3. **Contradicción en la documentación.** La fuente dice que los datos son "conocidos al matricularse", pero incluye el rendimiento del 1.er y 2.º semestre. Usarlo sin filtrar es fuga (sección 3).
4. **Deserción en sentido amplio.** Los autores cuentan como Dropout también los cambios de carrera o de institución.
5. **Preprocesamiento no auditable.** Los autores trataron anomalías y faltantes antes de publicar, sin detallar cómo.
6. **Variables codificadas.** Las categóricas son códigos enteros (hasta 46 ocupaciones, 21 nacionalidades, 17 cursos) que deben tratarse como nominales, y muchas categorías tienen menos de 10 estudiantes.
7. **Subgrupos pequeños.** Solo 110 estudiantes internacionales y 51 con necesidades educativas especiales: el análisis de equidad no puede evaluarlos con confianza (sección 6).

## 3. Momento de intervención y prevención de fuga

**Momento de intervención: al ingreso**, con la matrícula disponible y antes de conocer cualquier resultado del primer semestre ([decisión 001](../docs/decisiones/001_momento_intervencion.md), #4). Las 36 características se clasificaron según cuándo se conocen:

| Grupo | Variables | Uso |
|---|---|---|
| Ingreso (18) | Datos demográficos, de acceso (modo, curso, orden de preferencia, formación y notas previas, nota de admisión) y de la familia | **Modelo** |
| Financieras (3) | Deudor, matrícula al día, becario | Fuera del modelo: su fecha de registro no está documentada |
| Macroeconómicas (3) | Desempleo, inflación, PIB | Fuera del modelo: fecha no documentada y actúan como identificador de cohorte |
| 1.er y 2.º semestre (12) | Unidades inscritas, evaluadas, aprobadas, notas… | **Prohibidas**: son resultados posteriores al ingreso |

**Por qué importa.** Un análisis exploratorio con la misma regresión logística muestra el tamaño de la fuga posible (decisión 001, sección 5): con las 18 variables de ingreso, el F1-macro es 0.586; añadiendo el primer semestre sube a 0.827, y con todas las variables, a 0.851. **Un F1-macro cercano a 0.85 en este dataset casi seguro usa información futura.** Las financieras también lo suben a 0.692 (`Tuition fees up to date` = 0 corresponde a un 87 % de Dropout), y si se registraron al extraer los datos y no al ingreso, serían fuga. Por eso el proyecto acepta un desempeño más modesto a cambio de que sea real en el momento de decidir.

**Formulación del target** ([decisión 002](../docs/decisiones/002_formulacion_target.md), #5): binaria, Dropout = 1 frente a Graduate y Enrolled = 0, sin eliminar filas. La coordinación decide "acompañar o no", y excluir a los Enrolled (los más parecidos a los desertores) haría la tarea artificialmente fácil.

**Cómo se garantiza en el código.** Las listas de variables están en `src/config.py` y solo se seleccionan con `select_features()`. El preprocesador lanza un error si una variable prohibida llega al modelo. Tres pruebas automáticas sobre los datos reales (#10) comprueban el contrato del dataset, que ninguna variable posterior al ingreso entra en la matriz del modelo y que la partición es válida. Además, una prueba del pipeline (#8) comprueba que cambiar los valores de las variables prohibidas no altera ninguna predicción, y la prueba 4 (#16), que transformar test no cambia nada de lo aprendido con train.

## 4. Metodología: partición, pipeline y modelos

**Partición** ([decisión 003](../docs/decisiones/003_particion_validacion.md), #7). 80 % train (3,539 estudiantes) y 20 % test (885, de ellos 284 desertores), estratificada por las tres clases originales, con semilla 42. Dentro de train, 5 folds fijos de validación cruzada. Los índices se guardan en `data/processed/splits.json` junto con el SHA-256 del dataset. **Regla del equipo: el test se usa una sola vez, en la evaluación final**; toda decisión previa (hiperparámetros, umbral, modelo) se toma con la validación cruzada.

**Pipeline sin fuga** ([decisión 004](../docs/decisiones/004_preprocesamiento.md), #8). Un mismo preprocesamiento para los tres modelos, dentro de un `Pipeline` de scikit-learn que se reajusta en cada fold:

- 9 nominales: imputación por moda y one-hot; las categorías con menos de 10 casos en train, y las nunca vistas, van a una columna de "infrecuentes".
- 4 numéricas (orden de preferencia, nota previa, nota de admisión, edad): imputación por mediana y escalado estándar.
- 3 binarias (desplazado, necesidades especiales, género): sin transformar.
- `International` y `Daytime/evening attendance` quedan fuera porque son redundantes de forma exacta con la nacionalidad y el curso, aunque se conservan para describir subgrupos.
- Desbalance (2.11 : 1): `class_weight="balanced"` en los tres modelos, sin remuestreo.
- Un primer paso rechaza un registro al que le falte una columna, con un mensaje que la nombra (#16).

**Modelos.**

| Modelo | Issue | Ajuste | Hiperparámetros elegidos |
|---|---|---|---|
| Baseline: regresión logística | #9 | `C` en 11 valores; regla de una desviación estándar: el valor más simple a menos de una desviación del mejor ([decisión 005](../docs/decisiones/005_ajuste_y_metricas.md)) | `C` = 0.01 |
| Random Forest | #11 | Búsqueda aleatoria de 25 combinaciones, mayor F1-macro ([decisión 006](../docs/decisiones/006_busqueda_aleatoria.md)) | 300 árboles, `max_features` = log2, hojas de 2 o más, sin límite de profundidad |
| Gradient Boosting (`HistGradientBoosting`) | #12 | Igual, con early stopping sobre un 10 % del train de cada ajuste ([decisión 007](../docs/decisiones/007_gradient_boosting.md)) | `learning_rate` 0.05, profundidad 8, hojas de 20, `l2` = 1, `max_features` 0.5; 83 árboles |

Los hiperparámetros se eligen con F1-macro al umbral 0.5. El **umbral de decisión** se fija después, igual para los tres modelos (sección 5). Como referencia mínima, un clasificador aleatorio estratificado (`DummyClassifier`) obtiene F1-macro 0.508 y recall 0.32.

**Planificado y no ejecutado.** Las decisiones 001 y 002 preveían dos análisis complementarios que **no se realizaron** con los modelos finales: el de sensibilidad con las variables financieras y macroeconómicas, y la formulación multiclase. Solo existe la evidencia exploratoria de la decisión 001 (sección 3). Quedan como trabajo futuro (sección 13).

## 5. Resultados comparativos

Detalle en [comparacion_modelos.md](comparacion_modelos.md) (#13).

**Criterio fijado antes de ver test** ([decisión 008](../docs/decisiones/008_criterio_seleccion.md)). Se versionó en Git antes de la evaluación final: el commit del criterio es de las 10:07 del 2026-10-18 y la evaluación en test, de las 10:19.

1. Cada modelo usa el umbral de **mayor recall de Dropout con F1-macro ≥ 0.62**, elegido con sus predicciones out-of-fold de train. Todos compiten con la misma exigencia de calidad global.
2. Gana el de mayor recall. Si el intervalo bootstrap pareado (95 %) de la diferencia contiene 0, es un empate y gana el más simple (baseline < Random Forest < Gradient Boosting). **El baseline puede ganar.**

**Validación cruzada** (5 folds, umbral de cada fold elegido sin ese fold, media ± desviación):

| Modelo | Umbral (media de los folds) | F1-macro | Recall Dropout | Precisión Dropout |
|---|---|---|---|---|
| Baseline (regresión logística) | 0.431 | 0.620 ± 0.023 | 0.773 ± 0.014 | 0.453 ± 0.021 |
| **Random Forest** | **0.382** | 0.620 ± 0.021 | **0.808 ± 0.021** | 0.454 ± 0.018 |
| Gradient Boosting | 0.385 | 0.618 ± 0.023 | 0.792 ± 0.012 | 0.452 ± 0.020 |
| Referencia: clasificador aleatorio | — | 0.508 | 0.321 | 0.332 |

**¿Diferencias reales o ruido?** (bootstrap pareado sobre las 3,539 predicciones out-of-fold)

| Comparación | Recall: diferencia (IC 95 %) | Lectura |
|---|---|---|
| Random Forest − Baseline | +0.032 [+0.014, +0.051] | **Real**: unos 3 desertores más detectados por cada 100 |
| Gradient Boosting − Baseline | +0.018 [−0.001, +0.037] | Ruido, por poco |
| Random Forest − Gradient Boosting | +0.014 [−0.001, +0.030] | Ruido: empate |

En F1-macro, los tres son iguales (todas las diferencias están en ±0.01). **Selección:** los tres alcanzan el piso; el Random Forest tiene el mayor recall; empata con el Gradient Boosting y el desempate por simplicidad lo elige. **Modelo final: Random Forest con umbral 0.382.**

**Por qué un umbral de 0.382 y no de 0.5.** Bajar el umbral sube el recall del Random Forest de 0.60 a 0.81, a cambio de bajar la precisión de 0.55 a 0.45. Con revisión humana y una intervención de bajo costo (una entrevista o tutoría), es preferible contactar a algunos estudiantes que no lo necesitaban que dejar sin detectar a un desertor.

**Evaluación final en test** (una sola vez, 885 estudiantes, intervalos bootstrap al 95 %):

| Métrica | Test | IC 95 % | Validación cruzada |
|---|---|---|---|
| **Recall Dropout** | **0.803** | [0.755, 0.848] | 0.810 |
| Precisión Dropout | 0.446 | [0.402, 0.488] | 0.453 |
| F1-macro | 0.613 | [0.579, 0.644] | 0.620 |
| Balanced accuracy | 0.666 | [0.633, 0.695] | 0.673 |

![Matriz de confusión en test](figures/13_confusion_test.png)

De los 284 desertores de test, el modelo **detecta 228 y deja sin detectar 56**. Marca a 511 estudiantes, de los que 283 no desertan. Los resultados de test coinciden con los de validación cruzada dentro de sus intervalos, así que no hay señales de sobreajuste al elegir umbral e hiperparámetros.

**Lectura para el decisor.** De cada 10 estudiantes que van a abandonar, el modelo señala a 8 al ingreso. De cada 10 que señala, algo más de 4 abandonan. La señal disponible al ingreso es **moderada** (AUC-ROC ≈ 0.75 en los tres modelos): sirve para ordenar a quién contactar primero, no para afirmar quién va a abandonar.

## 6. Análisis de errores y subgrupos

Detalle en [analisis_errores.md](analisis_errores.md) (#14). Se analizaron las predicciones out-of-fold del modelo final en train (3,539 estudiantes, 1,137 desertores), porque el test ya se había usado. Son las mismas predicciones de la selección: 921 desertores detectados, 216 no detectados y 1,114 falsas alarmas.

**Hallazgos.**

1. **La edad decide casi todo.** El 88 % de los desertores no detectados (190 de 216) tiene 20 años o menos. En ese grupo el modelo detecta a 56 de cada 100 desertores; con más de 25 años marca al 99 % de los estudiantes.
2. **Las mujeres desertoras se detectan menos.** Recall 0.70 frente a 0.92 en hombres; el 81 % de los no detectados son mujeres.
3. **Los desertores no detectados "parecen buenos estudiantes" al ingreso.** Frente a los detectados son 8.5 años más jóvenes, tienen 5 puntos más de nota de admisión y casi todos estudian en horario diurno. Con los datos del ingreso no se distinguen de quienes terminan. Su riesgo aparece probablemente en el primer semestre, cuyos datos están prohibidos en este escenario. **Es la limitación principal del enfoque al ingreso.**
4. **Un tercio de las falsas alarmas son estudiantes Enrolled** (379 de 1,114), es decir, que no se graduaron a tiempo y para quienes el acompañamiento también tiene sentido.
5. **Tres carreras concentran los no detectados**: Nursing (recall 0.53), Social Service (0.45) y Journalism and Communication (0.65) suman 101 de los 216.

**Subgrupos en alerta.** Un subgrupo queda en alerta si tiene al menos 30 desertores y su recall está más de 0.10 por debajo del resto, con un intervalo de la brecha que excluye 0 ([decisión 009](../docs/decisiones/009_abstencion_y_subgrupos.md)):

| Subgrupo | Desertores | Recall (IC 95 %) | Brecha frente al resto (IC 95 %) |
|---|---|---|---|
| Mujer | 581 | 0.70 [0.66, 0.74] | −0.22 [−0.27, −0.18] |
| Hasta 20 años | 435 | 0.56 [0.52, 0.61] | −0.40 [−0.45, −0.35] |
| Soltero/a | 948 | 0.77 [0.75, 0.80] | −0.22 [−0.25, −0.18] |
| Becario (no está en el modelo) | 115 | 0.63 [0.54, 0.71] | −0.20 [−0.30, −0.11] |
| Desplazado | 534 | 0.73 [0.69, 0.77] | −0.15 [−0.19, −0.10] |
| Nursing | 90 | 0.53 [0.43, 0.63] | −0.30 [−0.41, −0.20] |
| Social Service | 55 | 0.45 [0.33, 0.58] | −0.37 [−0.51, −0.25] |
| Journalism and Communication | 82 | 0.65 [0.55, 0.74] | −0.18 [−0.28, −0.07] |

![Recall y falsos positivos por subgrupo](figures/14_subgrupos.png)

Los estudiantes internacionales (23 desertores) y con necesidades educativas especiales (13) **no se pudieron evaluar**: no hay evidencia de que el modelo sea justo con ellos, ni de que no lo sea.

**¿Quitar las variables sensibles?** Se repitió la validación cruzada del mismo Random Forest con las variables fijadas en un valor constante:

| Variante | Recall global | Δ recall (IC 95 %) | Brecha mujer | Brecha hasta 20 |
|---|---|---|---|---|
| Modelo elegido | 0.810 | — | −0.22 | −0.40 |
| Sin `Gender` | 0.799 | −0.011 [−0.026, +0.004] | −0.11 | −0.41 |
| Sin las 6 demográficas sensibles | 0.790 | −0.020 [−0.039, −0.001] | −0.09 | −0.31 |

Quitar el género **reduce a la mitad la brecha entre mujeres y hombres sin una pérdida de recall que se distinga del ruido**. La brecha no desaparece porque el curso, el horario y la formación previa actúan como proxies. El modelo final conserva `Gender` porque ya se evaluó en test con esa variable, y cambiarlo exigiría una evaluación nueva con datos no vistos. **Para una versión futura se recomienda el modelo sin `Gender`** (sección 13).

## 7. Green AI: costo computacional y energético

Detalle en [green_ai.md](green_ai.md) (#15). Se midió con `codecarbon` sin conexión (intensidad de carbono de República Dominicana), en el equipo de referencia del proyecto (AMD Ryzen 5 5600X, 12 hilos, 31.9 GB de RAM, sin GPU), con 3 repeticiones alternando los modelos en un único proceso ([decisión 011](../docs/decisiones/011_medicion_green_ai.md)). El entrenamiento incluye la búsqueda de hiperparámetros.

| Modelo | Entrenamiento | Energía | Emisiones | Inferencia (ms / 1,000 estudiantes) | Tamaño | Recall (CV) |
|---|---|---|---|---|---|---|
| Baseline | 3.5 s | 7.4 × 10⁻⁵ kWh | 0.043 g CO₂eq | 8 | 0.01 MB | 0.773 |
| **Random Forest** | 67 s | 1.5 × 10⁻³ kWh | 0.88 g CO₂eq | 70 | 18.6 MB | **0.808** |
| Gradient Boosting | 44 s | 1.0 × 10⁻³ kWh | 0.60 g CO₂eq | 11 | 0.28 MB | 0.792 |

![Desempeño frente a costo](figures/15_desempeno_vs_costo.png)

**¿La mejora del modelo complejo justifica su costo?**

- **En F1-macro no hay mejora que pagar**: los tres quedan en 0.62. Con ese criterio, el baseline domina con 14 a 20 veces menos energía.
- **En recall, el criterio de selección, el Random Forest sí mejora al baseline**, pero poco: +0.032, unos 3 desertores más por cada 100. A cambio, cuesta unas 20 veces más energía de entrenamiento, 9 veces más tiempo de inferencia y ocupa unas 1,700 veces más espacio.
- **En valores absolutos, el costo es despreciable**: entrenarlo emite menos de 1 g de CO₂eq (0.88 g), lo mismo que una bombilla LED de 10 W encendida unos 9 minutos, y puntuar una cohorte completa (≈ 4,400 estudiantes) tarda un tercio de segundo. Se reentrenaría a lo sumo una vez por cohorte.
- **Pero el Random Forest no es el modelo eficiente.** El Gradient Boosting empata con él en recall (y tampoco se distingue del baseline) con un 68 % de su energía, 7 veces menos tiempo de inferencia y un modelo 66 veces más pequeño. Una medición anterior en otro equipo (un portátil) dio otros valores absolutos, pero el mismo orden y casi las mismas proporciones: las conclusiones no dependen del equipo.

**Conclusión del trade-off.** Para este problema, el costo ambiental no es motivo para renunciar a la mejora de recall, que es el objetivo del proyecto. El costo **valida** la elección del #13 sin cambiarla. Si el costo fuera un criterio explícito o el modelo debiera desplegarse con recursos limitados, la elección sería el Gradient Boosting. El orden de "simplicidad" de la decisión 008 (Random Forest antes que Gradient Boosting) no reflejó el costo real, y queda documentado como limitación del criterio.

## 8. Conclusión accionable y delimitada

**Respuesta a la pregunta.** Con solo las variables disponibles al ingreso, un Random Forest con umbral 0.382 identifica en datos no vistos al **80 % de los estudiantes que abandonan** (recall 0.803, IC 95 % [0.755, 0.848]), con una **precisión de 0.45** (de cada 10 señalados, algo más de 4 abandonan) y un **F1-macro de 0.61**. Es una mejora real pero pequeña sobre una regresión logística (+3 desertores detectados por cada 100) y clara frente al azar (F1-macro 0.51, recall 0.32).

**Qué se puede afirmar:**

1. Al ingreso hay **señal suficiente para ordenar** a quién contactar primero: quien sale como "riesgo" tiene una tasa de deserción del 49 %, y quien sale como "sin riesgo", del 12 % (sección 10).
2. El desempeño es **estable**: test y validación cruzada coinciden dentro de sus intervalos.
3. El costo ambiental y computacional del modelo es **despreciable** frente al beneficio buscado (sección 7).

**Qué no se puede afirmar:**

1. **Que un estudiante concreto vaya a abandonar.** Más de la mitad de los señalados no abandona.
2. **Que un "sin riesgo" sea fiable para todos.** En las mujeres, los menores de 21 años y tres carreras, el modelo deja sin detectar a muchos más desertores (sección 6).
3. **Que el resultado valga fuera de este contexto**: una institución portuguesa, cohortes 2008/09–2018/19, sin validación temporal.
4. **Que las variables del modelo causen la deserción**, ni que el acompañamiento la reduzca.
5. **Nada sobre los estudiantes internacionales ni sobre los que tienen necesidades especiales**: hay demasiado pocos para evaluarlo.

**Recomendación para la coordinación académica:**

1. Usar el modelo **solo para priorizar el contacto temprano**, en tres niveles: contactar primero a los de **riesgo**, revisar caso a caso los **no concluyentes** y **no descartar** a nadie por salir "sin riesgo", en especial si pertenece a un subgrupo en alerta.
2. Como el error principal (desertores jóvenes que "parecen buenos estudiantes") no se puede corregir con datos del ingreso, **repetir la valoración tras el primer semestre**, cuando hay información de rendimiento.
3. **Antes de cualquier aplicación local**, revalidar con datos propios y cohortes recientes, adoptar la variante sin `Gender` (iguala el recall y reduce a la mitad la brecha de género) y comprobar que el recall se mantiene por encima de 0.75 (regla D3).

## 9. Uso responsable

**Uso previsto.** Apoyar a la coordinación académica para **priorizar el acompañamiento temprano** (contacto, entrevista, tutoría) de los estudiantes que acaban de ingresar, **siempre con revisión humana**. El resultado es una orientación sobre a quién contactar primero, no una decisión.

**Usos no previstos y excluidos** ([proyecto.md](../docs/proyecto.md), sección 5):

- Sancionar estudiantes.
- Excluir estudiantes o restringir su continuidad académica.
- Aceptar o rechazar solicitudes de **admisión**.
- Retirar becas o limitar oportunidades educativas.
- Etiquetar a una persona de forma definitiva como "desertora".
- Tomar **decisiones automáticas** sobre un estudiante sin revisión humana.

**Cómo se protege ese uso.** Los usos excluidos se repiten en el README, en este informe y en la demo, que muestra siempre el aviso *"Esta predicción es apoyo para priorizar acompañamiento académico. NO debe usarse para sancionar, excluir ni tomar decisiones automáticas sobre el estudiante."* La demo explica además qué **no** permite concluir cada resultado. El resultado no debe comunicarse al estudiante como un juicio sobre él: para los mayores de 25 años o los de horario nocturno, la etiqueta refleja sobre todo la pertenencia a un grupo (el modelo marca a casi todos). Cualquier aplicación real debe revisar la normativa sobre el uso de datos personales y sensibles (género, edad, estado civil, nacionalidad).

## 10. Zona de abstención y condiciones de detención y escalamiento

Las reglas las define la [decisión 009](../docs/decisiones/009_abstencion_y_subgrupos.md) (#14) y las aplica la demo (#17, [decisión 012](../docs/decisiones/012_demo_prediccion.md)) con las mismas funciones del código.

**Zona de abstención.** Si la probabilidad está a menos de 0.05 del umbral (entre **0.332 y 0.432**), el resultado es **"No concluyente – requiere revisión humana"**. En esa franja la tasa real de deserción (24 %) es incluso menor que la global (32 %): marcar o descartar a esos estudiantes no es mejor que elegir al azar. La salida queda así (predicciones out-of-fold de train):

| Resultado | Probabilidad | Estudiantes | Abandonan |
|---|---|---|---|
| **Riesgo**: priorizar | ≥ 0.432 | 1,673 (47 %) | 49 % |
| **No concluyente**: revisión humana | 0.332–0.432 | 747 (21 %) | 24 % |
| **Sin riesgo** | < 0.332 | 1,119 (32 %) | 12 % |

La zona recoge el 36 % de los desertores no detectados y el 24 % de las falsas alarmas. Fuera de ella, el recall sube de 0.81 a 0.86 y la precisión, de 0.45 a 0.49. Va a revisión el 28 % de los menores de 21 años, justo el grupo con peor recall.

![Zona de abstención](figures/14_zona_abstencion.png)

**Condiciones, en orden de prioridad:**

| N.º | Condición | Acción |
|---|---|---|
| D1 | Uso en otra institución, país o plan de estudios, o en cohortes nuevas sin revalidar | **Detener**: no usar el modelo |
| D2 | Uso para sancionar, excluir, admitir, retirar becas o decidir sin revisión humana | **Detener** |
| D3 | En una cohorte nueva con desenlace conocido, el recall cae por debajo de 0.75 (límite inferior del intervalo en test) | **Detener** y revisar o reentrenar |
| A1 | Falta un dato, un valor numérico está fuera del rango visto en train o un código no se vio nunca | **Abstenerse**: el modelo no da resultado y explica el motivo |
| E1 | La probabilidad cae en la zona [0.332, 0.432] | **Escalar**: "no concluyente, requiere revisión humana" |
| E2 | Un código es infrecuente en train (menos de 10 casos) | **Escalar**: se da el resultado con aviso de revisión |
| E3 | El estudiante pertenece a un subgrupo en alerta y sale "sin riesgo" | **No descartar**: ese "sin riesgo" no se usa para excluirlo del acompañamiento |

Todas son visibles en la demo: con una nota de admisión de 250 se abstiene ("fuera del rango de train [95, 190]"); un perfil con probabilidad 0.40 aparece como no concluyente; y un perfil de una estudiante de 18 años de Enfermería con riesgo bajo (0.14) muestra el aviso E3. La revisión de la entrada solo mira cada variable por separado: no detecta combinaciones raras de valores normales.

### Respuesta a la pregunta de defensa

> *¿Qué evidencia justifica priorizar acompañamiento temprano, nunca sancionar ni excluir, y qué condición obligaría a detener, abstener o escalar la recomendación?*

**Priorizar acompañamiento** se justifica porque, con datos del ingreso, el modelo detecta al 80 % de los desertores en datos no vistos (IC [0.755, 0.848]), y quien sale como "riesgo" abandona en el 49 % de los casos, frente al 12 % de quien sale "sin riesgo". Es una señal útil **para ordenar** un recurso de bajo costo como una tutoría.

**Nunca sancionar ni excluir** se justifica porque la evidencia no da para decidir sobre personas: más de la mitad de los señalados no abandona (precisión 0.45), el error se reparte de forma desigual (recall de 0.56 en los menores de 21 años frente a 1.00 en los mayores de 25, y de 0.70 en mujeres frente a 0.92 en hombres), y el modelo asocia sin explicar causas. Un error que lleve a una sanción sería grave e injusto, mientras que el de una tutoría innecesaria es barato.

**Obligan a detener**: usarlo fuera de su contexto (D1), para fines excluidos (D2) o que el recall caiga por debajo de 0.75 en una cohorte nueva (D3). **Obliga a abstenerse** una entrada incompleta o fuera de lo visto (A1). **Obligan a escalar** a revisión humana una probabilidad en la zona [0.332, 0.432] (E1), un código infrecuente (E2) y cualquier "sin riesgo" en un subgrupo en alerta (E3).

## 11. Trazabilidad: datos → modelo → métricas → conclusión

Cada eslabón tiene un identificador verificable y una prueba o un archivo versionado que lo respalda:

| Eslabón | Identificador | Dónde se verifica |
|---|---|---|
| **Datos** | UCI 697, DOI 10.24432/C5MC89, SHA-256 `a1b1a6531bbb…85b74801b` | `src/config.py` (`DATA_SHA256`); la descarga falla si el archivo cambia; prueba 1 (#10) |
| **Partición** | `data/processed/splits.json`: 3,539 train / 885 test, 5 folds, guarda el SHA-256 del dataset | `load_splits()` rechaza otra versión del dataset o solapamientos; prueba 3 (#10) |
| **Variables y preprocesamiento** | 18 variables de ingreso (16 en el modelo); decisiones 001 y 004 | Listas en `src/config.py`; pruebas 2 y 4 y prueba de variables prohibidas (#8) |
| **Modelo** | Random Forest, hiperparámetros en `reports/metrics/random_forest.json` (búsqueda original del 2026-10-17, reproducida idéntica el 2026-10-20); versión `a53851a8b97d` (SHA-256 de hiperparámetros + umbral + dataset) | La demo lo reconstruye con esos archivos y muestra la versión; prueba 5 (reproducibilidad) |
| **Selección y umbral** | Criterio en la decisión 008, commit `4363443` (2026-10-18 10:07); umbral 0.382 en `reports/metrics/comparacion_cv.json` | El commit del criterio es anterior al de la evaluación en test |
| **Métricas finales** | `reports/metrics/test_final.json` (2026-10-18 10:19, commit `b88881e`) | `src.comparacion --test` se niega a repetirse si el archivo existe; prueba 8 (desempeño mínimo, sin usar test) |
| **Errores y reglas** | `reports/metrics/analisis_errores.json`; decisión 009 | Se reproduce exactamente con `uv run python -m src.errores` |
| **Recomendación** | Demo (#17): muestra modelo, versión, fecha, hash, umbral, zona y métricas de test | `tests/test_predict.py` comprueba que la trazabilidad mostrada coincide con estos archivos |
| **Conclusión** | Este informe | Cada cifra remite a uno de los archivos anteriores |

Todo el código pasa 110 pruebas automáticas en cada PR (GitHub Actions). **Reproducción desde cero en un solo equipo.** El 2026-10-20 se volvió a ejecutar todo el proyecto en el equipo de referencia: descarga del dataset (mismo SHA-256), partición (`splits.json` idéntico byte a byte), auditoría, los tres modelos, la comparación, el análisis de errores y el Green AI. Las métricas, los hiperparámetros, la selección, los umbrales y las alertas salieron idénticos; solo cambiaron las fechas, los tiempos y la energía, que ahora proceden todos de la misma máquina. La evaluación en test **no se repitió** (regla de la decisión 008): se reprodujo en un archivo aparte y dio exactamente el mismo resultado, y `test_final.json` conserva su fecha original (2026-10-18). La comparación original, anterior al uso del test, está en el commit `b88881e`.

## 12. Riesgos del proyecto: materializados y mitigados

**Riesgos del registro inicial** ([proyecto.md](../docs/proyecto.md), sección 7):

| Riesgo | Estado | Evidencia |
|---|---|---|
| Fuga de información por variables posteriores a la intervención | **Mitigado** | Momento de intervención fijado y 12 variables prohibidas (decisión 001); el preprocesador rechaza variables prohibidas; pruebas 2 y 4 y prueba de variables prohibidas (#8, #10, #16) |
| Desbalance de clases y recall insuficiente en Dropout | **Mitigado** | Partición estratificada, `class_weight="balanced"` y umbral ajustado: el recall del modelo final pasa de 0.60 con el umbral 0.5 a 0.81 en validación cruzada y 0.80 en test |
| Sesgo contra subgrupos por género, edad o nacionalidad | **Materializado y mitigado en parte** | Brechas reales de recall por género (−0.22) y edad (−0.40). Se mitigan con la zona de abstención y la regla E3, y se recomienda la variante sin `Gender`. La nacionalidad no se pudo evaluar (23 desertores internacionales) |
| Baja generalización por provenir de una institución de Portugal | **Persiste** (no se puede mitigar con estos datos) | Se limita con la regla D1 y la recomendación de revalidar localmente (secciones 8 y 10) |
| Atraso del cronograma | **No materializado** en los entregables técnicos | Los issues #1–#17 se cerraron entre el 2026-10-10 y el 2026-10-20; quedan #18 (este informe) y #19 (ensayo y defensa) |
| Integrante sin evidencia verificable en Git | **Mitigado** | Los tres tienen PR propios fusionados y revisiones de PR ajenos (sección 14) |
| Uso indebido de los resultados para sancionar, excluir o etiquetar | **Mitigado** | Usos excluidos repetidos en el README, el informe y la demo; regla D2; aviso permanente en la demo (sección 9) |

**Riesgos no previstos que se materializaron:**

- **Reescritura del historial de `main`** (2026-10-19). Se reescribió el historial de `main` para quitar de 4 commits del #13 la línea de coautoría de Claude, que se había añadido sin pedirlo. Los archivos no cambiaron: se verificó que el árbol de cada commit era idéntico. Cambiaron los identificadores de todos los commits, se perdieron las firmas de GitHub y los integrantes realinearon sus copias. El uso de Claude Opus 5.5 en esos commits, y en el resto del proyecto, se declara en la [declaración de uso de IA](../docs/declaracion_ia.md).
- **Colisión de numeración de decisiones.** Los PR #40 y #41 crearon a la vez una "decisión 009". La del #41 se renumeró a 010, pero al resolver el conflicto se perdió la fila de la 009 en el índice de decisiones.
- **Fusiones sin revisión.** Un primer PR del baseline (#31) se fusionó antes de tiempo y se revirtió (#32); la versión aprobada fue la #33. El PR #23 se fusionó sin revisión registrada.
- **Análisis planificados sin ejecutar.** El análisis de sensibilidad y la formulación multiclase (sección 4).

## 13. Limitaciones éticas y técnicas, y trabajo futuro

**Limitaciones técnicas:**

1. **Validez externa.** Una sola institución portuguesa y cohortes 2008/09–2018/19, sin año en los datos: no hay validación temporal, y el desempeño en cohortes futuras no está garantizado.
2. **Señal moderada al ingreso** (AUC ≈ 0.75). Los desertores más jóvenes y con buenas notas no se distinguen de quienes terminan con los datos disponibles.
3. **Optimismo leve.** Hiperparámetros, umbral y zona de abstención se eligieron con los mismos folds con que se reportan. El test confirma que el sesgo es pequeño en el global, pero los resultados por subgrupo **no se verificaron en test**.
4. **Revisión de la entrada univariante.** Detecta valores fuera de rango variable a variable, no perfiles atípicos en conjunto.
5. **Datos preprocesados por los autores** sin un procedimiento auditable, y deserción definida en sentido amplio (incluye cambios de carrera).
6. **Criterio de desempate imperfecto.** El orden de "simplicidad" de la decisión 008 no reflejó el costo real (sección 7).

**Limitaciones éticas:**

1. **Desigualdad en el error costoso.** El modelo detecta peor a las desertoras, a los menores de 21, a los becarios y a los desplazados. Usado sin las reglas E1 y E3, reforzaría que esos grupos reciban menos apoyo.
2. **Etiquetas de grupo, no individuales.** El modelo marca al 99 % de los mayores de 25 años y a casi todos los de horario nocturno.
3. **Variables sensibles y sus proxies.** Género, edad, estado civil y nacionalidad entran en el modelo. Quitarlas reduce poco las brechas, porque otras variables las sustituyen.
4. **Subgrupos sin evaluar.** Estudiantes internacionales y con necesidades educativas especiales: requieren siempre revisión humana.
5. **Correlación, no causa.** El modelo no explica por qué un estudiante abandona ni demuestra que el acompañamiento lo evite.

**Trabajo futuro:**

1. **Variante sin `Gender`**, evaluada con datos nuevos no vistos: mismo recall y la mitad de la brecha de género.
2. **Segundo momento de intervención tras el primer semestre**, con las variables de rendimiento, para captar a los desertores que al ingreso "parecen buenos estudiantes".
3. **Análisis de sensibilidad** con las variables financieras y macroeconómicas, y la **formulación multiclase**, previstos en las decisiones 001 y 002.
4. **Validación local y temporal** con cohortes recientes de la institución usuaria, y seguimiento del recall por cohorte (regla D3).
5. **Detección de perfiles atípicos multivariantes** (por ejemplo, distancia al vecino más cercano en train) en la demo.
6. **Verificar en test o en datos nuevos los resultados por subgrupo** y revisar si la zona de abstención debería variar por grupo.

## 14. Contribución de cada integrante

La tabla se construyó con la evidencia de GitHub (PR fusionados en `main`, aprobaciones registradas y commits) y la validan los tres integrantes en la revisión de este informe. El reparto real difiere del inicial de [proyecto.md](../docs/proyecto.md), sección 2: varios issues cambiaron de responsable durante el proyecto, como muestran sus asignaciones en GitHub.

| Integrante | Usuario | PR propios fusionados (issue) | Revisiones aprobadas de PR ajenos |
|---|---|---|---|
| Roberto Byas | [RByas](https://github.com/RByas) | [#21](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/21) registro del equipo, decisor y pregunta ([#1](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/1)) · [#27](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/27) auditoría de datos ([#6](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/6)) · [#28](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/28) pruebas 1–3 ([#10](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/10)) · [#40](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/40) errores, subgrupos y abstención ([#14](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/14)) | #20, #22, #24, #26, #33, #34, #41, #43 |
| Manicaotex Mueses | [mannixmueses-cloud](https://github.com/mannixmueses-cloud) | [#20](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/20) estructura y entorno ([#2](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/2)) · [#33](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/33) baseline ([#9](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/9)) · [#36](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/36) aprobación de la decisión 007 ([#12](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/12)) · [#37](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/37) y [#39](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/39) comparación y selección ([#13](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/13)) · [#42](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/42) Green AI ([#15](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/15)) | #21, #29, #35 |
| Jose M. Maduro | [mudstart](https://github.com/mudstart) | [#22](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/22) target ([#5](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/5)) · [#23](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/23) dataset y ficha ([#3](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/3)) · [#24](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/24) momento de intervención ([#4](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/4)) · [#26](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/26) partición ([#7](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/7)) · [#29](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/29) pipeline sin fuga ([#8](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/8)) · [#34](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/34) Random Forest ([#11](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/11)) · [#35](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/35) Gradient Boosting ([#12](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/12)) · [#41](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/41) pruebas 4–8 y CI ([#16](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/16)) · [#43](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/43) demo ([#17](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/17)) · este informe, README y declaración de IA ([#18](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/18)) | #27, #28, #37, #40, #42 |

Además, Jose M. Maduro, como propietario del repositorio, creó los issues y fusionó los PR aprobados. Queda pendiente el [#19](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/19) (ensayo de la demo, versión final etiquetada y defensa). Commits sin fusiones en `main` a la fecha: Jose M. Maduro 14, Manicaotex Mueses 13 y Roberto Byas 4. El número de commits no mide el esfuerzo: un PR puede reunir mucho trabajo en pocos commits.

## Referencias

**Datos (cita obligatoria por la licencia CC BY 4.0):**

- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146
- Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: A case study. En *Trends and Applications in Information Systems and Technologies* (AISC, Vol. 1365, pp. 166–175). Springer. https://doi.org/10.1007/978-3-030-72657-7_16

**Métodos:**

- Bengio, Y., & Grandvalet, Y. (2004). No unbiased estimator of the variance of K-fold cross-validation. *Journal of Machine Learning Research, 5*, 1089–1105.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305.

**Herramientas:** Python 3.12, scikit-learn, pandas, NumPy, Matplotlib, codecarbon, Streamlit, pytest, uv y GitHub Actions. Versiones exactas en `uv.lock`. El uso de herramientas de IA se declara en [docs/declaracion_ia.md](../docs/declaracion_ia.md).

**Documentos del proyecto:** [proyecto.md](../docs/proyecto.md) · [ficha de datos](../docs/ficha_datos.md) · [registro de decisiones](../docs/decisiones/README.md) · informes parciales: [auditoría](auditoria_datos.md), [baseline](baseline.md), [comparación](comparacion_modelos.md), [errores y subgrupos](analisis_errores.md), [Green AI](green_ai.md).
