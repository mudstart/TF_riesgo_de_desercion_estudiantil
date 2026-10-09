# Proyecto: riesgo de deserción estudiantil

## 1. Identificación

- Asignatura: INF-8239 Ciencia de Datos II.
- Modalidad: A — Proyecto aplicado de ciencia de datos.
- Tema: 01 — Riesgo de deserción estudiantil.
- Repositorio: https://github.com/mudstart/riesgo_de_desercion_estudiantil
- Rama de esta tarea: `feature/1-proyecto`.
- Responsable de esta tarea: Roberto Byas.
- Estado: documento revisado y aprobado por los tres integrantes (2026-10-07).

## 2. Equipo y distribución acordada

El equipo y la distribución de tareas ya fueron definidos. Esta tabla registra esa distribución; no introduce una nueva asignación.

| Integrante | Usuario de GitHub | Rol o responsabilidad acordada |
|---|---|---|
| Roberto Byas | RByas | Documentación inicial del proyecto: equipo, modalidad, tema, decisor, datos, riesgos y seguimiento de los requisitos de esta tarea (#1); partición estratificada y validación cruzada (#7); pruebas de contrato de datos y anti-fuga, 1–3 de 8 (#10); evaluación comparativa final de modelos (#13); pruebas de pipeline y modelos, 4–8 de 8 (#16) |
| Mannix Mueses | mannixmueses-cloud | Estructura del repositorio y entorno reproducible (#2); adquisición del dataset y verificación de licencia, versión y limitaciones (#3); pipeline de preprocesamiento sin fuga (#8); modelo candidato 1: Random Forest (#11); análisis de errores y de subgrupos (#14); artefacto práctico: demo de predicción (#17) |
| Jose Maduro | mudstart | Momento de intervención y variables permitidas/prohibidas (#4); formulación del target (#5); auditoría de datos, EDA (#6); modelo baseline: regresión logística (#9); modelo candidato 2: Gradient Boosting (#12); medición Green AI (#15); informe técnico, README y declaración de uso de IA (#18); ensayo de demo, versión final y defensa (#19) |

## 3. Registro del equipo y alcance fijo

El equipo está conformado por tres integrantes. El alcance mínimo establecido para este tamaño de equipo es:

- Un baseline y dos modelos candidatos.
- Ocho pruebas como mínimo.
- Documentación del proyecto.
- Control de versiones con evidencia individual de participación.
- Demostración integrada.

El alcance se fija al registrar el equipo ante el profesor y no se reduce al final. Este documento registra el compromiso; no afirma que los modelos o las pruebas ya estén terminados.

El mensaje de registro para el profesor y su constancia de envío están en [registro_equipo.md](registro_equipo.md).

| Evidencia de registro | Información |
|---|---|
| Medio indicado por el profesor | POR COMPLETAR |
| Fecha de registro | POR COMPLETAR |
| Constancia o referencia del registro | POR COMPLETAR |
| Alcance incluido en el registro | Sí: el mensaje declara el alcance completo |
| Aprobación del profesor | Sí: equipo aprobado |

## 4. Decisor y problema de análisis

El decisor es la coordinación académica. El proyecto busca aportar evidencia sobre la identificación de estudiantes con riesgo de deserción, de modo que esa coordinación pueda valorar la priorización de acompañamiento académico temprano.

La Modalidad A produce un análisis reproducible. No se presenta como un sistema de soporte a decisiones (DSS) ni como una herramienta validada para tomar decisiones reales sobre estudiantes.

### Delimitación temporal aprobada por el equipo

El riesgo se identifica al ingreso, una vez disponible la información de matrícula y antes de conocer resultados académicos del primer semestre. Las variables permitidas se derivan de esta delimitación (sección 6) y se formalizan en el issue [#4](https://github.com/mudstart/riesgo_de_desercion_estudiantil/issues/4).

### Pregunta de análisis

¿Con qué nivel de recall de la clase Dropout, acompañado de su precisión, pueden identificarse estudiantes con riesgo de deserción en el conjunto de datos UCI utilizando exclusivamente variables cuya disponibilidad al ingreso esté documentada, para orientar la priorización de acompañamiento por la coordinación académica?

El recall de Dropout representa la proporción de estudiantes que abandonaron y fueron identificados por el modelo como casos de riesgo. Se acompañará de precisión y análisis de errores para no valorar favorablemente un modelo que clasifique a todos como riesgo. No se fija un resultado numérico antes de realizar la evaluación.

## 5. Decisión apoyada y usos excluidos

### Decisión apoyada

Valorar la priorización de contacto, tutorías y acompañamiento académico temprano, con revisión humana y considerando los errores y las limitaciones del análisis.

### Decisiones explícitamente no apoyadas

- Sancionar estudiantes.
- Excluir estudiantes o restringir su continuidad académica.
- Aceptar o rechazar solicitudes de admisión.
- Retirar becas o limitar oportunidades educativas.
- Etiquetar a una persona de forma definitiva como desertora.
- Automatizar decisiones individuales sin revisión humana.

La conclusión final explicará al decisor el desempeño obtenido, los errores más relevantes y las condiciones necesarias antes de considerar una aplicación local. El análisis no demuestra que el acompañamiento cause una reducción de la deserción.

## 6. Datos del proyecto

| Elemento | Descripción |
|---|---|
| Nombre | Predict Students' Dropout and Academic Success |
| Fuente | UCI Machine Learning Repository, conjunto 697 |
| Contexto | Educación superior en Portugal: Instituto Politécnico de Portalegre, varias titulaciones de pregrado |
| Unidad de análisis | Estudiante |
| Archivo suministrado | `data.csv`, dentro del ZIP adjunto |
| Tamaño verificado | 4,424 filas y 37 columnas: 36 características y Target |
| Variable objetivo | `Target` |
| Clases | Dropout: 1,421; Enrolled: 794; Graduate: 2,209 |
| Valores faltantes | Ninguno según la ficha; los autores preprocesaron anomalías y valores atípicos |
| Codificación | Variables categóricas codificadas como enteros; los diccionarios de códigos están en la ficha |
| Licencia | CC BY 4.0, con atribución |
| DOI | https://doi.org/10.24432/C5MC89 |
| Ficha | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |

Se propone conservar las tres clases originales. Enrolled no equivale a graduación ni garantiza que el estudiante no abandonará posteriormente.

Para el escenario al ingreso, se propone excluir las variables de resultados del primer y segundo semestre. Las variables de deuda, pagos, becas o condiciones económicas requieren verificar su fecha de referencia; no se utilizarán si no se acredita su disponibilidad en el momento definido. Esta regla deberá quedar alineada con el trabajo del integrante responsable de datos.

### Inventario preliminar de variables según su disponibilidad temporal

La ficha indica que el conjunto incluye información conocida al momento de la matrícula, junto con el rendimiento académico al final del primer y del segundo semestre. La agrupación de esta tabla es una propuesta del equipo: la ficha no clasifica las variables así. El integrante responsable de datos la validará.

| Grupo | Variables | Disponibilidad |
|---|---|---|
| Demográficas | Marital status, Nacionality (así en el original), Gender, Age at enrollment, International, Displaced, Educational special needs | Al ingreso |
| Acceso y trayectoria previa | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Admission grade | Al ingreso |
| Contexto familiar | Mother's qualification, Father's qualification, Mother's occupation, Father's occupation | Al ingreso |
| Situación financiera | Debtor, Tuition fees up to date, Scholarship holder | Por verificar: la ficha no indica su fecha de referencia |
| Macroeconómicas | Unemployment rate, Inflation rate, GDP | Por verificar: probablemente asociadas al año de ingreso |
| Rendimiento del primer semestre | Curricular units 1st sem: credited, enrolled, evaluations, approved, grade, without evaluations | Al final del primer semestre |
| Rendimiento del segundo semestre | Curricular units 2nd sem: credited, enrolled, evaluations, approved, grade, without evaluations | Al final del segundo semestre |

Con la delimitación al ingreso, quedan excluidas las 12 variables de rendimiento semestral. Las 3 de situación financiera solo se incorporarán si se acredita su disponibilidad. Como análisis de sensibilidad, se podría comparar el modelo con y sin ellas, siempre que la versión principal sea la que las excluye.

## 7. Registro inicial de riesgos

Las valoraciones son cualitativas. Cada responsable corresponde al integrante asignado en GitHub al issue relacionado con el riesgo.

| Riesgo | Probabilidad | Impacto | Mitigación | Responsable según reparto vigente |
|---|---|---|---|---|
| Fuga de información por variables posteriores a la intervención | Alta | Alto | Fijar el momento de intervención; documentar disponibilidad temporal; excluir información futura y variables de temporalidad no verificable | Jose Maduro (#4 anti-fuga, #6 EDA) |
| Desbalance de clases y recall insuficiente en Dropout | Media | Alto | Planificar partición estratificada y evaluación por clase; analizar recall junto con precisión y matriz de confusión | Roberto Byas (#7 partición, #13 evaluación comparativa) |
| Sesgo contra subgrupos por género, edad o nacionalidad | Media | Alto | Comparar errores por subgrupos, reportar tamaños y limitaciones; evitar usos discriminatorios | Mannix Mueses (#14 análisis de errores y subgrupos) |
| Baja generalización por provenir de una institución y contexto de Portugal | Alta | Alto | Limitar conclusiones al contexto estudiado y exigir validación local antes de aplicación en otra institución | Jose Maduro (#18 informe técnico: limitaciones y conclusiones) |
| Atraso del cronograma, especialmente si S3 concentra tres semanas | Media | Alto | Confirmar calendario y significado de S3; establecer fechas internas y revisar dependencias semanalmente | Roberto Byas (#1 seguimiento de requisitos) |
| Integrante sin evidencia verificable en Git | Media | Alto | Dar seguimiento a issues, commits sustantivos, PR y revisiones desde cuentas individuales; no confundir asignación con contribución | Cada integrante; seguimiento de esta tarea: Roberto Byas |
| Uso indebido de los resultados para sancionar, excluir o etiquetar estudiantes | Baja | Alto | Repetir los usos excluidos de la sección 5 en el README, el informe y la demostración; presentar los resultados como apoyo a la revisión humana, no como decisión | Jose Maduro (#18 informe técnico, #19 demo final) |

La compresión de tres semanas en S3 se registra como condición por verificar: las orientaciones generales adjuntas presentan seis semanas y no explican esa compresión.

## 8. Issues iniciales y acceso

Se conservarán los issues y responsables ya acordados. No se crearán duplicados ni se reasignarán funciones mediante este documento.

| Integrante | Issue o issues asignados: enlace real | Acceso efectivo confirmado |
|---|---|---|
| Roberto Byas (`RByas`) | [#1](https://github.com/mudstart/riesgo_de_desercion_estudiantil/issues/1) Registro del equipo, decisor, pregunta de análisis y registro de riesgos | Sí: colaborador con invitación aceptada |
| Mannix Mueses (`mannixmueses-cloud`) | [#2](https://github.com/mudstart/riesgo_de_desercion_estudiantil/issues/2) Estructura del repositorio y entorno reproducible; [#3](https://github.com/mudstart/riesgo_de_desercion_estudiantil/issues/3) Adquisición del dataset UCI y verificación de licencia, versión y limitaciones | Sí: colaborador con invitación aceptada |
| Jose Maduro (`mudstart`) | [#4](https://github.com/mudstart/riesgo_de_desercion_estudiantil/issues/4) Definir el momento de intervención y variables permitidas/prohibidas (anti-fuga); [#5](https://github.com/mudstart/riesgo_de_desercion_estudiantil/issues/5) Decidir formulación del target: 3 clases o binario | Sí: propietario del repositorio |

Cuando falte un acceso, el propietario o administrador del repositorio enviará la invitación correspondiente y verificará su aceptación. Una invitación pendiente no equivale a acceso confirmado.

## 9. Revisión y criterios de aceptación

| Integrante | Fecha de revisión | Evidencia: comentario, revisión o PR |
|---|---|---|
| Roberto Byas | 2026-10-07 | Aprobación del documento por el equipo |
| Mannix Mueses | 2026-10-07 | Aprobación del documento por el equipo |
| Jose Maduro | 2026-10-07 | Aprobación del documento por el equipo |

- [x] Equipo registrado ante el profesor por el medio indicado.
- [x] Alcance fijo declarado y confirmado en el registro.
- [x] Integrantes, usuarios y roles acordados consignados.
- [x] Decisor, pregunta y momento de intervención revisados.
- [x] Datos y registro de riesgos documentados, con responsables identificados.
- [x] `docs/proyecto.md` revisado por los tres integrantes.
- [x] Los tres integrantes tienen acceso efectivo al repositorio.
- [x] Los tres integrantes tienen issues asignados según el reparto vigente.

Solo se marcarán criterios respaldados por evidencia. La conformación del grupo no sustituye el registro ante el profesor.

## 10. Fuentes

Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: a case study. En *Trends and Applications in Information Systems and Technologies* (Advances in Intelligent Systems and Computing). Springer. https://doi.org/10.1007/978-3-030-72657-7_16

Nolasco, E. R. J. (s. f.). Trabajo final · Orientaciones (TF.00). INF-8239 Ciencia de Datos II. UASDVirtual. Material docente suministrado.

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
