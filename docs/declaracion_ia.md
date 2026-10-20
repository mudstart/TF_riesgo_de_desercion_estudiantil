# Declaración de uso de herramientas de IA

Trabajo final de INF-8239 Ciencia de Datos II · Tema 01, riesgo de deserción estudiantil (#18).

Esta declaración recoge qué herramientas de inteligencia artificial usó cada integrante, para qué tareas, qué parte generó la herramienta y qué parte hizo o decidió la persona, y cómo se verificó el resultado. Los tres integrantes usaron el modelo **Claude Opus 5.5**, de Anthropic, a través de Claude Code, su asistente de programación. En las secciones se nombra solo el modelo. Cada integrante redactó y aprobó su propia sección a partir de la evidencia disponible: sus commits y PR en GitHub y los historiales de sus sesiones.

## Criterios comunes del equipo

- **La responsabilidad es de las personas.** Las decisiones del proyecto (momento de intervención, variables, métricas, criterio de selección, reglas de abstención) se discutieron y aprobaron en el equipo y quedaron documentadas en `docs/decisiones/`, con un responsable y un revisor.
- **Nada entra a `main` sin revisión.** Todo cambio pasa por un PR que revisa otro integrante. Las excepciones están registradas en la sección 12 del informe técnico.
- **Verificación independiente de la herramienta:** pruebas automáticas (110, ejecutadas en cada PR con GitHub Actions), reproducción de los scripts de análisis y contraste de las cifras de los informes con los archivos de métricas versionados.

## Jose M. Maduro (mudstart)

Sección elaborada a partir de sus commits y PR y de los historiales de sus sesiones con Claude Opus 5.5: 2026-10-05 a 07 (planificación), 2026-10-10 a 18 (issues de datos y modelos) y 2026-10-19 a 20 (pruebas, demo e informe). Todas usaron Claude Opus 5.5. No usó otras herramientas de IA. Desde el 2026-10-10 trabajó con una regla fija: la herramienta explica qué hacer y cómo, él autoriza cada paso o ejecuta los comandos, y la herramienta no hace commits ni push.

| Herramienta | Tareas e issues | Qué generó la IA | Qué hice o decidí yo | Cómo se verificó |
|---|---|---|---|---|
| Claude Opus 5.5 | Planificación del proyecto (5–7 oct) | Análisis del enunciado, borradores de los issues con sus descripciones, milestones y etiquetas, nombres cortos para las ramas y un plan de trabajo por issue | Elegí la modalidad con el equipo, creé los issues a mano en GitHub y propuse el reparto | Revisión contra el enunciado: por ejemplo, quité la Model Card, que no exige el alcance para 2–3 integrantes |
| Claude Opus 5.5 | #5: formulación del target (PR [#22](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/22)); decisión 002 | Código, pruebas y borrador de la decisión | Fijé la regla de trabajo; autoricé cada paso y ejecuté los comandos | `pytest` en verde; aprobación del PR por Roberto Byas |
| Claude Opus 5.5 | #3: descarga verificada y ficha de datos (PR [#23](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/23)); #4: momento de intervención (PR [#24](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/24)), decisión 001; registro de decisiones y plantilla de PR | Código de descarga con verificación SHA-256, `docs/ficha_datos.md`, borrador de la decisión 001, `docs/decisiones/README.md`, la plantilla de decisión y la plantilla de PR | Hice obligatorio el registro de decisiones para todo el equipo; autoricé cada paso | `pytest` y comprobación del hash contra la descarga; aprobación del PR #24 por Roberto Byas (el #23 se fusionó sin revisión registrada) |
| Claude Opus 5.5 | #7: partición (PR [#26](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/26)), decisión 003; #8: pipeline sin fuga (PR [#29](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/29)), decisión 004; revisión del #6 | Código de la partición y `splits.json`, el pipeline, sus pruebas y los borradores de las decisiones; análisis del notebook de la auditoría | Ejecuté el notebook y los comandos; asumí el #7, que estaba asignado a un compañero enfermo | `pytest`; aprobaciones de los PR #26 (Roberto Byas) y #29 (Manicaotex Mueses); aprobé el PR #27 del #6 |
| Claude Opus 5.5 | #11: Random Forest (PR [#34](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/34)), decisión 006; #12: Gradient Boosting (PR [#35](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/35)), decisión 007 | Código de entrenamiento y búsqueda aleatoria, pruebas, informes de métricas y borradores de las decisiones | Detecté que los comandos de ejecución no estaban en el README y pedí añadirlos | `pytest`; aprobaciones de los PR #34 (Roberto Byas) y #35 (Manicaotex Mueses) |
| Claude Opus 5.5 | Revisión de los PR #37 (#13) y #38 (#14), 18 oct | Revisión del PR #37, reproducción de la comparación y de los resultados de test, y análisis del PR #38 | Aprobé el #37 y pedí a la herramienta reescribir el comentario de revisión en un tono más natural antes de publicarlo; el #38 se descartó por estar mal planteado y se rehízo en el #40 | Pruebas del PR #37 (71 en verde) y comparación de sus cifras |
| Claude Opus 5.5 | Revisión de los PR #39, #40 y #42 | Análisis de cada PR, comprobación de sus cifras y borradores de los comentarios de revisión | Decidí aprobar o pedir cambios; publiqué los comentarios | El #40 se ejecutó fusionado con el #41 (89 pruebas) y su análisis se reprodujo exactamente; las cifras del #42 se contrastaron con `green_ai.json` |
| Claude Opus 5.5 | Realineación de la copia local tras la reescritura del historial (19 oct) | Detectó el cambio de identificadores de los commits y propuso los comandos para realinear la copia local | Ejecuté los comandos (`git reset --hard origin/main`) | Comparación de los árboles de cada commit antes de realinear |
| Claude Opus 5.5 | #16: pruebas 4–8, `ValidarColumnas` y GitHub Actions; decisión 010 | Código de las pruebas, del validador y del workflow; borrador de la decisión | Elegí el diseño (prueba 8 sin test, margen de 0.02, error propio en español); renumeré la decisión a 010; hice los commits y el PR | `pytest`; cada prueba se comprobó **rompiendo a propósito** lo que protege (7 roturas, todas detectadas); CI en verde |
| Claude Opus 5.5 | #17: demo de predicción; decisión 012 | `src/predict.py`, `src/diccionario.py` (etiquetas extraídas de la página de UCI), `app/app.py`, pruebas y borrador de la decisión | Elegí las categorías (la zona de abstención como "medio"), la explicación global + local y la reconstrucción del modelo; hice los commits y el PR | `pytest` (14 pruebas nuevas); los perfiles ficticios se comprobaron fuera del dataset; prueba de la app en el navegador con cada perfil y con una abstención |
| Claude Opus 5.5 | #18: informe técnico, README, aprobación de decisiones y esta declaración | Borradores de las secciones, la tabla de contribuciones (a partir de la API de GitHub) y el prompt para recoger las declaraciones del equipo | Revisé y aprobé cada sección; elegí el formato y el criterio de contribución por evidencia de Git | Cifras contrastadas con los JSON de `reports/metrics/` y con los datos; enlaces revisados |

**Reescritura del historial.** El 2026-10-19 se reescribió el historial de `main` para quitar de 4 commits del #13 la línea de coautoría de Claude, que se había añadido sin pedirlo. Los archivos no cambiaron. El uso de Claude Opus 5.5 en esos commits se declara en este documento (sección de Manicaotex Mueses, issue #13).

**Errores de la IA o usos descartados.** La herramienta no hizo commits ni push: los ejecuté yo. Durante la revisión se corrigieron errores suyos:

- atribuyó una prueba al issue equivocado;
- afirmó que el equipo ya había validado la tabla de contribuciones antes de que ocurriera;
- afirmó que todos los PR tenían revisión, cuando tres no la tienen;
- redactó un texto sobre la zona de abstención que no coincidía con la decisión 009;
- describió la reescritura del historial como si no hubiera cambiado el contenido de ningún commit.

Las pruebas detectaron además un fallo real de `revisar_entrada()` con campos vacíos, que se corrigió.

**Evidencia:** PR [#22](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/22), [#23](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/23), [#24](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/24), [#26](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/26), [#29](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/29), [#34](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/34), [#35](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/35), [#41](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/41), [#43](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/43) y el PR del #18; revisiones de los PR #27, #28, #37, #40 y #42; sesiones con Claude Opus 5.5 del 2026-10-05 al 2026-10-20.

## Roberto Byas (RByas)

Sección elaborada a partir de sus commits en `main` y de los historiales de sus sesiones con Claude Opus 5.5 en este repositorio. Todas las sesiones usaron Claude Opus 5.5. No declaró otras herramientas de IA.

| Herramienta | Tareas e issues | Qué generó la IA | Qué hice o decidí yo | Cómo se verificó |
|---|---|---|---|---|
| Claude Opus 5.5 | #1: registro del equipo, decisor y pregunta de análisis (PR #21, commit `3b80e64`) | Apoyo en la redacción del documento | Contenido acordado por el equipo; commit y PR | Revisión y aprobación del PR por Manicaotex Mueses |
| Claude Opus 5.5 | #6: auditoría de datos (PR #27, commit `09f460d`) | `reports/auditoria_datos.md`, la ejecución del notebook y la descripción del PR | Revisión de los hallazgos y las decisiones derivadas | Notebook y pruebas ejecutados; revisión y aprobación del PR por Jose M. Maduro |
| Claude Opus 5.5 | #10: pruebas 1–3 de contrato de datos, anti-fuga y partición (PR #28, commit `3a41817`) | `tests/test_data.py` | Definición de qué debía proteger cada prueba; commit y PR | `pytest`; revisión y aprobación del PR por Jose M. Maduro |
| Claude Opus 5.5 | #14: análisis de errores, subgrupos y zona de abstención (PR #40, commits `7c41c8c` y `85c4543`) | `src/errores.py`, `tests/test_errores.py`, `reports/analisis_errores.md`, la decisión 009 y ajustes en `src/config.py` | Revisión del análisis y de la decisión 009, de la que es responsable | `pytest` y los scripts de análisis ejecutados varias veces; revisión del PR por Jose M. Maduro, que reprodujo el análisis exactamente |
| Claude Opus 5.5 | Revisión de los PR #20, #22, #24, #26, #29, #33, #34, #41 y #43 | Lectura de los diffs, pasos para probar cada PR y borradores de comentarios y solicitudes de cambios | Decidir si aprobar o pedir cambios; publicar la revisión | `pytest` en la mayoría (por ejemplo, 49 y 53 pruebas en el #33 y 109 en el #43), pruebas con el dataset y su hash (#24) y prueba de la demo con cada perfil (#43) |

**Errores de la IA o usos descartados.**

- En la sesión del #6, Claude hizo el commit y el push. Desde la sesión del #10, Roberto le indicó que no hiciera commits ni push, y desde entonces los hizo él.
- En la revisión del PR #29, Claude no pudo ejecutar las pruebas porque lo impidió el sistema: esa revisión se basó solo en la lectura del diff.
- Una sesión sobre un tema ajeno al proyecto, ubicada en la misma carpeta, se excluyó de esta declaración.

**Evidencia:** commits `3b80e64`, `09f460d`, `3a41817`, `7c41c8c` y `85c4543` en `main`; PR [#21](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/21), [#27](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/27), [#28](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/28) y [#40](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/40); sesiones con Claude Opus 5.5 del 2026-10-09 al 2026-10-20.

## Manicaotex Mueses (mannixmueses-cloud)

| Herramienta | Tareas e issues | Qué generó la IA | Qué hice o decidí yo | Cómo se verificó |
|---|---|---|---|---|
| Claude Opus 5.5 | #2: estructura del proyecto | Revisión del proyecto contra el enunciado y correcciones anotadas en el README | Pidió la revisión, decidió que las correcciones fueran al README y pidió retirar el commit que la IA había subido | `pytest` (3 pruebas en verde) |
| Claude Opus 5.5 | #9: baseline de regresión logística | `training.py` y `train_baseline.py` (después movido a `src/models/`), pruebas, `reports/baseline.md`, decisión 005 y la descripción del PR #33 | Revisó el código y el informe paso a paso antes de cada commit, aprobó cada corrección, decidió rehacer el #9 desde `main` e hizo los commits y el push | `pytest` (se detectó y corrigió un fallo; después, entre 52 y 61 pruebas en verde) y comparación de métricas y coeficientes antes y después del refactor |
| Claude Opus 5.5 | #12: revisión del PR #35 y aprobación de la decisión 007 | Revisión del PR #35, ejecutado en una copia aparte del repositorio | Pidió la revisión antes de aprobar e hizo el commit de aprobación de la decisión 007 | `pytest` sobre la rama del PR (61 pruebas en verde) |
| Claude Opus 5.5 | #13: comparación de modelos | `comparacion.py` y `evaluation.py` (validación cruzada, umbral ajustado y bootstrap pareado), pruebas, decisión 008 y `reports/comparacion_modelos.md`. Detectó el choque de nombres entre `src/models.py` y `src/models/` | Pidió verificar el cumplimiento del issue, aplicó los ajustes de la revisión del PR #37 y decidió que la IA no figurara como colaboradora en los commits | `pytest` (hasta 71 pruebas en verde), evaluación única en test (`comparacion --test`) y revisión de los PR #37 y #39 |
| Claude Opus 5.5 | #15: medición Green AI | `green_ai.py` con codecarbon, pruebas, `reports/green_ai.md`, decisión 011 y ajustes del README | Puso como condición que la IA no subiera nada, ejecutó él los comandos y revisó el PR #42 contra el issue | `pytest` (las 6 pruebas de Green AI y las 95 del proyecto en verde) |

**Errores de la IA o usos descartados.**

- **#2:** Claude Opus 5.5 hizo commit y push en `feature/2-estructura` sin que se lo pidiera. Se deshizo y la rama volvió a su commit anterior.
- **#13:** Claude Opus 5.5 hizo commit y push con la línea "Co-Authored-By: Claude", aunque solo se le habían pedido los comandos. Se pidió quitarlo y se corrigió en el PR #39.
- **#9:** hubo una confusión con el PR #31, que se fusionó y luego se revirtió. El trabajo se rehízo desde `main` en el PR #33.
- **Uso descartado:** a raíz de esos incidentes, decidió que la IA no hiciera commits ni push. Desde entonces los hace él con los comandos que le propone la herramienta.

**Evidencia.**

- Commits en `main`: `f2e134d` y `4ea854f` (#2); `a252c18`, `97fda15` y `90baec1` (#9); `138cacd` (#12); `4363443`, `88362a5`, `b88881e`, `a5ff514`, `08f5124` y `e3cd0f6` (#13); `85f327c` (#15).
- Copias anteriores a la reescritura del historial de `main` del 2026-10-19 (mismos archivos y mensaje, otro hash): `64ccce0`, `be1d00f`, `34dce03`, `0cbf9fd` y `aa0ac9c`. `6dfd2ee` y `a101b1c` son el primer borrador del #13 (2026-10-14), en una rama que nunca se fusionó ni evaluó en test (decisión 008). La fusión del PR #31 (`9b63fb0`) y su reversión (`2aacfec`) no forman parte del historial actual de `main`. `fcec161` y `6b27614` solo existen en su copia local: según el equipo, pertenecen a un PR que se descartó por contener un error y nunca llegó al repositorio compartido.
- PR [#20](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/20), [#31](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/31), [#33](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/33), [#36](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/36), [#37](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/37), [#39](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/39) y [#42](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/pull/42).
- Sesiones con Claude Opus 5.5: 2026-10-10, 10-11, 10-14, 10-15, 10-16 (dos), 10-18 y 10-19 (dos, una interrumpida).
