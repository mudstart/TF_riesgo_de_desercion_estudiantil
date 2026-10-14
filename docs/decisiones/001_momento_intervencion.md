# Decisión 001: momento de intervención y variables permitidas

- Issue: [#4](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/4) Definir el momento de intervención y variables permitidas/prohibidas (anti-fuga).
- Rama: `feature/4-momento-intervencion`.
- Responsable: Jose Maduro.
- Fecha: 2026-10-13.
- Estado: Aprobada (PR #24).

## 1. Contexto

El enunciado exige clasificar el riesgo temprano "sin utilizar información posterior al momento de intervención". El dataset UCI 697 mezcla información de matrícula con rendimiento del primer y segundo semestre, y el `Target` se mide al final de la duración normal del curso (3 años; 4 en Enfermería). Usar una variable que todavía no existe en el momento de decidir es **fuga de información**: el modelo parece mejor de lo que sería en la práctica.

La pregunta de análisis aprobada ([proyecto.md](../proyecto.md), sección 4) ya limita el estudio a "variables cuya disponibilidad al ingreso esté documentada". Esta decisión concreta qué variables cumplen esa condición.

## 2. Momento de intervención

**Al ingreso**: una vez disponible la información de matrícula y antes de conocer cualquier resultado académico del primer semestre.

Se modela **un solo escenario**. Se descartó un segundo escenario "tras el primer semestre" porque:

1. Contradice la delimitación y la pregunta de análisis ya aprobadas.
2. Con la formulación multiclase secundaria (decisión [002](002_formulacion_target.md)), un segundo escenario duplicaría otra vez el entrenamiento y la evaluación de los tres modelos.
3. Intervenir al ingreso es lo más temprano posible, que es lo que pide el tema.

## 3. Clasificación de las 36 variables

| Grupo | Variables | N | Disponibilidad | Uso |
|---|---|---|---|---|
| Ingreso | Marital Status, Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Nacionality, Mother's qualification, Father's qualification, Mother's occupation, Father's occupation, Admission grade, Displaced, Educational special needs, Gender, Age at enrollment, International | 18 | Al ingreso | **Modelo principal** |
| Sensibilidad: financieras | Debtor, Tuition fees up to date, Scholarship holder | 3 | No documentada | Solo análisis de sensibilidad |
| Sensibilidad: macroeconómicas | Unemployment rate, Inflation rate, GDP | 3 | No documentada | Solo análisis de sensibilidad |
| 1er semestre | Curricular units 1st sem (credited, enrolled, evaluations, approved, grade, without evaluations) | 6 | Fin del 1er semestre | **Prohibidas** |
| 2do semestre | Curricular units 2nd sem (credited, enrolled, evaluations, approved, grade, without evaluations) | 6 | Fin del 2do semestre | **Prohibidas** |
| | | **36** | | |

Las listas están en `src/config.py` (`FEATURES_INGRESO`, `FEATURES_SENSIBILIDAD`, `FEATURES_SEM1`, `FEATURES_SEM2` y `FEATURES_PROHIBIDAS`). El pipeline debe obtener las columnas con `select_features()` de `src/features.py`, nunca escribiéndolas a mano.

## 4. Justificación

**Variables de semestre: prohibidas.** Son resultados académicos posteriores al ingreso; usarlas es fuga por definición.

**Variables financieras: fuera del modelo principal.** Ninguna fuente consultada documenta cuándo se registran (ver [ficha_datos.md](../ficha_datos.md), sección 5). "Matrícula al día" y "deudor" describen estados de pago que cambian durante la carrera, y su relación con el resultado es sospechosamente fuerte: `Tuition fees up to date` = 0 corresponde a 87 % de Dropout y `Debtor` = 1 a 62 %, frente a 32 % en el total. Si se registraron al extraer los datos y no al ingreso, son fuga.

**Variables macroeconómicas: fuera del modelo principal.** El año al que corresponden no está documentado. Probablemente son del año de ingreso, pero la pregunta aprobada exige disponibilidad documentada. Además, con solo 10 combinaciones distintas, funcionan como un identificador de cohorte más que como información individual.

**Supuesto explícito sobre las 18 variables de ingreso.** Los datos de los padres y `Educational special needs` provienen del mismo registro académico que las financieras y tampoco tienen fecha documentada. Se consideran disponibles al ingreso porque son, por naturaleza, datos del formulario de matrícula o del concurso de acceso, y no estados que cambien durante la carrera. Si se demostrara lo contrario, deberían pasar al grupo de sensibilidad.

## 5. Evidencia cuantitativa

Regresión logística con escalado estándar, validación cruzada estratificada de 5 folds (`RANDOM_STATE` = 42), formulación binaria Dropout vs. resto, umbral 0.5. Medias de los 5 folds. Es un análisis exploratorio para esta decisión, no el baseline del #9.

| Variables | N | F1-macro | Recall Dropout | Precisión Dropout |
|---|---|---|---|---|
| **Ingreso (principal)** | 18 | **0.586** | **0.262** | 0.600 |
| Ingreso + macroeconómicas | 21 | 0.585 | 0.260 | 0.596 |
| Ingreso + financieras | 21 | 0.692 | 0.426 | 0.742 |
| Ingreso + sensibilidad (las 6) | 24 | 0.694 | 0.425 | 0.752 |
| + 1er semestre | 30 | 0.827 | 0.681 | 0.845 |
| Todas | 36 | 0.851 | 0.723 | 0.871 |

Con pesos de clase balanceados (`class_weight="balanced"`), el modelo principal sube a F1-macro 0.639 y recall 0.607, con precisión 0.485.

Lectura:

1. **El costo de evitar la fuga es real.** Con solo información de ingreso, el modelo identifica muchos menos estudiantes en riesgo que con todas las variables. Un resultado de F1-macro ≈ 0.85 sobre este dataset casi seguro incluye información futura.
2. **Las variables macroeconómicas no aportan.** Excluirlas no tiene costo.
3. **Las financieras explican todo el salto de la sensibilidad** (+0.10 de F1-macro). Esa diferencia es justamente lo que estaría en juego si fueran fuga; el análisis de sensibilidad debe reportarla.
4. **El umbral importa.** Con umbral 0.5 el recall de Dropout es bajo; equilibrar las clases o ajustar el umbral lo sube a costa de precisión. Esa decisión corresponde a la evaluación (#13), considerando que el uso es priorizar acompañamiento, no sancionar.

## 6. Consecuencias

- Los modelos del baseline y candidatos (#9, #11, #12) usan solo `FEATURES_INGRESO`.
- El análisis de sensibilidad (con `FEATURES_SENSIBILIDAD`) se reporta aparte y nunca sustituye al modelo principal.
- El pipeline (#8) selecciona columnas con `select_features()`.
- Las pruebas anti-fuga (#10) deben comprobar que ninguna variable de `FEATURES_PROHIBIDAS` llega al modelo.
- El informe (#18) debe explicar que el desempeño al ingreso es menor que el publicado para este dataset, y por qué.

## 7. Condición para detener o escalar la recomendación

Relacionada con la pregunta de defensa. La recomendación de priorizar acompañamiento debe detenerse o escalarse a revisión si:

- Se detecta que alguna variable usada no estaba disponible al ingreso.
- El recall de Dropout del modelo principal no supera claramente al de una regla trivial, de modo que el modelo no aporta evidencia.
- Se quiere aplicar en otra institución o cohorte sin validación local.

## 8. Aprobación

| Campo | Valor |
|---|---|
| Revisado por | POR COMPLETAR: revisión del PR |
| Fecha | POR COMPLETAR |
