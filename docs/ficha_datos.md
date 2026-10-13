# Ficha de datos: Predict Students' Dropout and Academic Success

- Issue: [#3](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/3) Adquisición del dataset UCI y verificación de licencia, versión y limitaciones.
- Rama: `feature/3-dataset-uci`.
- Fecha de verificación: 2026-10-12.

## 1. Qué es

Conjunto de datos tabular con un registro por estudiante de grado de una institución de educación superior portuguesa. Reúne información demográfica, socioeconómica, macroeconómica y académica al ingreso, junto con el rendimiento académico al final del primer y del segundo semestre. Los autores lo construyeron para predecir deserción y éxito académico dentro de una herramienta de *learning analytics* que orienta al equipo de tutorías (Realinho et al., 2022, sec. 1).

En este proyecto se usa para identificar estudiantes con riesgo de deserción y apoyar la priorización de acompañamiento temprano (ver [proyecto.md](proyecto.md)).

## 2. Procedencia y versión

| Campo | Valor |
|---|---|
| Página | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| Descarga directa | https://archive.ics.uci.edu/static/public/697/data.csv |
| Formato | CSV separado por comas, UTF-8 con BOM (leer con `encoding="utf-8-sig"`) |
| DOI del dataset (UCI) | [10.24432/C5MC89](https://doi.org/10.24432/C5MC89) |
| DOI del dataset (Zenodo, versión de los autores) | [10.5281/zenodo.5777339](https://doi.org/10.5281/zenodo.5777339) |
| Licencia | CC BY 4.0: uso libre con atribución obligatoria |
| Creadores | Valentim Realinho, Mónica Vieira Martins, Jorge Machado, Luís Baptista |
| Donación a UCI / última actualización | 2021-12-12 / 2024-02-26 |
| Financiamiento | Programa SATDAP, Capacitação da Administração Pública, POCI-05-5762-FSE-000191 |
| Registros × columnas | 4,424 × 37: 36 características y `Target` |
| Nulos / duplicados | 0 / 0 (los autores preprocesaron anomalías, valores atípicos y faltantes) |
| `Target` | Graduate 2,209 (49.9 %) · Dropout 1,421 (32.1 %) · Enrolled 794 (17.9 %) |
| SHA-256 de `data.csv` | `a1b1a6531bbb93a5c7fdf0093b47172776652a4ffb12342fb79655c85b74801b` |

Todos los valores de la tabla se verificaron el 2026-10-12 contra el archivo descargado y la API de UCI. El SHA-256 está fijado en `src/config.py`: si UCI publica otra versión, la descarga falla y el cambio debe documentarse aquí antes de usarla.

## 3. Cobertura

| Aspecto | Valor | Fuente |
|---|---|---|
| Institución | Instituto Politécnico de Portalegre, Portugal | Realinho et al. (2022), sec. 4 |
| Periodo | Estudiantes matriculados entre los años académicos 2008/2009 y 2018/2019 | Realinho et al. (2022), sec. 2 |
| Nivel | 17 grados de licenciatura (pregrado) | Realinho et al. (2022), sec. 2 |
| Horizonte del `Target` | Situación al final de la duración normal del curso: 3 años, excepto Enfermería con 4 | Realinho et al. (2022), sec. 3.1 |

Cursos, con el código usado en el CSV de UCI:

| Código | Curso | Código | Curso |
|---|---|---|---|
| 33 | Biofuel Production Technologies | 9238 | Social Service |
| 171 | Animation and Multimedia Design | 9254 | Tourism |
| 8014 | Social Service (evening attendance) | 9500 | Nursing |
| 9003 | Agronomy | 9556 | Oral Hygiene |
| 9070 | Communication Design | 9670 | Advertising and Marketing Management |
| 9085 | Veterinary Nursing | 9773 | Journalism and Communication |
| 9119 | Informatics Engineering | 9853 | Basic Education |
| 9130 | Equinculture | 9991 | Management (evening attendance) |
| 9147 | Management | | |

## 4. Construcción del dataset

Según Realinho et al. (2022, sec. 2 y 3.1), el dataset combina cuatro fuentes:

1. **DGES / CNAES** (concurso nacional de acceso): datos de acceso a la educación superior, recibidos cada año tras el concurso.
2. **Sistema de gestión académica (AMS) de la institución**: registros de estudiantes, de donde salen los datos demográficos y socioeconómicos.
3. **AMS y plataforma interna PAE**: evaluaciones, de donde salen los datos del primer y segundo semestre.
4. **PORDATA**: indicadores macroeconómicos, añadidos manualmente al combinar los datos.

Después de combinar las fuentes, los autores trataron anomalías, valores atípicos y faltantes, y clasificaron a cada estudiante como Dropout, Enrolled o Graduate según su situación al final de la duración normal del curso. Los datos están anonimizados y su publicación fue validada por el delegado de protección de datos de la institución conforme al RGPD (sec. 3.3 y declaración institucional).

## 5. Disponibilidad temporal de las variables

Insumo para el issue [#4](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/4) (momento de intervención y variables permitidas).

| Grupo (según los autores) | Variables | Momento documentado |
|---|---|---|
| Demográficas | Marital Status, Nacionality, Displaced, Gender, Age at enrollment, International | Al ingreso |
| Académicas al ingreso | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Admission grade | Al ingreso |
| Socioeconómicas: contexto familiar y necesidades | Mother's/Father's qualification, Mother's/Father's occupation, Educational special needs | Al ingreso |
| Socioeconómicas: situación financiera | Debtor, Tuition fees up to date, Scholarship holder | **No documentado** |
| Macroeconómicas | Unemployment rate, Inflation rate, GDP | **No documentado** |
| Fin del 1er semestre | Curricular units 1st sem (6 variables) | Al final del 1er semestre |
| Fin del 2do semestre | Curricular units 2nd sem (6 variables) | Al final del 2do semestre |

**Variables financieras.** Ninguna fuente consultada (artículo de datos, ficha de UCI, API de UCI, Zenodo ni la página de variables de los autores) indica la fecha de referencia de `Debtor`, `Tuition fees up to date` y `Scholarship holder`. Los autores las agrupan como "socioeconómicas", las extraen de los registros del AMS y afirman en términos generales que el dataset contiene información conocida al momento de la matrícula, pero no lo precisan para estas variables. Además:

- "Matrícula al día" describe por naturaleza un estado de pago durante la carrera, no un dato previo al ingreso.
- `Tuition fees up to date` = 0 corresponde a 87 % de Dropout y `Debtor` = 1 a 62 % de Dropout (frente a 32 % en el total).
- En el análisis de los propios autores, `Tuition fees up to date` aparece entre las cinco variables más importantes de los cuatro modelos, junto con el rendimiento del segundo semestre (Realinho et al., 2022, sec. 3.2.4).

Por eso no puede acreditarse su disponibilidad al ingreso. La decisión sobre su uso corresponde al #4.

**Variables macroeconómicas.** El artículo solo indica que se añaden desde PORDATA al combinar los datos; no dice a qué año corresponden. Hay 10 combinaciones distintas de (Unemployment rate, Inflation rate, GDP) para 11 años académicos, por lo que probablemente identifican el año de ingreso, pero la correspondencia no está confirmada.

## 6. Limitaciones conocidas

1. **Cobertura.** Una sola institución de Portugal y cohortes de 2008/2009 a 2018/2019. Los resultados no se generalizan a otra institución (por ejemplo, la UASD) ni a cohortes recientes sin revalidación local.
2. **Contradicción en la documentación.** UCI y los autores dicen que el dataset incluye información conocida al momento de la matrícula, pero también incluye rendimiento del 1er y 2do semestre. Usar todas las columnas sin filtrar introduce fuga de información (ver #4).
3. **Sin columna de año o cohorte.** No es posible una partición temporal. Las variables macroeconómicas actúan como indicador indirecto del año de ingreso (sección 5).
4. **Error en el diccionario de UCI.** La descripción de `Curricular units 2nd sem (without evaluations)` dice "in the 1st semester".
5. **Tipo declarado incorrecto.** UCI declara las notas semestrales como `Integer`, pero son decimales entre 0 y 20 (2,994 valores no enteros en el 1er semestre; máximo 18.875).
6. **La nota 0 no es una nota.** Las 718 notas 0 del 1er semestre y las 870 del 2do coinciden siempre con 0 unidades aprobadas: 0 indica que no hay promedio de unidades aprobadas, no una calificación de cero.
7. **Variables categóricas codificadas como enteros.** Hasta 46 categorías (ocupación del padre), 21 nacionalidades, 18 modos de acceso y 17 cursos. Los códigos son nominales: no deben tratarse como magnitudes.
8. **Códigos distintos según la fuente.** El apéndice del artículo codifica los cursos del 1 al 17; el CSV de UCI usa otros códigos (33, 171, 9003, ...). Para interpretar los valores debe usarse el diccionario de UCI.
9. **Versiones distintas.** El artículo describe 35 atributos; la versión de UCI añade `Previous qualification (grade)` y `Admission grade` y tiene 37 columnas.
10. **Desbalance de clases.** Graduate 49.9 %, Dropout 32.1 % y Enrolled 17.9 %. Requiere partición estratificada y métricas por clase.
11. **Definición amplia de deserción.** Los autores cuentan como Dropout los cambios de carrera o de institución, sin importar cuándo ocurren (sec. 1). La tasa de deserción es mayor que la de quienes abandonan la educación superior.
12. **Enrolled es un estado intermedio.** Indica que el estudiante seguía matriculado al final de la duración normal; no garantiza que se gradúe ni que no abandone después.
13. **Colinealidad.** Por ejemplo, `Nacionality` e `International` (r = 0.91) y las variables del 1er y 2do semestre entre sí (Realinho et al., 2022, tabla 9).
14. **Preprocesamiento no reproducible.** Los autores trataron anomalías, atípicos y faltantes antes de publicar, sin describir el procedimiento en detalle. No es posible auditar qué registros se modificaron o eliminaron.

## 7. Variables sensibles

| Variable | Codificación | Distribución |
|---|---|---|
| Gender | 1 = hombre, 0 = mujer | 1,556 hombres (35.2 %), 2,868 mujeres |
| Age at enrollment | Años | Mediana 20, rango 17–70 |
| Nacionality / International | Código / 1 = sí | 4,314 portugueses (97.5 %); 110 internacionales |
| Marital Status | 1–6 | 3,919 solteros (88.6 %); 379 casados; el resto, menos de 3 % |
| Educational special needs | 1 = sí | 51 estudiantes (1.2 %) |
| Debtor | 1 = sí | 503 (11.4 %) |
| Scholarship holder | 1 = sí | 1,099 (24.8 %) |
| Displaced | 1 = sí | 2,426 (54.8 %) |

Los subgrupos pequeños (internacionales, necesidades especiales) limitan la confianza de cualquier análisis por subgrupo (#14). Estas variables no deben usarse para sancionar, excluir ni etiquetar estudiantes (ver [proyecto.md](proyecto.md), sección 5).

## 8. Cómo obtener los datos

Desde la raíz del proyecto:

```bash
uv run python -m src.data
```

El comando descarga `data.csv` a `data/raw/`, verifica su SHA-256 y no vuelve a descargarlo si ya existe con la huella correcta. Los datos crudos no se versionan en Git. Para cargarlos en código:

```python
from src.data import load_raw

df = load_raw()
```

## 9. Licencia y citas obligatorias

El dataset se distribuye bajo **CC BY 4.0**, que exige atribución. Todo uso en el informe, la demostración o el repositorio debe citar:

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), 146. https://doi.org/10.3390/data7110146

Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: A case study. En *Trends and Applications in Information Systems and Technologies* (Advances in Intelligent Systems and Computing, Vol. 1365, pp. 166–175). Springer. https://doi.org/10.1007/978-3-030-72657-7_16

## 10. Fuentes consultadas

- Ficha y API de UCI: https://archive.ics.uci.edu/dataset/697 y https://archive.ics.uci.edu/api/dataset?id=697
- Artículo de datos: Realinho et al. (2022), https://doi.org/10.3390/data7110146
- Registro en Zenodo: https://zenodo.org/records/5777340
- Página de variables de los autores: http://valoriza.ipportalegre.pt/piaes/features-info-stats.html
