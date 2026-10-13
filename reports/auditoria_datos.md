# Auditoría de datos (EDA)

- Issue: [#6](https://github.com/mudstart/TF_riesgo_de_desercion_estudiantil/issues/6) Auditoría de datos (EDA).
- Rama: `feature/6-auditoria-datos`.
- Notebook: [`notebooks/01_auditoria_datos.ipynb`](../notebooks/01_auditoria_datos.ipynb). Figuras: [`reports/figures/`](figures/).
- Datos: UCI 697, `data.csv` con SHA-256 verificado (ver [ficha de datos](../docs/ficha_datos.md)).
- Fecha: 2026-10-13.

## 1. Alcance

Auditoría de la calidad del dataset antes de modelar. La intervención se fija al ingreso ([proyecto.md](../docs/proyecto.md), sección 4), así que la relación con el target se analiza solo con las **21 variables permitidas** según #4. Las 3 variables ambiguas (`Debtor`, `Tuition fees up to date`, `Scholarship holder`) se revisan solo como posible fuga y las 12 de rendimiento semestral, solo como control de calidad.

Las medidas de asociación se calcularon sobre todo el dataset y son descriptivas: no deben usarse para seleccionar variables ni para fijar hiperparámetros, que se deciden solo con la partición de entrenamiento (#7, #8).

> Las listas `FEATURES_INGRESO`, `FEATURES_SEM1` y `FEATURES_PROHIBIDAS` que exige #4 aún no están en `src/config.py`. El notebook reconstruye la misma clasificación y deberá importarlas cuando se integren.

## 2. Calidad de los datos

| Verificación | Resultado |
|---|---|
| Dimensiones | 4,424 filas × 37 columnas (36 características y `Target`) |
| Tipos | 29 enteras, 7 decimales y `Target` como texto. Las notas semestrales son decimales aunque UCI las declare `Integer` (2,994 y 2,869 valores no enteros) |
| Nulos | 0 (confirmado) |
| Duplicados | 0 filas completas, 0 sin el target y 0 perfiles repetidos en las 21 variables permitidas |
| Fuera de rango | 0. Edad 17–70; nota de admisión y de titulación previa 95–190 (escala 0–200); notas semestrales 0–18.9 (escala 0–20); binarias en {0, 1} |
| Coherencia | En ningún semestre hay más unidades aprobadas que inscritas |
| `Application order` | Valores 1–6, más un único 0 y un único 9 |

### Casos especiales (variables prohibidas al ingreso)

- **180 estudiantes con 0 unidades inscritas en el 1er semestre**, que también tienen 0 en el 2do, 0 acreditadas y nota 0. **Todos pertenecen al curso 171** (Animation and Multimedia Design): son el 84 % de ese curso (180 de 215), y el 41.7 % de ellos figura como Graduate sin haber inscrito ninguna unidad en el primer año. No describe el comportamiento de los estudiantes: **faltan los registros semestrales de ese curso**. Es una limitación nueva, no documentada en la ficha de datos.
- **Nota semestral 0 = "sin notas"**: coincide siempre con 0 unidades aprobadas (718 casos en el 1er semestre y 870 en el 2do). Con nota 0, el Dropout es de 79.4 % y 83.6 %, frente a 32.1 % de base. Figura: `01_auditoria_nota_cero.png`.

## 3. Variable objetivo

| Clase | Estudiantes | % |
|---|---|---|
| Graduate | 2,209 | 49.9 |
| Dropout | 1,421 | 32.1 |
| Enrolled | 794 | 17.9 |

En la formulación principal (decisión 002), Dropout 1,421 (32.1 %) frente a No-Dropout 3,003 (67.9 %): desbalance de 2.11 : 1. Un modelo que nunca predice Dropout obtiene 67.9 % de exactitud. Figura: `02_auditoria_target.png`.

## 4. Distribuciones y relación con el target

**Univariadas** (`03_auditoria_univariadas.png`). `Admission grade` (mediana 126.1) y `Previous qualification (grade)` (mediana 133.1) son aproximadamente simétricas. `Age at enrollment` es muy asimétrica (asimetría 2.05): mediana 20, pero el 25.7 % ingresa con 25 años o más y el 10 % con 35 o más.

**Numéricas permitidas** (AUC univariada para separar Dropout; 0.5 = sin señal). Figura: `04_auditoria_numericas_vs_target.png`.

| Variable | AUC | Dropout asociado a valores |
|---|---|---|
| Age at enrollment | 0.671 | altos |
| Admission grade | 0.561 | bajos |
| Previous qualification (grade) | 0.549 | bajos |
| Application order | 0.542 | bajos |
| GDP / Inflation rate / Unemployment rate | 0.535 / 0.514 / 0.506 | — |

**Categóricas permitidas** (V de Cramér con corrección de sesgo). Figura: `05_auditoria_categoricas_vs_target.png`.

| Variable | V | Variable | V |
|---|---|---|---|
| Application mode | 0.294 | Father's qualification | 0.183 |
| Course | 0.253 | Father's occupation | 0.167 |
| Previous qualification | 0.205 | Marital Status | 0.111 |
| Gender | 0.203 | Displaced | 0.106 |
| Mother's occupation | 0.197 | Daytime/evening attendance | 0.079 |
| Mother's qualification | 0.189 | Nacionality, International, Educational special needs | 0.000 |

**Variables ambiguas: señal de fuga.**

| Variable | Valor | n | % Dropout |
|---|---|---|---|
| Tuition fees up to date | 0 | 528 | 86.6 |
| Debtor | 1 | 503 | 62.0 |
| Scholarship holder | 1 | 1,099 | 12.2 |

`Tuition fees up to date` tiene una asociación (V = 0.43) mayor que la de cualquier variable permitida. Es compatible con un registro durante la carrera.

## 5. Correlaciones y multicolinealidad

Figura: `06_auditoria_correlaciones.png`.

- **Redundancias exactas:** `International` = 1 si y solo si `Nacionality` ≠ 1, y `Daytime/evening attendance` queda determinada por `Course` (los únicos cursos nocturnos son 8014 y 9991). Con one-hot, ambas son combinaciones lineales exactas de otras columnas.
- **Numéricas:** sin multicolinealidad preocupante. La mayor correlación es `Admission grade` ~ `Previous qualification (grade)` (ρ = 0.58) y todos los VIF son < 1.6.
- **Categóricas:** ocupaciones de madre y padre (V = 0.57), titulaciones de madre y padre (0.43), `Course` ~ `Gender` (0.42) y `Application mode` con varias variables (0.41–0.53).
- **Proxy de cohorte:** `Unemployment rate`, `Inflation rate` y `GDP` toman solo 10 combinaciones, con Dropout entre 24.0 % y 38.4 %.

## 6. Variables categóricas y cardinalidad

Las 9 nominales están codificadas como enteros sin orden (por ejemplo, cursos de 33 a 9991). El Dropout por curso va de 15.4 % (Nursing) a 66.7 % sin relación con el código (ρ = −0.08).

| Variable | Categorías | Con < 10 registros | Filas afectadas | Categorías tras agrupar |
|---|---|---|---|---|
| Father's occupation | 46 | 32 | 78 | 15 |
| Father's qualification | 34 | 22 | 54 | 13 |
| Mother's occupation | 32 | 18 | 62 | 15 |
| Mother's qualification | 29 | 19 | 68 | 11 |
| Nacionality | 21 | 16 | 32 | 6 |
| Application mode | 18 | 4 | 6 | 15 |
| Course | 17 | 0 | 0 | 17 |
| Previous qualification | 17 | 7 | 29 | 11 |
| Marital Status | 6 | 2 | 10 | 5 |

## 7. Variables sensibles

Figura: `07_auditoria_sensibles.png`.

| Grupo | n | % estudiantes | % Dropout |
|---|---|---|---|
| Hombre / Mujer | 1,556 / 2,868 | 35.2 / 64.8 | 45.1 / 25.1 |
| Edad 17–19 / 20–24 / 25–34 / 35+ | 1,952 / 1,334 / 697 / 441 | 44.1 / 30.2 / 15.8 / 10.0 | 21.0 / 28.6 / 56.5 / 53.7 |
| Portuguesa / Otra nacionalidad | 4,314 / 110 | 97.5 / 2.5 | 32.2 / 29.1 |
| Becario: No / Sí (ambigua) | 3,325 / 1,099 | 75.2 / 24.8 | 38.7 / 12.2 |
| Soltero/a / Otro estado civil | 3,919 / 505 | 88.6 / 11.4 | 30.2 / 46.9 |
| Necesidades educativas especiales: Sí | 51 | 1.2 | 33.3 |

La diferencia por género puede estar mezclada con el curso (V `Course`–`Gender` = 0.42) y la del estado civil, con la edad. Los extranjeros y los estudiantes con necesidades especiales son grupos demasiado pequeños para conclusiones firmes.

## 8. Decisiones derivadas

| # | Hallazgo | Decisión | Issue |
|---|---|---|---|
| 1 | 0 nulos y 0 duplicados | Sin imputación ni deduplicación. El pipeline conserva un imputador y `handle_unknown` para datos nuevos; las pruebas de contrato verifican 0 nulos y 0 duplicados | #8, #10 |
| 2 | Sin valores fuera de rango; edad con cola larga | No recortar extremos (son subgrupos reales). Usar los rangos de la sección 2 como contrato de entrada | #10 |
| 3 | 180 registros del curso 171 sin datos semestrales; nota 0 = "sin notas" | No afecta al escenario de ingreso. En un escenario tras el 1er semestre: tratarlos como faltantes, no como ceros, y nota 0 → faltante + indicador. Añadir la limitación a la ficha de datos | #3, #4 |
| 4 | Desbalance 2.11 : 1 | Partición y CV estratificadas por la clase original; F1-macro y recall de Dropout con su precisión; evaluar `class_weight="balanced"` y el umbral antes que remuestreo | #7, #9 |
| 5 | Señal moderada en las permitidas | Esperar un desempeño modesto en el escenario de ingreso; no añadir variables posteriores al ingreso para mejorarlo | #9, #13 |
| 6 | Ambiguas con asociación muy fuerte | Fuera del modelo principal; solo análisis de sensibilidad con y sin ellas | #4, #13 |
| 7 | `International` y `Daytime/evening attendance` redundantes de forma exacta | Excluirlas de las features del modelo; conservarlas para describir subgrupos | #4, #8 |
| 8 | Macroeconómicas = proxy de cohorte | Mantenerlas como permitidas y documentar que la validación no puede ser temporal | #13 |
| 9 | 9 nominales codificadas como enteros | `OneHotEncoder` para nominales, binarias sin transformar y `StandardScaler` para numéricas en un `ColumnTransformer` ajustado solo con entrenamiento | #8 |
| 10 | Muchas categorías con < 10 registros | Agrupar en "Otros" con `OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist")`, aprendido solo con entrenamiento (agrupar con todo el dataset sería una fuga leve) | #8 |
| 11 | Diferencias por género y edad; subgrupos pequeños | Reportar recall, precisión y falsos positivos por género, edad y nacionalidad con el tamaño de cada grupo; comparar un modelo sin `Gender` | #9, #14 |

## 9. Reproducir

```bash
uv sync
uv run python -m src.data
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_auditoria_datos.ipynb
```
