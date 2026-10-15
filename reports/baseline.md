# Baseline: regresión logística (#9)

Modelo de referencia simple e interpretable. **No es un trámite:** compite en la comparación (#13) con el mismo criterio que los candidatos y, si empata con ellos, gana por ser el más simple ([decisión 005](../docs/decisiones/005_criterio_seleccion.md)).

Se reproduce con:

```bash
uv run python -m src.train_baseline
```

Salidas: `reports/metrics/baseline.json` (métricas en el formato estándar de `src/training.py`, el mismo que usarán #11 y #12), `reports/metrics/baseline_coeficientes.csv` (los 112 coeficientes) y `models/baseline.joblib` (no versionado).

## 1. Configuración

- `LogisticRegression(class_weight="balanced", max_iter=5000)` dentro de `build_pipeline` (#8): escenario principal, 16 columnas disponibles al ingreso, 112 tras el one-hot.
- Solo train (3,539 filas) y los 5 folds fijos ([decisión 003](../docs/decisiones/003_particion_validacion.md)). El test no se toca.
- **Ajuste ligero de `C`:** 11 valores de 0.001 a 100 en escala logarítmica. Gana el de mayor recall de Dropout out-of-fold con F1-macro ≥ 0.62, cada uno a su umbral: el mismo criterio con el que #13 juzga a los modelos. A igualdad, gana el más regularizado.

| C | 0.001 | 0.003 | 0.01 | **0.032** | 0.1 | 0.32 | 1 | 3.2 | 10 | 32 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Recall Dropout | 0.731 | 0.764 | 0.778 | **0.801** | 0.799 | 0.786 | 0.785 | 0.784 | 0.776 | 0.779 | 0.778 |

Se elige **C = 0.032**. A partir de C ≈ 0.01 la curva es casi plana (0.78–0.80): la ventaja sobre C = 1 (+0.016) es del orden del ruido entre folds (± 0.014). Elegir C con los mismos folds con que se reporta da un sesgo optimista pequeño, que se acepta por ser un único parámetro y una rejilla corta.

## 2. Métricas (validación cruzada, media ± desviación entre folds)

| Modelo | Umbral | F1-macro | Recall Dropout | Precisión Dropout | Balanced acc. |
|---|---|---|---|---|---|
| Dummy estratificado (piso absoluto) | — | 0.508 ± 0.020 | 0.321 ± 0.027 | 0.332 ± 0.028 | 0.508 ± 0.020 |
| Baseline, umbral 0.5 | 0.5 | 0.670 ± 0.016 | 0.650 ± 0.034 | 0.523 ± 0.025 | 0.684 ± 0.013 |
| **Baseline, umbral ajustado** | 0.412 ± 0.006 | 0.620 ± 0.019 | **0.799 ± 0.014** | 0.453 ± 0.017 | 0.671 ± 0.013 |

El umbral ajustado de cada fold se elige con los otros cuatro (sin fuga). Out-of-fold, sobre todo train y al umbral 0.412:

| | Predicho: no Dropout | Predicho: Dropout |
|---|---|---|
| **Real: no Dropout** | 1,305 | 1,097 |
| **Real: Dropout** | 226 | 911 |

De cada 5 desertores se detectan 4 (recall 0.80). Menos de la mitad de los marcados son desertores (precisión 0.45), frente al 0.32 de marcar al azar. Con el umbral 0.5 se detectarían solo 739 de 1,137, es decir, 172 desertores menos.

**Coste:** entrenamiento 0.06 s por fold (0.18 s con todo train) e inferencia ≈ 16 ms por cada 1,000 registros. Energía y emisiones: pendientes (#15).

## 3. Interpretación de los coeficientes

Los coeficientes son log-odds: un valor positivo aumenta el riesgo de Dropout. Las numéricas están estandarizadas (efecto de +1 desviación estándar). Los 20 de la tabla conservan el signo en los 5 folds. Los 40 de 112 que lo cambian en algún fold tienen |coef| < 0.09.

**Cómo leerlos:**

- **Sirven para el signo y el orden, no para la magnitud exacta.** Las one-hot no tienen categoría de referencia: la regularización L2 reparte el efecto entre todas las categorías de una variable y el intercepto.
- **Las escalas no son iguales.** En las binarias (Gender) el coeficiente es el efecto de pasar de 0 a 1; en las numéricas (Age, Admission grade), el de subir una desviación estándar; en las one-hot, el de pertenecer a la categoría. Ordenarlas juntas por tamaño es una guía, no una comparación exacta.
- **Están encogidos.** Con C = 0.032 la regularización es fuerte y todos los coeficientes se acercan a 0. Los odds ratio (OR) son menores que los de una logística sin penalizar, así que no deben leerse como estimaciones sin sesgo ni como efectos causales.

Nombres de los cursos según [docs/ficha_datos.md](../docs/ficha_datos.md). Los de los demás códigos (modo de acceso, ocupación y formación de los padres, formación previa) vienen del diccionario de variables de UCI (dataset 697). «Infrecuente» es el grupo de categorías con menos de 10 casos en train (decisión 004): junta categorías distintas y **no se interpreta**.

| Aumentan el riesgo | coef | OR | Reducen el riesgo | coef | OR |
|---|---|---|---|---|---|
| Gender = hombre | +0.53 | 1.69 | Course 9500 Nursing | −0.62 | 0.54 |
| Application mode 39 (mayores de 23) | +0.46 | 1.58 | Course 8014 Social Service (noche) | −0.47 | 0.63 |
| Course 9853 Basic Education | +0.44 | 1.56 | Application mode 1 (1.ª fase, contingente general) | −0.40 | 0.67 |
| Course 9130 Equinculture | +0.41 | 1.50 | Father's occupation infrecuente (no interpretable) | −0.38 | 0.69 |
| Mother's occupation 0 (estudiante) | +0.36 | 1.43 | Course 9238 Social Service | −0.30 | 0.74 |
| Course 171 Animation and Multimedia Design | +0.35 | 1.42 | Mother's occupation infrecuente (no interpretable) | −0.27 | 0.77 |
| Application mode 7 (titulados de otro curso superior) | +0.34 | 1.41 | Previous qualification 1 (secundaria) | −0.23 | 0.80 |
| Mother's qualification 34 (desconocida) | +0.33 | 1.40 | Course 9070 Communication Design | −0.21 | 0.81 |
| Age at enrollment (+1 DE) | +0.29 | 1.34 | Father's qualification 38 (básica, 2.º ciclo) | −0.21 | 0.81 |
| Father's occupation 90 (otra situación) | +0.29 | 1.33 | Admission grade (+1 DE) | −0.21 | 0.81 |

**En pocas líneas:**

1. **Ingreso no tradicional:** el perfil de mayor riesgo es el del estudiante adulto: mayor edad, acceso por la vía de mayores de 23 o como titulado de otro curso superior. Las dos primeras variables se solapan, así que conviene leerlas juntas. El acceso por la 1.ª fase del contingente general reduce el riesgo.
2. **El curso pesa mucho:** Enfermería y Servicio Social (diurno y nocturno) protegen. Educación Básica, Equinocultura y Diseño de Animación y Multimedia aumentan el riesgo, en línea con la auditoría (#6: el Dropout va del 15 % al 67 % según el curso).
3. **Rendimiento previo:** una nota de admisión más alta reduce el riesgo, con un efecto moderado (OR 0.81 por desviación estándar).
4. **Atributos sensibles y datos faltantes:** ser hombre es el mayor coeficiente positivo (OR 1.69), pero está mezclado con el curso (V = 0.42) y es asociación, no causa. Que la formación de la madre sea «desconocida» también aumenta el riesgo. Ambas cosas deben revisarse en el análisis de equidad (#14) antes de usar el modelo para priorizar.

## 4. Lo que pasa a #13

- `models/baseline.joblib` con `C = 0.032`, que `src.comparacion` carga como modelo `baseline`.
- Con la regla de la decisión 005, su umbral (0.412) y su recall out-of-fold (0.801) son los que #13 recalcula y compara con Random Forest y Gradient Boosting.
