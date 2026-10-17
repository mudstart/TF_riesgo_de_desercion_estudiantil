# Baseline: regresión logística (#9)

Modelo de referencia simple e interpretable. **No es un trámite:** compite con los candidatos (#11, #12) en la comparación (#13) y, si gana, se reporta y se elige.

Se reproduce con:

```bash
uv run python -m src.models.train_baseline
```

Salidas: `reports/metrics/baseline.json` (métricas en el formato estándar de `src/training.py`, el mismo que usarán #11 y #12), `reports/metrics/baseline_coeficientes.csv` (los 112 coeficientes) y `models/baseline.joblib` (no versionado).

## 1. Configuración

- `LogisticRegression(class_weight="balanced", max_iter=5000)` dentro de `build_pipeline` (#8): escenario principal, 16 columnas disponibles al ingreso, 112 tras el one-hot.
- Solo train (3,539 filas) y los 5 folds fijos estratificados ([decisión 003](../docs/decisiones/003_particion_validacion.md)). El test no se toca.
- Métricas al umbral 0.5. El umbral de decisión de todos los modelos se fija después, en #13, con un mismo criterio.
- **Ajuste ligero de `C`:** 11 valores de 0.001 a 100 en escala logarítmica, con F1-macro medio entre folds. Se aplica la **regla de una desviación estándar** ([decisión 005](../docs/decisiones/005_ajuste_y_metricas.md)): gana el C más regularizado (más simple) cuyo F1-macro alcanza el del mejor menos su desviación estándar entre folds. Es más conservadora que la regla 1-SE clásica, que resta el error estándar (std/√5): con ella el corte sería 0.671 y saldría C = 1.

| C | 0.001 | 0.003 | **0.01** | 0.032 | 0.1 | 0.32 | 1 | 3.2 | 10 | 32 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1-macro | 0.657 | 0.664 | **0.670** | 0.670 | 0.670 | 0.671 | 0.673 | 0.673 | 0.672 | 0.672 | 0.673 |
| Recall Dropout | 0.573 | 0.599 | **0.631** | 0.650 | 0.658 | 0.659 | 0.659 | 0.663 | 0.661 | 0.665 | 0.664 |

El mejor F1-macro es 0.673 ± 0.005 (C = 3.16), así que el corte es 0.668 y se elige **C = 0.01**. Desde C = 0.01 la curva es plana: las diferencias (≤ 0.003) son menores que la variación entre folds.

**Por qué no el máximo.** Con C = 3.16 la regularización es débil y los coeficientes más grandes salían de grupos de 22 a 31 estudiantes. Por ejemplo, una ocupación de la madre con 22 casos y ningún desertor quedaba como el factor más protector (OR 0.11). Con C = 0.01 el modelo se apoya en variables con mucha evidencia, y todos los coeficientes que cambian de signo entre folds son despreciables (|coef| < 0.04). Elegir C con los mismos folds con que se reporta da un sesgo optimista pequeño, que se acepta por ser un único parámetro y una rejilla corta.

## 2. Métricas (validación cruzada, umbral 0.5, media ± desviación entre folds)

| Modelo | F1-macro | Recall Dropout | Precisión Dropout | Balanced acc. |
|---|---|---|---|---|
| Dummy estratificado (piso absoluto) | 0.508 ± 0.020 | 0.321 ± 0.027 | 0.332 ± 0.028 | 0.508 ± 0.020 |
| **Baseline (C = 0.01)** | **0.670 ± 0.018** | **0.631 ± 0.031** | 0.527 ± 0.030 | 0.681 ± 0.015 |

Matriz de confusión out-of-fold, sobre todo train:

| | Predicho: no Dropout | Predicho: Dropout |
|---|---|---|
| **Real: no Dropout** | 1,753 | 649 |
| **Real: Dropout** | 419 | 718 |

El baseline supera claramente al azar (+0.16 de F1-macro). Al umbral 0.5 detecta a 718 de 1,137 desertores (63 %), y algo más de la mitad de los marcados lo son (precisión 0.53, frente a 0.32 al azar). El recall de Dropout es la métrica prioritaria del proyecto: subirlo moviendo el umbral, y cuánto cuesta en precisión, se decide en #13 para todos los modelos por igual.

**Coste:** entrenamiento ≈ 0.06 s por fold y con todo train; inferencia ≈ 18 ms por cada 1,000 registros. Energía y emisiones: pendientes (#15).

## 3. Interpretación de los coeficientes

Los coeficientes son log-odds: un valor positivo aumenta el riesgo de Dropout. Los 20 de la tabla conservan el signo en los 5 folds.

**Cómo leerlos:**

- **Sirven para el signo y el orden, no para la magnitud exacta.** Las one-hot no tienen categoría de referencia: la regularización L2 reparte el efecto entre todas las categorías de una variable y el intercepto.
- **Las escalas no son iguales.** En las binarias (Gender) el coeficiente es el efecto de pasar de 0 a 1; en las numéricas (Age, Admission grade), el de subir una desviación estándar; en las one-hot, el de pertenecer a la categoría. Ordenarlas juntas por tamaño es una guía, no una comparación exacta.
- **Están encogidos.** Con C = 0.01 la regularización es fuerte y todos los coeficientes se acercan a 0. Los odds ratio (OR) son menores que los de una logística sin penalizar, así que no deben leerse como estimaciones sin sesgo ni como efectos causales.

Nombres de los cursos según [docs/ficha_datos.md](../docs/ficha_datos.md). Los de los demás códigos (modo de acceso, ocupación y formación de los padres, formación previa) vienen del diccionario de variables de UCI (dataset 697). «Infrecuente» es el grupo de categorías con menos de 10 casos en train (decisión 004): junta categorías distintas y **no se interpreta**.

| Aumentan el riesgo | coef | OR | Reducen el riesgo | coef | OR |
|---|---|---|---|---|---|
| Gender = hombre | +0.43 | 1.53 | Course 9500 Nursing | −0.41 | 0.66 |
| Age at enrollment (+1 DE) | +0.30 | 1.35 | Application mode 1 (1.ª fase, contingente general) | −0.32 | 0.73 |
| Application mode 39 (mayores de 23) | +0.30 | 1.34 | Course 8014 Social Service (noche) | −0.20 | 0.82 |
| Course 9853 Basic Education | +0.22 | 1.25 | Course 9238 Social Service | −0.18 | 0.83 |
| Mother's occupation 0 (estudiante) | +0.20 | 1.23 | Admission grade (+1 DE) | −0.18 | 0.84 |
| Course 9130 Equinculture | +0.20 | 1.22 | Previous qualification 1 (secundaria) | −0.17 | 0.84 |
| Mother's qualification 34 (desconocida) | +0.20 | 1.22 | Father's occupation infrecuente (no interpretable) | −0.16 | 0.85 |
| Course 171 Animation and Multimedia Design | +0.17 | 1.19 | Father's qualification 38 (básica, 2.º ciclo) | −0.14 | 0.87 |
| Application mode 7 (titulados de otro curso superior) | +0.17 | 1.18 | Mother's occupation infrecuente (no interpretable) | −0.12 | 0.89 |
| Father's occupation 0 (estudiante) | +0.14 | 1.15 | Father's occupation 7 (trabajador cualificado de industria) | −0.12 | 0.89 |

**En pocas líneas:**

1. **Ingreso no tradicional:** el perfil de mayor riesgo es el del estudiante adulto: mayor edad, acceso por la vía de mayores de 23 o como titulado de otro curso superior. La edad y el acceso por mayores de 23 se solapan, así que conviene leerlas juntas. Entrar por la 1.ª fase del contingente general, la vía habitual al salir de secundaria, reduce el riesgo.
2. **El curso pesa mucho:** Enfermería y Servicio Social (diurno y nocturno) protegen. Educación Básica, Equinocultura y Diseño de Animación y Multimedia aumentan el riesgo, en línea con la auditoría (#6: el Dropout va del 15 % al 67 % según el curso).
3. **Rendimiento previo:** una nota de admisión más alta reduce el riesgo, con un efecto moderado (OR 0.84 por desviación estándar).
4. **Atributos sensibles y datos faltantes:** ser hombre es el mayor coeficiente positivo (OR 1.53), pero está mezclado con el curso (V = 0.42) y es asociación, no causa. Que la formación de la madre sea «desconocida» también aumenta el riesgo (74 % de Dropout en ese grupo). Ambas cosas deben revisarse en el análisis de equidad (#14) antes de usar el modelo para priorizar.

## 4. Lo que pasa a otros issues

- **#11 y #12:** ajustan y reportan con las mismas funciones de `src/training.py` (`tune`, `model_report`, `save_metrics`), con la rejilla ordenada del modelo más simple al más complejo ([decisión 005](../docs/decisiones/005_ajuste_y_metricas.md)).
- **#13:** carga `models/baseline.joblib` (C = 0.01) como modelo `baseline` y fija su umbral con el mismo criterio que el de los candidatos.
- **#15:** escribe `energia_kwh` y `emisiones_gco2eq` en `reports/metrics/baseline.json`; volver a ejecutar este script conserva esos valores.
- **#14:** revisa el peso de `Gender` y de la formación de la madre «desconocida».
