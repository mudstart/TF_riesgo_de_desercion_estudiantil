Modelo random_forest, umbral 0.382, predicciones out-of-fold de train (n = 3539).

## Métricas por subgrupo

| Variable | Subgrupo | n | Desertores | Prevalencia | Marcados | Recall (IC 95 %) | FPR (IC 95 %) | Precisión | Brecha recall vs resto (IC 95 %) | Brecha FPR vs resto | Alerta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gender | Mujer | 2298 | 581 | 0.25 | 0.44 | 0.70 [0.66, 0.74] | 0.36 [0.34, 0.38] | 0.40 | -0.22 [-0.27, -0.18] | -0.37 | **sí** |
| Gender | Hombre | 1241 | 556 | 0.45 | 0.82 | 0.92 [0.90, 0.95] | 0.73 [0.69, 0.76] | 0.51 | +0.22 [+0.18, +0.27] | +0.37 | no |
| Rango de edad | Hasta 20 | 2061 | 435 | 0.21 | 0.34 | 0.56 [0.52, 0.61] | 0.29 [0.26, 0.31] | 0.35 | -0.40 [-0.45, -0.35] | -0.55 | **sí** |
| Rango de edad | 21 a 25 | 648 | 237 | 0.37 | 0.78 | 0.90 [0.86, 0.94] | 0.71 [0.66, 0.75] | 0.42 | +0.11 [+0.07, +0.16] | +0.29 | no |
| Rango de edad | Más de 25 | 830 | 465 | 0.56 | 0.99 | 1.00 [0.99, 1.00] | 0.98 [0.97, 0.99] | 0.56 | +0.31 [+0.28, +0.35] | +0.61 | no |
| Estado civil | Soltero/a | 3144 | 948 | 0.30 | 0.53 | 0.77 [0.75, 0.80] | 0.42 [0.40, 0.44] | 0.44 | -0.22 [-0.25, -0.18] | -0.51 | **sí** |
| Estado civil | Otro estado civil | 395 | 189 | 0.48 | 0.96 | 0.99 [0.97, 1.00] | 0.93 [0.89, 0.96] | 0.49 | +0.22 [+0.19, +0.24] | +0.51 | no |
| Scholarship holder | No becario | 2645 | 1022 | 0.39 | 0.65 | 0.83 [0.81, 0.85] | 0.53 [0.51, 0.56] | 0.50 | +0.20 [+0.12, +0.30] | +0.22 | no |
| Scholarship holder | Becario | 894 | 115 | 0.13 | 0.36 | 0.63 [0.54, 0.71] | 0.32 [0.29, 0.35] | 0.23 | -0.20 [-0.30, -0.11] | -0.22 | **sí** |
| Displaced | Desplazado | 1942 | 534 | 0.27 | 0.47 | 0.73 [0.69, 0.77] | 0.37 [0.35, 0.40] | 0.43 | -0.15 [-0.19, -0.10] | -0.22 | **sí** |
| Displaced | No desplazado | 1597 | 603 | 0.38 | 0.70 | 0.88 [0.85, 0.91] | 0.59 [0.56, 0.62] | 0.47 | +0.15 [+0.10, +0.19] | +0.22 | no |
| International | Nacional | 3453 | 1114 | 0.32 | 0.57 | 0.81 [0.79, 0.83] | 0.46 [0.44, 0.48] | 0.45 | +0.03 [-0.13, +0.21] | -0.05 | no |
| International | Internacional | 86 | 23 | 0.27 | 0.58 | 0.78 [0.61, 0.96] | 0.51 [0.40, 0.63] | 0.36 | -0.03 [-0.21, +0.14] | +0.05 | sin evidencia (n < 30) |
| Educational special needs | Sin necesidades especiales | 3498 | 1124 | 0.32 | 0.58 | 0.81 [0.79, 0.84] | 0.47 [0.45, 0.49] | 0.45 | +0.27 [+0.03, +0.53] | +0.11 | no |
| Educational special needs | Con necesidades especiales | 41 | 13 | 0.32 | 0.41 | 0.54 [0.23, 0.77] | 0.36 [0.18, 0.54] | 0.41 | -0.27 [-0.57, -0.03] | -0.11 | sin evidencia (n < 30) |
| Course | Nursing | 607 | 90 | 0.15 | 0.17 | 0.53 [0.43, 0.63] | 0.11 [0.08, 0.14] | 0.46 | -0.30 [-0.41, -0.20] | -0.45 | **sí** |
| Course | Management | 302 | 109 | 0.36 | 0.65 | 0.83 [0.76, 0.90] | 0.54 [0.47, 0.61] | 0.47 | +0.03 [-0.05, +0.09] | +0.08 | no |
| Course | Social Service | 291 | 55 | 0.19 | 0.27 | 0.45 [0.33, 0.58] | 0.23 [0.18, 0.29] | 0.31 | -0.37 [-0.51, -0.25] | -0.26 | **sí** |
| Course | Veterinary Nursing | 277 | 79 | 0.29 | 0.47 | 0.76 [0.67, 0.85] | 0.35 [0.29, 0.42] | 0.46 | -0.05 [-0.15, +0.04] | -0.12 | no |
| Course | Journalism and Communication | 267 | 82 | 0.31 | 0.48 | 0.65 [0.55, 0.74] | 0.41 [0.34, 0.48] | 0.41 | -0.18 [-0.28, -0.07] | -0.06 | **sí** |
| Course | Management (evening) | 217 | 113 | 0.52 | 0.97 | 0.98 [0.96, 1.00] | 0.95 [0.90, 0.99] | 0.53 | +0.19 [+0.16, +0.23] | +0.51 | no |
| Course | Advertising and Marketing Management | 213 | 70 | 0.33 | 0.63 | 0.81 [0.73, 0.90] | 0.55 [0.46, 0.62] | 0.42 | +0.00 [-0.09, +0.10] | +0.09 | no |
| Course | Tourism | 198 | 78 | 0.39 | 0.74 | 0.81 [0.72, 0.90] | 0.70 [0.62, 0.78] | 0.43 | -0.00 [-0.09, +0.09] | +0.25 | no |
| Course | Animation and Multimedia Design | 175 | 67 | 0.38 | 0.70 | 0.76 [0.66, 0.87] | 0.66 [0.57, 0.75] | 0.42 | -0.05 [-0.16, +0.05] | +0.20 | no |
| Course | Communication Design | 174 | 39 | 0.22 | 0.51 | 0.74 [0.59, 0.87] | 0.44 [0.36, 0.52] | 0.33 | -0.07 [-0.22, +0.06] | -0.03 | no |
| Course | Agronomy | 171 | 67 | 0.39 | 0.80 | 0.93 [0.87, 0.99] | 0.72 [0.63, 0.81] | 0.45 | +0.12 [+0.05, +0.18] | +0.27 | no |
| Course | Social Service (evening) | 165 | 53 | 0.32 | 0.95 | 0.98 [0.94, 1.00] | 0.93 [0.88, 0.97] | 0.33 | +0.18 [+0.13, +0.22] | +0.49 | no |
| Course | Basic Education | 158 | 72 | 0.46 | 0.77 | 0.88 [0.81, 0.94] | 0.69 [0.58, 0.78] | 0.52 | +0.07 [-0.01, +0.15] | +0.23 | no |
| Course | Informatics Engineering | 133 | 68 | 0.51 | 0.90 | 0.96 [0.90, 1.00] | 0.85 [0.75, 0.94] | 0.54 | +0.16 [+0.10, +0.20] | +0.39 | no |
| Course | Equinculture | 115 | 64 | 0.56 | 0.92 | 0.98 [0.95, 1.00] | 0.84 [0.73, 0.94] | 0.59 | +0.18 [+0.14, +0.22] | +0.39 | no |
| Course | Oral Hygiene | 66 | 24 | 0.36 | 0.65 | 0.88 [0.75, 1.00] | 0.52 [0.38, 0.67] | 0.49 | +0.07 [-0.08, +0.19] | +0.06 | sin evidencia (n < 30) |
| Course | Biofuel Production Technologies | 10 | 7 | 0.70 | 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.70 | +0.19 [+0.17, +0.21] | +0.54 | sin evidencia (n < 30) |

## Perfil de los errores (medias; binarias como proporción)

| Variable | TP | FN | FP | TN | FN − TP | FP − TN |
|---|---|---|---|---|---|---|
| Age at enrollment | 27.68 | 19.20 | 25.23 | 19.05 | -8.48 | +6.18 |
| Admission grade | 123.84 | 128.97 | 125.70 | 129.71 | +5.13 | -4.01 |
| Previous qualification (grade) | 130.05 | 135.26 | 130.31 | 136.19 | +5.21 | -5.88 |
| Application order | 1.45 | 2.07 | 1.51 | 2.05 | +0.62 | -0.54 |
| Gender | 0.56 | 0.19 | 0.45 | 0.14 | -0.36 | +0.30 |
| Displaced | 0.42 | 0.66 | 0.47 | 0.68 | +0.24 | -0.21 |
| Daytime/evening attendance | 0.82 | 0.99 | 0.82 | 0.99 | +0.16 | -0.17 |
| Scholarship holder | 0.08 | 0.20 | 0.22 | 0.41 | +0.12 | -0.19 |
| Debtor | 0.22 | 0.22 | 0.07 | 0.04 | +0.00 | +0.03 |
| Tuition fees up to date | 0.66 | 0.75 | 0.97 | 0.98 | +0.08 | -0.01 |

## Zona de abstención

| Ancho | Zona | Casos en zona | % | Tasa de deserción en zona | % FN en zona | % FP en zona | Recall decididos | Precisión decididos | F1-macro decididos |
|---|---|---|---|---|---|---|---|---|---|
| ±0.000 | [0.382, 0.382] | 0 | 0.0% | nan | 0.0% | 0.0% | 0.810 | 0.453 | 0.620 |
| ±0.025 | [0.357, 0.407] | 374 | 10.6% | 0.26 | 22.7% | 11.3% | 0.839 | 0.469 | 0.633 |
| ±0.050 | [0.332, 0.432] | 747 | 21.1% | 0.24 | 36.1% | 23.5% | 0.856 | 0.491 | 0.644 |
| ±0.075 | [0.307, 0.457] | 1093 | 30.9% | 0.26 | 58.3% | 33.0% | 0.895 | 0.507 | 0.658 |
| ±0.100 | [0.282, 0.482] | 1433 | 40.5% | 0.24 | 67.6% | 42.2% | 0.912 | 0.528 | 0.661 |

## Revisión de la entrada (cada fold contra el dominio de los otros cuatro)

| Condición | Casos | % | Recall | FPR | Recall del resto | FPR del resto | Motivos más frecuentes |
|---|---|---|---|---|---|---|---|
| abstenerse | 68 | 1.9% | 0.87 | 0.51 | 0.80 | 0.46 | Father's occupation: código no visto en train (25); Mother's occupation: código infrecuente en train (16); Father's qualification: código no visto en train (11); Mother's qualification: código no visto en train (9); Mother's occupation: código no visto en train (9) |
| escalar | 220 | 6.2% | 0.89 | 0.56 | 0.80 | 0.46 | Nacionality: código infrecuente en train (47); Mother's qualification: código infrecuente en train (42); Previous qualification: código infrecuente en train (39); Father's occupation: código infrecuente en train (39); Father's qualification: código infrecuente en train (38) |

## Con y sin variables sensibles (cada variante con su umbral)

| Variante | Umbral | F1-macro | Recall | Δ recall vs elegido (IC 95 %) | Precisión | Recall mujer / hombre | Brecha mujer (IC 95 %) | Recall por edad | Brecha hasta 20 (IC 95 %) |
|---|---|---|---|---|---|---|---|---|---|
| modelo elegido | 0.382 | 0.620 | 0.810 | +0.000 [+0.000, +0.000] | 0.453 | 0.70 / 0.92 | -0.22 [-0.27, -0.18] | Hasta 20: 0.56 / 21 a 25: 0.90 / Más de 25: 1.00 | -0.40 [-0.45, -0.35] |
| sin Gender | 0.388 | 0.621 | 0.799 | -0.011 [-0.026, +0.004] | 0.453 | 0.75 / 0.85 | -0.11 [-0.15, -0.06] | Hasta 20: 0.55 / 21 a 25: 0.87 / Más de 25: 1.00 | -0.40 [-0.45, -0.36] |
| sin demográficas sensibles | 0.401 | 0.620 | 0.790 | -0.020 [-0.039, -0.001] | 0.453 | 0.75 / 0.84 | -0.09 [-0.14, -0.04] | Hasta 20: 0.60 / 21 a 25: 0.80 / Más de 25: 0.97 | -0.31 [-0.36, -0.26] |
