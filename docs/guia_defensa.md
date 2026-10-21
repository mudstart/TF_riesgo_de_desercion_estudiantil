# Guía de apoyo para la defensa (#19)

Material de **evidencia** para preparar la defensa individual y el ensayo de la demo. No son respuestas para memorizar: la defensa exige que cada integrante responda **con argumentos propios**. Aquí están las cifras, dónde encontrarlas y qué ejecutar, para que cada uno construya su explicación.

Todas las cifras salen de los archivos versionados de `reports/metrics/` y del [informe técnico](../reports/informe_tecnico.md).

## 1. La pregunta de defensa

> *¿Qué evidencia justifica priorizar acompañamiento temprano, nunca sancionar ni excluir, y qué condición obligaría a detener, abstener o escalar la recomendación?*

La respuesta tiene tres partes. Evidencia disponible para cada una:

**Por qué priorizar acompañamiento temprano**

| Evidencia | Cifra | Fuente |
|---|---|---|
| Recall de Dropout en test (datos no vistos) | 0.803, IC 95 % [0.755, 0.848]: detecta 8 de cada 10 desertores | `test_final.json`; informe §5 |
| Tasa real de deserción según el nivel del modelo | Riesgo 49 % · no concluyente 24 % · sin riesgo 12 % | `analisis_errores.json`; informe §10 |
| Frente al azar | Clasificador aleatorio: F1-macro 0.51 y recall 0.32 | `baseline.json` (`referencia_dummy`); informe §4 |
| El acompañamiento es barato | Un falso positivo cuesta una entrevista o tutoría con revisión humana | `proyecto.md` §5; decisión 008 |

**Por qué nunca sancionar ni excluir**

| Evidencia | Cifra | Fuente |
|---|---|---|
| Precisión baja | 0.446: más de la mitad de los marcados no deserta (283 de 511 en test) | `test_final.json`; informe §5 |
| Error desigual por edad | Recall 0.56 hasta 20 años frente a 1.00 con más de 25 | `analisis_errores.json`; informe §6 |
| Error desigual por género | Recall 0.70 en mujeres frente a 0.92 en hombres | Ídem |
| Correlación, no causa | El modelo asocia variables del ingreso, no explica por qué alguien abandona | Informe §8 y §13 |
| Contexto limitado | Una institución portuguesa, cohortes 2008/09–2018/19, sin validación temporal | Ficha de datos §6; informe §2 |

**Qué condición obliga a detener, abstenerse o escalar** ([decisión 009](decisiones/009_abstencion_y_subgrupos.md))

| Regla | Condición | Acción |
|---|---|---|
| D1 | Otra institución, país o plan de estudios, o cohortes sin revalidar | Detener |
| D2 | Uso para sancionar, excluir, admitir, retirar becas o sin revisión humana | Detener |
| D3 | En una cohorte nueva, recall < 0.75 (límite inferior del IC en test) | Detener y revisar |
| A1 | Dato faltante, valor fuera del rango de train o código nunca visto | Abstenerse |
| E1 | Probabilidad entre 0.332 y 0.432 (tasa real de deserción 24 %, peor que el azar) | Escalar a revisión humana |
| E2 | Código infrecuente en train (< 10 casos) | Escalar con aviso |
| E3 | Subgrupo en alerta y resultado "sin riesgo" | No descartar al estudiante |

## 2. Preguntas probables

| Pregunta | Datos clave para responder | Dónde |
|---|---|---|
| **¿Por qué esa partición?** | 80/20 estratificada por las 3 clases y semilla 42; 5 folds solo dentro de train; test usado **una sola vez** (el script se niega a repetirlo). Sin año en los datos, no es posible una partición temporal | Decisión 003; informe §4 |
| **¿Cómo evitaron la fuga?** | Intervención **al ingreso**: 12 variables de semestre prohibidas; financieras y macroeconómicas fuera por fecha no documentada. Con todas las variables, F1-macro ≈ 0.85 (fuga); al ingreso, ≈ 0.59–0.62. Pipeline ajustado solo con train; pruebas 2 y 4 | Decisión 001; informe §3 |
| **¿Por qué esas métricas?** | Recall de Dropout porque el error costoso es no detectar a un desertor; precisión para no premiar marcar a todos; F1-macro como piso de calidad (≥ 0.62) | `proyecto.md` §4; decisión 008 |
| **¿Ganó el baseline?** | No. Mismo F1-macro (0.62) en los tres, pero el Random Forest detecta más desertores: +0.032 de recall, IC [+0.014, +0.051], diferencia real. El baseline **podía** ganar: el criterio desempataba a su favor | Decisión 008; informe §5 |
| **¿Qué subgrupo sale peor?** | Menores de 21 años (recall 0.56, brecha −0.40) y mujeres (0.70, brecha −0.22); también Nursing, Social Service y Journalism. Sin `Gender`, la brecha de género baja a la mitad sin perder recall | Decisión 009; informe §6 |
| ¿Por qué umbral 0.382 y no 0.5? | Con 0.5 el recall era 0.60; con 0.382 sube a 0.81 manteniendo F1-macro ≥ 0.62. Una tutoría innecesaria cuesta menos que un desertor no detectado | Informe §5 |
| ¿Cómo saben que no sobreajustaron? | Test coincide con validación cruzada dentro de los intervalos (recall 0.803 frente a 0.810) | Informe §5 |
| ¿Vale la pena el modelo complejo (Green AI)? | El RF gasta unas 20 veces la energía del baseline, pero en absoluto es < 1 g de CO₂eq por entrenamiento. El GB sería la opción eficiente | Informe §7 |
| ¿Cómo usaron la IA? | Claude Opus 5.5 a través de Claude Code; cada uno decidía, revisaba y verificaba con pruebas | `declaracion_ia.md` |

## 3. Aporte de cada integrante y qué ejecutar en vivo

El detalle con enlaces está en la [sección 14 del informe](../reports/informe_tecnico.md#14-contribución-de-cada-integrante). Comandos sugeridos para mostrar en vivo, desde la raíz del repositorio:

**Roberto Byas**: registro del equipo y pregunta (#1), auditoría (#6), pruebas 1–3 (#10), errores, subgrupos y abstención (#14).

```bash
uv run pytest tests/test_data.py -k "contrato or anti_fuga or particion"
uv run python -m src.errores
```

**Manicaotex Mueses**: estructura (#2), baseline (#9), comparación y selección (#13), Green AI (#15).

```bash
uv run python -m src.models.train_baseline
uv run python -m src.comparacion
uv run python -m src.green_ai --repeticiones 1 --lotes 20
```

`--repeticiones 1 --lotes 20` hace una medición corta para mostrarla en vivo (~3 min). La oficial usa 3 repeticiones.

**Jose M. Maduro**: dataset (#3), intervención (#4), target (#5), partición (#7), pipeline (#8), Random Forest (#11), Gradient Boosting (#12), pruebas 4–8 y CI (#16), demo (#17), informe (#18).

```bash
uv run python -m src.data
uv run pytest tests/test_pipeline.py tests/test_model.py
uv run streamlit run app/app.py
```

## 4. Guion del ensayo de la demo

Demo integrada, al menos una vez completa (criterio de aceptación del #19):

1. Ejecutar `uv run streamlit run app/app.py` y mostrar el **aviso de uso** y la **trazabilidad** de la barra lateral (versión del modelo, hash del dataset, umbral y zona).
2. **Perfil inventado en vivo**: elegir "Ingresar datos propios" y cambiar varios valores delante del público (edad, notas, carrera). Explicar el nivel obtenido y "qué pesa en este caso".
3. **No concluyente**: con un perfil cerca del umbral (por ejemplo, el de ejemplo "Ingreso a Ingeniería Informática", 0.40), explicar por qué va a revisión humana.
4. **Abstención**: escribir una nota de admisión de 250 y mostrar que el modelo no responde y dice por qué.
5. **Aviso E3**: el perfil "Recién egresada de secundaria, Enfermería" (riesgo bajo) muestra que ese "sin riesgo" no debe usarse para excluirla.
6. Cerrar con "qué **no** permite concluir" y "cuándo no usar el modelo" (D1–D3).

## 5. Coevaluación

Debe basarse en la evidencia de Git y no en impresiones. Para cada integrante: PR propios fusionados, revisiones aprobadas de PR ajenos y commits, que están en la §14 del informe. La sección de cada uno en `declaracion_ia.md` documenta además qué hizo cada persona y qué generó la herramienta.

## 6. Lista de cierre

- [ ] Fusionar el PR del #19 y crear el tag `v1.0` con su release.
- [ ] Ensayar la demo completa en equipo, al menos una vez.
- [ ] Cada integrante: explicar su aporte, ejecutar su parte en vivo y responder la pregunta de defensa con sus palabras.
- [ ] Coevaluación con la evidencia de Git.
- [ ] Subir la entrega a la plataforma **antes del 31-oct-2026 a las 23:59**.
