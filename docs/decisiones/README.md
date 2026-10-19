# Registro de decisiones

Cada archivo de esta carpeta documenta **una** decisión del proyecto: qué se decidió, por qué, qué alternativas se descartaron y qué consecuencias tiene. Sirve para la trazabilidad que exige el enunciado (desde los datos y el modelo hasta la conclusión) y para que los tres integrantes puedan explicar cada decisión en la defensa.

## Regla obligatoria

**Todo PR que tome una decisión de las listadas abajo debe incluir su documento en esta carpeta.** Sin el documento, el revisor no aprueba el PR.

Es obligatorio documentar una decisión cuando:

1. Define o cambia **qué datos o variables** se usan (por ejemplo, variables permitidas, filtrado de filas).
2. Define o cambia **qué se predice o cómo se mide** (target, métricas, umbral de decisión, criterio para elegir el modelo final).
3. Define **cómo se valida** (partición, validación cruzada, semilla).
4. Define **cómo se preparan los datos** (codificación, escalado, tratamiento de desbalance).
5. Afecta el trabajo de **otro integrante u otro issue**.
6. Es **difícil de revertir** o contradice algo ya escrito en [proyecto.md](../proyecto.md).

No hace falta documentar detalles de implementación que no cambian resultados (nombres de funciones internas, refactorizaciones, formato).

## Cómo crear una decisión

1. Copiar [plantilla.md](plantilla.md) como `NNN_tema_corto.md`, con el siguiente número libre (`003_`, `004_`, ...).
2. Completar todas las secciones. Si una no aplica, escribir "No aplica" y por qué.
3. Si la decisión cambia una lista o constante del código, añadir junto a ella un comentario `# ver docs/decisiones/NNN_tema_corto.md`.
4. Añadir la decisión al índice de abajo.
5. Incluir el documento en el mismo PR que implementa la decisión.

## Cambiar una decisión

Una decisión aprobada **no se reescribe**. Se crea una nueva que indique "Reemplaza a NNN" y se cambia el estado de la anterior a "Reemplazada por MMM". Así queda la historia de por qué cambió.

## Estados

- **Propuesta:** en revisión en un PR.
- **Aprobada:** el PR se fusionó con al menos una revisión de otro integrante.
- **Reemplazada por NNN:** ya no está vigente.

## Índice

| N.º | Decisión | Issue | Responsable | Estado |
|---|---|---|---|---|
| [001](001_momento_intervencion.md) | Momento de intervención al ingreso y variables permitidas | #4 | Jose Maduro | Aprobada |
| [002](002_formulacion_target.md) | Target binario (principal) y multiclase (secundaria) | #5 | Jose Maduro | Aprobada |
| [003](003_particion_validacion.md) | Partición 80/20 estratificada y 5 folds fijos; test solo al final | #7 | Jose Maduro | Aprobada |
| [004](004_preprocesamiento.md) | Preprocesamiento sin fuga: one-hot con infrecuentes, escalado y class_weight | #8 | Jose Maduro | Propuesta |
| [005](005_ajuste_y_metricas.md) | Ajuste de hiperparámetros (F1-macro, regla de una desviación estándar) y formato estándar de métricas | #9 | Manicaotex Mueses | Aprobada (PR #33); criterio de ajuste reemplazado por 006 para #11 y #12 |
| [006](006_busqueda_aleatoria.md) | Búsqueda aleatoria (25 combinaciones, mayor F1-macro) para los candidatos | #11 | Jose Maduro | Aprobada (PR #34) |
| [007](007_gradient_boosting.md) | Gradient Boosting: HistGradientBoosting con early stopping y class_weight | #12 | Jose Maduro | Aprobada (PR #35) |
| [008](008_criterio_seleccion.md) | Umbral de decisión (mayor recall con F1-macro ≥ 0.62) y criterio de selección del modelo final, fijados antes de test | #13 | Manicaotex Mueses | Aprobada (PR #37) |
| [010](010_pruebas_y_ci.md) | Prueba de desempeño mínimo sin test (out-of-fold, F1-macro ≥ 0.60 y mejor que Dummy) y pytest en cada PR con GitHub Actions | #16 | Jose Maduro | Propuesta |

