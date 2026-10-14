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
| [003](003_particion_validacion.md) | Partición 80/20 estratificada y 5 folds fijos; test solo al final | #7 | Jose Maduro | Propuesta |

Decisiones previstas: preprocesamiento (#8), umbral de decisión y criterio de elección del modelo (#13).
