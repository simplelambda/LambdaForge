# Tareas genéricas de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Las tareas genéricas son la unidad de LambdaForge para trabajo que no tiene por qué ser training:
preprocesado, descargas, extracción de features, inferencia, evaluación, export, figuras o cualquier
operación batch. Usan un Schema estricto independiente y reutilizan la factoría, provenance de
entorno/plugins, resultados atómicos, historial de intentos y ResultCatalog.

## Empieza aquí

Una tarea tiene cuatro partes visibles: **inputs** declarados, el **objeto de tarea** Python,
**outputs/métricas** devueltos y **artefactos** verificados en disco. Contenido de inputs y
configuración determinan el fingerprint. Los outputs son valores JSON pequeños para lógica
posterior; los artefactos son ficheros/directorios con hash. `required_artifacts` añade una condición
de éxito, no es una lista de ficheros que LambdaForge cree automáticamente.

Usa una tarea para una operación acotada, un experimento cuando necesites la semántica de training
Lightning y un workflow cuando varias operaciones completas dependan entre sí. El ejemplo mínimo
supone que `mi_proyecto.preprocessing.SurfaceTask` pertenece a un paquete consumidor instalado.

La versión 0.5 también acepta autoría concisa. `inspect --resolved` muestra la tarea estricta que se
genera; la validación y ejecución siguen usando el mismo runner:

```yaml
name: preparar-superficies
inputs: {raw: data/raw}
outputs: {surfaces: surfaces}
task:
  target: mi_proyecto.preprocessing.SurfaceTask
  params: {resolution: 1.0}
resources: {cpus: 4, memory: 8GiB}
```

## Tarea mínima

```yaml
schema_version: "1.0"
kind: task
name: prepare-surfaces
output_root: runs/tasks
resume: true
rerun_completed: false
inputs:
  - name: raw
    path: data/raw
required_artifacts: [surfaces]
task:
  target: mi_proyecto.preprocessing.SurfaceTask
  params: {resolution: 1.0}
execution: {mode: sequential}
metadata: {purpose: Construir superficies moleculares reutilizables.}
```

Los `inputs` se resuelven respecto al YAML y se hashean antes de planificar. Los bytes de ficheros
o directorios participan en el fingerprint: cambiar los datos raw selecciona otro run aunque la
ruta sea idéntica. Declara todo input científico local. Los outputs permanecen bajo el directorio
fingerprinted del task.

Los comandos existentes despachan según `kind`:

```bash
lambdaforge validate task.yaml
lambdaforge inspect task.yaml
lambdaforge run task.yaml --dry-run
lambdaforge run task.yaml
lambdaforge results task.yaml --write-index --fail-on-ambiguous
```

La validación comprueba Schema 1.0, imports, plugins y firma del constructor sin instanciar la tarea
ni crear outputs. Inspect y dry-run devuelven el mismo `TaskExecutionPlan` inmutable y no construyen
código del usuario.

## Contrato Python

```python
from lambdaforge.tasks import ArtifactDeclaration, ArtifactType, Task, TaskContext, TaskOutput


class SurfaceTask(Task):
    def __init__(self, resolution: float) -> None:
        self.resolution = resolution

    def run(self, context: TaskContext) -> TaskOutput:
        raw = context.input("raw")
        output = context.output("surfaces", create=True)
        output.mkdir(exist_ok=True)
        # El trabajo específico del proyecto escribe bajo output.
        return TaskOutput(
            outputs={"surface_dir": "surfaces"},
            metrics={"resolution": self.resolution},
            artifacts=[ArtifactDeclaration("surfaces", kind=ArtifactType.DATASET)],
        )
```

`context.input(name)` resuelve/verifica un input lógico declarado. `context.output(name,
create=True)` resuelve un output configurado bajo el run y crea su padre. Los métodos de paths
anteriores siguen por compatibilidad; una tarea no debe deducir manualmente directorios con hash.

Heredar se recomienda para plugins pero es opcional con `target`: se admite un objeto externo con
`run(context)` y también la forma duck-typed concisa `run()` sin argumentos. Puede devolver
`TaskOutput`, un mapping (interpretado íntegramente como `outputs`) o `None`. Usa `TaskOutput` para
publicar métricas, artefactos o metadata.

`TaskContext` proporciona attempt ID, fingerprint, directorio del YAML, inputs materializados,
resume, cancelación y helpers seguros `input_path`/`output_path`. Una tarea no debe escribir fuera
del `run_dir` y declarar después esa ruta externa como artefacto.

## Resultados, artefactos y resume

Cada identidad se almacena en:

```text
<output_root>/<task-name>/<fingerprint-prefix>/
├── config.yaml
├── environment.json
├── task.log
├── events.jsonl
├── result.json
└── .lambdaforge/attempts/result-<attempt-id>.json
```

`TaskResult` contiene estado, tiempos, error estructurado, outputs, métricas escalares, metadata y
entradas `TaskArtifact`. Cada artefacto es relativo al run, no puede escapar mediante `..` o
symlinks, debe existir y registra rol, tamaño y SHA-256 determinista. Los directorios incluyen en el
hash nombres relativos ordenados y contenidos. `events.jsonl` es el stream estructurado append-only
de inicio/fin; los fallos incluyen categoría conservadora. Complementa `task.log`, pero no es fuente
de verdad de resultados.

Un éxito sólo se omite mientras todos sus artefactos conserven digest y estén presentes los
`required_artifacts`. `rerun_completed: true` crea otro intento y archiva antes el resultado previo.
Los resultados task conservan los campos comunes de identidad, por lo que `ResultCatalog` y
`results` auditan ambas familias sin otra fuente de verdad.

Cada intento task ejecuta localmente y en secuencial. Compón varios documentos task/experiment con
`kind: workflow`: el runner limita nodos listos y cada nodo conserva este `TaskExecutionPlan`,
fingerprint y resume. Los barridos de entrenamiento permiten además slots de proceso CPU/GPU.

## Plugins y seguridad

Las tareas reutilizables pueden publicarse bajo `lambdaforge.tasks` y seleccionarse así:

```yaml
task:
  plugin: {kind: task, name: surface_builder}
  params: {resolution: 1.0}
```

Los plugins task deben heredar `lambdaforge.tasks.Task`; los targets admiten duck typing. Ambos
mecanismos importan Python de confianza y no son un sandbox: no ejecutes YAML no confiable.
