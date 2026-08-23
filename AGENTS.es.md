# Guía de LambdaForge para agentes

Este fichero es la entrada de bajo coste para usar o modificar LambdaForge 0.12.0. Consulta solo la
sección necesaria de `docs/MANUAL.md` y después la firma, docstring o implementación concreta.

## Arquitectura no negociable

Un YAML ejecuta exclusivamente una o más clases que heredan `lambdaforge.Work`. `Work.run(...)` es
la única entrada científica. El framework crea la instancia, inyecta el runtime y envuelve el
código del usuario; el usuario nunca construye contextos, IDs, fingerprints ni rutas internas.

No introduzcas otra abstracción ejecutable, funciones como targets YAML, tipos `kind`, selección de
constructor/método, grafos arbitrarios, construcción recursiva `target/ref/params`, APIs globales de
runtime ni rutas de compatibilidad. No conviertas el YAML actual en una fachada de otro runner.

## Rutas rápidas

| Necesidad | Comando |
|---|---|
| Validar | `lf validate CONFIG` |
| Explicar firma y recursos | `lf explain CONFIG` |
| Plan sin efectos | `lf run CONFIG --dry-run` |
| Ejecutar | `lf run CONFIG [--on CLUSTER]` |
| Nueva Execution deliberada | `lf run CONFIG --rerun` |
| Monitorizar | `lf top`; `lf overview --json` |
| Operar Work | `lf show/logs/cancel/retry/delete SELECTOR` |
| Diagnóstico de bajo nivel | `lf jobs ...`; `lf doctor --on CLUSTER` |
| Datasets | `lf datasets list/show/verify/stats/members/diff/materialize/delete` |
| Resultados | `lf results list/show/compare` |
| Limpiar caché | `lf clean`; aplicar con `--apply` |

Usa `--json` para automatización y `--debug` solo para traceback interno. Un envío remoto devuelve
control tras crear el registro durable de preparación salvo que se solicite `--wait-for-submit`.

## Contrato Work

La firma y el docstring de `run()` son la verdad de parámetros. Los únicos marcadores especiales
son `{file: ...}`, `{dataset: NOMBRE@VERSION}` y `{from: PASO.SALIDA}`. Las vistas inmutables son
`name`, `config`, `inputs`, `resources`, `seed`, `trial`, `source_dir` y `resuming`. Los servicios
gestionados son `outputs.value/artifact/dataset`, `metrics.log/log_many`,
`checkpoints.path/exists/save_json/load_json`, `cache.path` y `progress.update`. `run_dir` es durable,
`temp_dir` efímero y `map(...)` es concurrencia interna acotada con checkpoints JSON.

`search` pasa sus variantes como parámetros normales y `objective` debe nombrar una métrica escalar
registrada. Si una variante tiene varias seeds, se ordena por su media, no por su mejor seed.

La jerarquía conceptual es Work → Execution → Run → Attempt → Job. La identidad científica incluye
clase, código consumidor, parámetros, hashes de ficheros, IDs de contenido de datasets, seed y
variante; excluye clúster, rutas, IDs operacionales y tiempo. Retry crea otro Attempt; resume usa
checkpoint compatible; rerun crea otra Execution.

Los datasets publicados son objetos durables independientes. Los resultados y checkpoints son
estado científico; bundles, entornos compartidos y caché son reconstruibles. Todo borrado debe ser
exacto, seguro frente a symlinks, idempotente y con vista previa. Nunca contactes un clúster real ni
modifiques datos científicos reales al probar el repositorio.

Antes de finalizar: actualiza manual, README, AGENTS, esquema, ejemplos y changelog; ejecuta pruebas
focalizadas, `ruff`, `mypy`, una suite local razonable, build de wheel y smoke instalado. Declara de
forma explícita cualquier test CUDA no ejecutado.
