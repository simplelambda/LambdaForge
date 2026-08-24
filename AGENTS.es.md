# Guía de LambdaForge para agentes

Este fichero es la entrada de bajo coste para usar o modificar LambdaForge 0.12.0. Consulta solo la
sección necesaria de `docs/MANUAL.es.md` y después la firma, docstring o implementación concreta.

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
| Jobs de bajo nivel | `lf jobs list/show/logs/cancel/retry/delete/clear`; `lf doctor --on CLUSTER` |
| Datasets | `lf datasets list/show/verify/stats/members/diff/materialize/delete` |
| Resultados | `lf results list/show/compare` |
| Limpiar caché | `lf clean`; aplicar con `--apply` |

Usa `--json` para automatización y `--debug` solo para traceback interno. Los envíos locales y
remotos devuelven control tras crear el registro durable de preparación salvo que se solicite
`--wait-for-submit`; `--dry-run` es directo y sin efectos.

## Contrato Work

La firma y el docstring de `run()` son la verdad de parámetros. Los únicos marcadores especiales
son `{file: ...}`, `{dataset: NOMBRE@VERSION}` y `{from: PASO.SALIDA}`. Las vistas inmutables son
`name`, `config`, `inputs`, `resources`, `seed`, `trial`, `source_dir` y `resuming`. Los servicios
gestionados son `outputs.file/directory/value/dataset`, `metrics.log/log_many`,
`checkpoints.file/exists/save_json/load_json`, `cache.put/get/file/fetch/rate_limit`, `tools.require/run`,
`progress.update` y
`log(message, level=...)`. `run_dir` es durable, `temp_dir` efímero y `map(...)` es concurrencia
interna ordenada sin persistencia. `resume_map(..., key=...)` añade de forma explícita checkpoints
JSON que conocen dependencias; `map(..., key=...)` antiguo delega en esta operación.

Cache es reconstruible, checkpoints es estado de Run para resume y outputs es evidencia durable.
Un Work no gestiona directorios de cache, `.part`, locks, `fsync` ni `os.replace`: usa
`cache.put/get` para bytes/texto/JSON estricto, `cache.file/fetch` para contenido path-like,
checkpoint files y outputs gestionados. `cache.path`, `checkpoints.path`,
`run_dir` y `outputs.artifact` son escapes avanzados, no la vía recomendada. `ManagedFile` es
path-like, pero `resume_map` solo serializa key/SHA/tamaño; si `lf clean` elimina o corrompe una
dependencia, se recalcula únicamente su elemento. Nunca serialices rutas de máquina ni pickle.

`outputs.file/directory(..., publish_to=RUTA)` publica opcionalmente una copia verificada tras el
éxito y no reemplaza contenido distinto salvo `overwrite=True`. Una ruta relativa parte de
`source_dir`; en remoto sigue siendo remota. Usa almacenamiento persistente absoluto del clúster si
debe sobrevivir al Job y no lo describas como una transferencia automática al controlador.

Las herramientas externas usan `tools.require(..., version_args=...)` y `tools.run(argv, ...)`.
Nunca uses strings de shell. Los límites de threads/env pertenecen al hijo, stdout/stderr llegan a
los logs y la procedencia se guarda exclusivamente en `environment.json`.

`print()` y el `logging` estándar aparecen en los logs del Job. Usa `self.log()` para narración
humana con fecha y vaciado inmediato, `self.progress.update()` para avance y `metrics.log()` para
evidencia numérica. En `lf top`, Enter/derecha avanza de Work a Attempt numerado y logs o abre un
clúster; izquierda vuelve y la vista principal no muestra IDs largos. `d` borra una selección
terminal confirmada y `D` limpia el historial terminal
confirmado sin tocar Jobs activos. Los agentes usan `lf overview --json`, `lf logs` y
`lf jobs clear [--apply]`, nunca parsean el TUI; `work.items[].attempt_history` conserva los IDs
para automatización. Las rutas del proveedor local pertenecen al Job
durable y no se recalculan desde el directorio actual del observador.

`search` pasa sus variantes como parámetros normales y `objective` debe nombrar una métrica escalar
registrada. Si una variante tiene varias seeds, se ordena por su media, no por su mejor seed.

La jerarquía conceptual es Work → Execution → Run → Attempt → Job. La identidad científica incluye
clase, código consumidor, parámetros, hashes de ficheros, IDs de contenido de datasets, seed y
variante; excluye clúster, rutas, IDs operacionales y tiempo. Retry crea otro Attempt; resume usa
checkpoint compatible; rerun crea otra Execution.

Antes de añadir un modelo consulta `lambdaforge.nn.models` y la sección 14 del manual. Ya existen
familias MLP/CNN, grafos/equivariantes, secuencias/Transformer/Conformer, conjuntos, tabular, visión,
composición, generativos, científicos/implícitos y árboles diferenciables. No añadas un `GNN`
genérico, alias/factory redundante ni policy de dominio: solo un primitivo reutilizable con contrato
tensorial preciso y pruebas focalizadas.

## Contrato de clustering

Usa `lambdaforge.clustering`. `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `HDBSCAN` y `Agglomerative`
comparten `Clusterer.cluster(X) -> ClusteringResult`; sklearn es lazy y opcional mediante
`lambdaforge[clustering]`. No añadas factory/registro, estimadores backend ni otro tipo Distance.
Reutiliza `lambdaforge.nn.distances.Distance`, aplica la tabla de capacidades y comprueba memoria
antes de una matriz O(N²). KMeans y Ward son euclídeos. Escalado, imputación, PCA, parámetros,
thresholds e interpretación de estabilidad son ciencia explícita del proyecto.

Los datasets publicados son objetos durables independientes. Los resultados y checkpoints son
estado científico; bundles, entornos compartidos y caché son reconstruibles. Todo borrado debe ser
exacto, seguro frente a symlinks, idempotente y con vista previa. Nunca contactes un clúster real ni
modifiques datos científicos reales al probar el repositorio.

Antes de finalizar: actualiza ambos manuales, READMEs y AGENTS, esquema/ejemplos si aplica y
changelog; ejecuta pruebas
focalizadas, `ruff`, `mypy`, una suite local razonable, build de wheel y smoke instalado. Declara de
forma explícita cualquier test CUDA no ejecutado.
