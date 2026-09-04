# Guía de LambdaForge para agentes

Este fichero es la entrada de bajo coste para usar o modificar LambdaForge 0.14.0. Consulta solo la
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
| Operación interactiva | `lf` sin argumentos (Consola en TTY) |
| Monitorizar | Consola de investigación; `lf overview --json` |
| Operar Work | `lf show/logs/cancel/retry/delete SELECTOR`; Run: `show/logs WORK --run CLAVE` |
| Jobs de bajo nivel | `lf jobs list/show/logs/cancel/retry/delete/clear`; `lf doctor --on CLUSTER` |
| Datasets | `lf datasets list/show/verify/stats/members/diff/materialize/delete` |
| Resultados | `lf results list/show/compare/analyze/report` |
| Perfiles de clúster | Consola; `lf clusters add/set/unset/...` para automatización |
| Limpiar almacenamiento seguro | `lf clean`; aplicar con `--apply` |

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

`outputs.file/directory(..., publish_to=RUTA)` publica un resultado verificado tras el éxito y no
reemplaza contenido distinto salvo `overwrite=True`. Tras persistir la Execution elimina por
defecto los bytes internos redundantes; `retain_internal=True` conserva ambos. Un Attempt fallido o
interrumpido compacta `artifacts/`, pero mantiene logs, resultado, métricas, procedencia y
checkpoints. No pongas bulk desechable directamente en `run_dir`. Una ruta relativa parte del
directorio del YAML original. En remoto requiere el mirror `project_root` del clúster y se mapea al
mismo directorio relativo; sin él se usa una ruta remota absoluta explícita. No lo describas como
una transferencia automática al controlador.

Para inputs remotos `{file: RUTA}`, hasta 10 MiB se incluye automáticamente en el bundle. Una ruta
mayor debe pertenecer al proyecto local con `pyproject.toml` y tener una copia exacta bajo el
`project_root` absoluto del clúster; LambdaForge verifica tipo/tamaño/SHA-256 antes del envío y en
el worker, y nunca sincroniza ni elimina ese mirror del investigador. Configura con
`lf clusters set NOMBRE project_root /proyecto/remoto/absoluto` y después
`lf doctor --on NOMBRE`. Contenido ausente, viejo o con symlinks falla de forma segura. Para datos
muy grandes reutilizables usa datasets gestionados: verificar un mirror lee todos los bytes en cada
envío.

Las herramientas nativas se declaran una vez en `[tool.lambdaforge.environment]` con
`manager="conda"`, exactamente un `file` o `lockfile` contenido en el proyecto y nombres simples en
`required_executables`. Usa `lf clusters bootstrap NOMBRE --project . --dry-run`, revisa y aplica.
El YAML solo admite name/channels/dependencias string: no pip anidado, variables, prefix, hooks ni
Torch/CUDA. Offline exige lock `@EXPLICIT` por plataforma con SHA-256 y `package_cache` coincidente;
pip/Torch necesita además wheelhouse para ser totalmente offline. Se publica un único prefijo Conda
inmutable con inventario/tools verificados. `tools.require` comprueba, nunca instala. Pip-only y
`environment: existing` no cambian.

Las herramientas externas usan `tools.require(..., version_args=...)` y `tools.run(argv, ...)`.
Nunca uses strings de shell. Los límites de threads/env pertenecen al hijo, stdout/stderr llegan a
los logs y la procedencia se guarda exclusivamente en `environment.json`.

`print()` y el `logging` estándar aparecen en los logs del Job. Usa `self.log()` para narración
humana con fecha y vaciado inmediato, `self.progress.update()` para avance y `metrics.log()` para
evidencia numérica. La Consola presenta modelos de lectura acotados para Work, Studies y Results;
no coloques ficheros de resultado, llamadas al proveedor ni lógica científica en widgets. Los
agentes usan `lf overview --json`, `lf logs` y
`lf jobs clear [--apply]`, nunca parsean el TUI; `work.items[].attempt_history` conserva los IDs
para automatización. Las rutas del proveedor local pertenecen al Job
durable y no se recalculan desde el directorio actual del observador.
Los estudios de parámetros son Works ordinarios con `search` o varias `seeds`, no un tipo
específico de entreno. `work.items[].study_expected` existe antes de la telemetría de ejecución y
`study` deja de ser nulo cuando el worker actual publica su índice acotado. Los pasos, la
composición paralela y `self.map()` por sí solos siguen la navegación ordinaria Attempt/log y no
deben crear telemetría de estudio.

`search` pasa variantes como parámetros normales. `objective` es `metric`/`mode` heredado o una
utilidad `metrics` de rangos fijos con `weighted_mean`, `geometric` o `chebyshev`; sus componentes
comparten un step entero. Una utilidad gobierna HPO, las constraints son guardas duras separadas y
el frente Pareto crudo solo es diagnóstico. Toda
búsqueda con objective activa por defecto inicio Sobol scrambled, GP mixto qLogNEI opcional y sensible al ruido con
fallback k-NN mixto, carrera probabilística de seeds, pruning, convergencia y confirmación con seeds
nuevas. Solo se publican candidatos propuestos. `strategy: exhaustive` enumera exactamente
combinaciones finitas `values`/`when` y todas las seeds; rechaza `range` y `trials`. Las seeds de
confirmación son disjuntas; `confirmation_seeds: []` las desactiva. `search.fidelity` declara
presupuesto acumulativo definido por el Work; `self.fidelity` y checkpoints permiten continuar y
LightningRunner enlaza epochs. Decisiones/snapshot viven en `hpo-control/decisions.jsonl` y
`state.json`. Confirmación es inmune a pruning/preemption; un conjunto incompleto persiste
`confirmation_incomplete` y no selecciona una media solo de supervivientes. `trials` limita
candidatos ejecutados y `proposal_pool_size` el pool determinista
mayor. El refill por eventos compara `START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY` y
`RESUME_PREEMPTED` mediante un proxy auditable de valor del controlador por coste incremental
observado; nunca lo llames ganancia de información calibrada. Startup no es barrera, la cola no
despachada es provisional y registra `CANCEL_QUEUED_ACTION` al replanificar; las identidades
pendientes `(candidato, seed, fidelidad)` evitan duplicados. Se eliminan
`search.reduction_factor` y `search.confidence` raíz en favor de controles separados.
`objective.constraints.METRICA.min/max` declara guardas del mismo checkpoint. Entre seeds,
`seed_aggregation` es `mean` legacy, `worst` o `lcb` con `confidence`; evidencia ausente o LCB
insuficiente es no factible. Un componente de utilidad también puede ser constraint. Nunca infieras guardas o pesos
multiobjetivo ocultos desde otras métricas. `runs_per_gpu` empaqueta Runs spawn independientes dentro de una
reserva fija y valores >1 exigen `resources.gpu_memory` por Run. Es un umbral vivo de VRAM libre por
nuevo Run y `runs_per_gpu` solo un máximo: GPU llenas esperan y se sondean, las demás continúan y
los lanzamientos sobre una GPU se escalonan. No multipliques el umbral por los Runs activos ni
mantengas una segunda reserva oculta. Solo se rechaza si ninguna GPU asignada tiene VRAM
total suficiente. Cada Run GPU admitido usa un proceso spawn nuevo de un único worker y termina al
recibir resultado/error; no restaures pools CUDA persistentes porque sus contextos ociosos retienen
VRAM y pueden bloquear la cola. El probe de memoria GPU debe seguir siendo un hijo efímero: el
controlador no debe retener un contexto CUDA por dispositivo ni consumir una plaza científica. Un
fallo transitorio del probe pausa la admisión y conserva las Runs activas; no debe cancelar todo el
Study ni admitir desde memoria obsoleta. Para early stopping registra la
métrica repetida con `step=`; `LightningRunner` lo enlaza, y un loop propio retorna en un límite
seguro al detectar `self.stop_requested`.

Cada Run adaptativo CPU/GPU posee proceso nuevo de un worker. Un worker perdido/matado o una OOM
CUDA puede reintentarse como Attempt compatible con checkpoints hasta `failure_retries` (1 por
defecto, máximo 3); si se repite queda terminal. Una OOM CUDA reduce el límite de packing en memoria
por debajo de la concurrencia observada antes de readmitir el reintento; no mates Runs hermanos
sanos ni reintentes sin límite. No reintentes excepciones consumidoras arbitrarias.
Un Run agotado deja el Work honestamente fallido, pero no cancela Runs ajenos activos/en cola. La
telemetría guarda índice GPU lógico y token heredado exacto; nunca infieras ni amplíes dispositivos
físicos desde el campo de visualización.

Agrupa evidencia terminada por rung exacto `(candidato,target,maximum)`; nunca promedies fidelidades
heterogéneas ni enfrentes rungs incompatibles. la Consola de investigación separa diagnósticos marginales/por pares del
snapshot `.surrogate_belief` emitido por el sampler real.

La telemetría de estudio es un modelo de lectura acotado, no otro almacén de resultados. Referencia
logs y JSONL escalares por Run, nunca copia checkpoints/outputs, y expone claves exactas en
`overview --json` → `work.items[].study`. `lf show WORK --run CLAVE --json` devuelve parámetros,
curvas reducidas, objective y época actual/óptima, tiempos/fallo/log; `lf logs WORK --run CLAVE`
aísla la salida. HPO terminado usa la media del mejor checkpoint de cada seed; las curvas actuales
al mismo step deciden pruning y un Run podado es evidencia terminal censurada, no fallo, y se
excluye como valor exacto del ajuste/estadística de objetivos completos. Sus tasas de poda por
región siguen visibles. Cualquier seed podada censura el candidato entero; nunca promedies sus seeds
terminadas como evidencia solo de supervivientes. Un modelo conjunto de supervivencia por candidato con incertidumbre
modifica suavemente adquisición; varias seeds no sobrecuentan y fallos operacionales/preemption son
neutrales. Por defecto el pruning exige dos steps
comunes desfavorables distintos mediante `early_stopping.confirmations`. Lightning publica
automáticamente escalares de callback y tiempos de época/validación. Un trainer propio registra
curvas con `self.metrics.log(nombre, valor, step=epoch)`; `progress.update` es progreso grueso y
`self.log`/print solo narración humana.

El controlador solo puede preemptar cooperativamente un Run de fidelidad puntuado, no de
confirmación, con checkpoint propio verificado, al menos 30 segundos de ejecución y una alternativa
que supere en 50 % las prioridades original y de continuación. Nunca mata por planificación.
Conserva la evidencia `PREEMPT` → `PAUSE` → `RESUME_PREEMPTED`; startup sin score comparable queda
protegido y una parada que llega tras alcanzar el target sigue siendo `completed`.

La telemetría adaptativa incluye `hpo_analysis` acotado y las últimas 25 acciones estructuradas del
`controller`. La pantalla Studies lo consume y automatización lee
`work.items[].study.hpo_analysis/controller`. Trata cada panel por parámetro como asociación
marginal con cobertura/confianza/advertencias, nunca causalidad ni sustituto del sampler conjunto
multivariable. Enter/derecha abre la curva de respuesta y el panel acotado de ganancia predictiva
conjunta; respuesta/cobertura descriptiva empiezan con dos observaciones comparables y la ganancia
predictiva con tres, siempre con suelos conservadores de confianza. El JSON incluye puntos de
respuesta, señal de poda y matriz acotada.
La calidad retrospectiva del pruner vive en `hpo-control/state.json` → `pruner_calibration`: informa
ahorro simulado, falsos prunes, regret y calibración probabilística/de curva sin inventar un
objective completo para Runs censurados.

El runtime científico nunca depende del renderizado; el widget base `textual-plot` de la Consola
solo visualiza telemetría acotada. Studies nombra objective, propuestos/planificados, GPU y época
última/óptima mediante telemetría acotada.
Lightning elige curvas visibles con `LightningTrainConfig(epoch_chart_include=[...],
epoch_chart_exclude=[...])` sin perder las restantes. `gpu_mem_mb` es el pico vivo;
`gpu_reserved_mb`/`gpu_peak_reserved_mb` diagnostican la caché del allocator. Nunca conviertas esa
caché reservada en petición del scheduler: el packing usa `resources.gpu_memory` explícito y memoria
libre del driver para admisión dinámica por Run, y cada Run empaquetado termina su proceso y libera
el contexto CUDA completo antes de reutilizar slot. La presión temporal espera y se sondea; no es fallo científico.

La jerarquía conceptual es Work → Execution → Run → Attempt → Job. La identidad científica incluye
clase, código consumidor, parámetros, hashes de ficheros, IDs de contenido de datasets, seed y
variante; excluye clúster, rutas, IDs operacionales y tiempo. Retry crea otro Attempt; resume usa
checkpoint compatible; rerun crea otra Execution.

`gpu_access.mode` es `auto|scheduler|exclusive|shared|command`: auto elige scheduler en SLURM y
leases exclusivos en hosts directos; shared admite ocupación externa solo por decisión explícita;
command exige `command_prefix` argv del centro, nunca shell; prefiere wrappers autocontenidos como
`gpu exec`. `claim_command`/`release_command` son una pareja atómica, solo expanden `{gpu_count}` y
no son válidos con SLURM. En command/scheduler, `CUDA_VISIBLE_DEVICES` heredado son grants opacos:
nunca se sustituyen ni amplían. Una asignación ausente/duplicada falla cerrada; scheduler es exacto,
mientras un Study adaptativo tras un launcher command puede reducirse a menos tokens heredados.
La Consola edita el mismo catálogo y servicio de credenciales que los comandos nativos y nunca
invoca `lf` por subprocess. Direct frente a SLURM decide cómo se lanza el proceso; wrappers/claims
GPU son la política `gpu_access` separada, por lo que `gpu exec` normalmente usa
Direct. Los entornos gestionados obsoletos se podan tras activar reemplazo verificado y proteger
Jobs vivos.
La modal Add/Edit debe conservar el `ClusterProfile` durable completo: campos guiados para identidad,
runtime, almacenamiento, SSH y GPU habituales, más YAML Advanced validado para mapas OpenSSH/SLURM
menos comunes. Antes de persistir atómicamente se revalida con `ClusterProfile.from_mapping`. Nunca
entra una contraseña en editor/perfil; solo su referencia segura, dejando el secreto al servicio de
credenciales.

Cancelar un Work debe intentar todos sus Jobs activos. La cancelación directa termina y verifica el
conjunto completo de procesos propios, incluidos workers reparentados/con sesión nueva identificados
por el marcador exacto heredado; la salida normal del proceso principal aplica la misma limpieza.
Cancelar Job/Attempt conserva alcance estrecho. Nunca permitas que overrides del consumidor
sustituyan marcadores de ownership del framework.

Antes de añadir un modelo consulta `lambdaforge.nn.models` y la sección 14 del manual. Ya existen
familias MLP/CNN, grafos/equivariantes, secuencias/Transformer/Conformer, conjuntos, tabular, visión,
composición, generativos, científicos/implícitos y árboles diferenciables. No añadas un `GNN`
genérico, alias/factory redundante ni policy de dominio: solo un primitivo reutilizable con contrato
tensorial preciso y pruebas focalizadas.

## Análisis de estudios y consola

`lambdaforge.analysis.StudyAnalysis` analiza evidencia a posteriori; nunca es otro controlador HPO.
Los estudios terminales persisten `analysis.json` versionado y atómico; `lf results analyze SELECTOR
[--recompute] [--json]` usa el mismo núcleo y `results report` solo añade Plotly offline opcional.
Mantén separados `current_observed_objective`, `best_observed_objective`, `final_objective` y
`selection_objective`. Una Run podada es evidencia parcial censurada y no recibe score final
inventado; los componentes ausentes son estructurados.

Con el mismo fingerprint, el análisis es determinista y separa screening/confirmación. Una seed
líder no basta para estabilidad empírica; la incertidumbre modelada va aparte. Usa un único orden
min/max, metadata del CV realmente calculado, predicados `when` exactos y validación de todo punto.
La fiabilidad combina soporte, calidad CV, cobertura y extrapolación. Indica si la región superior
es observada/predicha y no atribuyas resolución a un pool no persistido. Confirmación usa el margen
de equivalencia. Separa coste intrínseco por Run de gasto total del controlador y Pareto científico
de Pareto de recursos. Todo sigue siendo predictivo/descriptivo, no causal.

`trials` adaptativo es el presupuesto de candidatos y se consume por defecto; cobertura escasa o
confianza baja del análisis no equivalen a convergencia. La convergencia por racha solo es opt-in
con `convergence_patience` positivo (cero/omitido la desactiva), y el snapshot terminal debe
persistir FINISH y su motivo exacto de presupuesto/pool/convergencia antes del análisis final.
Conserva una sola vez el pool determinista en el estado de planificación. Cada especificación Run
contiene únicamente sus valores candidato/seed y una definición compacta; nunca copies el pool
completo en cada una, porque un Study grande válido debe usar memoria lineal.

`lf` sin argumentos abre la Consola Textual solo en un TTY; sin TTY o con `--json` imprime ayuda.
Overview separa Cluster, Work ordinario y Study, conserva historial acotado de recursos y no vuelca
JSON crudo. Las gráficas de recursos y curvas de aprendizaje usan el widget nativo declarado
`textual-plot`: no añadas otro renderer ni persistas las muestras acotadas de la TUI. El historial
de recursos usa una ventana fija de cinco minutos y una clave de color externa. Un refresh debe
conservar panel y clave estable. Clusters
y Datasets usan tarjetas/secciones semánticas, no JSON de proveedor. Un Work terminal con
`study_expected` sigue en Studies y usa `study_job_id` para leer el
Attempt más reciente con telemetría. Las seis pantallas son Overview, Work, Studies, Clusters,
Datasets y Results. La navegación real es
`Study -> Trial -> Seed -> Epoch`: Enter/derecha apila y Esc/izquierda o Back desapila. Las migas
anteriores actúan sobre la misma pila; una lectura remota de Study/Seed muestra carga/error y no
tablas vacías como evidencia. Seeds conserva
current/best/final/selection, pruning censurado, páginas de cuatro gráficas seleccionables por ratón
o `n`/`p`, tabla horizontal con todos los escalares por step, GPU/recursos y
logs aislados; Analysis consume el JSON persistido. `Ctrl+P` solo anuncia handlers reales: un nombre
no equivale a paridad. Los widgets llaman servicios Python, el I/O lento sale del event loop y los
refresh conservan selección/época/scroll. Las acciones destructivas mantienen preview/apply. `lf top`,
`lf clusters setup` y `lf clusters modify` están retirados; no restaures otra UI.
Trata `study: null` como evidencia no disponible: abre Study/logs e intenta una recarga acotada, sin
convertirlo directamente ni degradarlo a Work ordinario.
Cancelar un Study es una operación sobre el Work semántico y sigue disponible antes de existir
telemetría. Los fallos recurrentes de refresh/proveedor se muestran inline como estado
obsoleto/error; nunca producen un aviso emergente en cada intervalo de sondeo.
HPO consume ese mismo análisis: dominios declarados/observados/prometedores, fiabilidad, respuesta
con incertidumbre/soporte, dispersión, interacciones e historial completo Acción/Trial/Motivo. No
ajusta otro modelo ni afirma causalidad. Antes de existir un resultado de Execution usa
`study.hpo_analysis` vivo y acotado mediante el servicio compartido; no ocultes evidencia por faltar
Results finales. Cambiar de página de métricas restablece límites automáticos; el zoom manual solo
afecta a la página visible. Las categorías conservan sus etiquetas reales y pulsar un punto/barra
muestra su valor exacto. La terminal mantiene la incertidumbre numérica sin pseudobandas engañosas;
exportaciones Plotly opcionales explícitas pueden mostrar intervalos reales, hover, mapas de calor
y superficies 3D. La cobertura usa tarjetas/gráficas/tablas acotadas, no JSON crudo, y la ayuda
contextual explica términos estadísticos. Las decisiones se añaden a
`study/controller-history.jsonl`: conserva `controller.json.recent` acotado, carga el historial
completo solo bajo demanda y mantiene el tail disponible en Studies antiguos sin ese fichero. Las
claves de métrica siguen estables; las etiquetas pueden
configurarse con `LightningTrainConfig.epoch_metric_display_names`, y la UI nunca expone
`__lambdaforge_utility__` en vez de «Composite selection score». Las marcas de trial van en columna
separada.

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

Los logs de Attempts en la Consola de investigación se actualizan automáticamente. Un Work fallido añade tras cualquier
tail el tipo/mensaje/fase/ruta del resultado persistido; usa `--verbose` o `--debug` para traceback y
`e` en la TUI, o `--json` para `failure`/`failures` estructurados. Un estudio terminal sin telemetría
publicada cae a Attempts/logs numerados. Los modelos inmutables que cruzan procesos spawn deben ser
pickle-safe; nunca expongas un `mappingproxy` crudo en esa frontera. `cache.fetch` reintenta cortes HTTP/chunked/gzip y
estados transitorios desde temporales no publicados limpios; no añadas workarounds en el consumidor.
El progreso humano de bootstrap solo usa stderr y no contamina JSON.
La Consola usa los mismos servicios de dominio en workers, comparte un factory de proveedores,
serializa operaciones de clúster y pausa probes de recursos durante bootstrap. Su panel de actividad
muestra fases/tiempo y admite alturas predefinidas o arrastre; el viewport superior conserva altura
mínima y scroll. Las confirmaciones de mutación y planes modales muestran secciones semánticas
acotadas, nunca JSON crudo. Las pantallas raíz ocultas no cargan.
El indicador de carga raíz solo aparece hasta el primer snapshot correcto. Los refresh posteriores
conservan el último read model, muestran su antigüedad y lo marcan obsoleto si fallan; filas y logs
acotados de Work se actualizan en vivo con una sola petición activa por vista.
El selector de Run Work filtra YAML; recientes combina el historial local de Jobs con un MRU acotado
que solo persiste ruta/nombre/fecha locales. Explorar/elegir reciente solo rellena un selector. La
única acción Submit debe volver a validar, explicar y solo después enviar asíncronamente al clúster
visible capturado; bloquea repeticiones y muestra cada fase. Nunca trates la presencia en MRU como
validación; representar el preview es solo diagnóstico y nunca puede impedir un envío ya validado.
No copies la configuración al estado de consola. Los nombres visibles de Work no son
claves: usa identidades `work_id` exactas para filas y mutaciones, de forma que ejecuciones
homónimas simultáneas sean independientes.
Los fallos sintácticos YAML deben ser acotados y ligados a la fuente: ruta, línea/columna, fragmento,
cursor e indicación una sola vez.
Summary de Dataset puede leer el índice lógico pero nunca recorrer assets; Stats/Integrity son
explícitos. El borrado de versión previsualiza todas las ubicaciones, confirma una vez y trata
`registered_but_missing` como convergencia segura antes de olvidar registros vacíos. El control
permanece visible y debe mostrar inmediatamente espera/éxito/fallo evitando activaciones duplicadas.
