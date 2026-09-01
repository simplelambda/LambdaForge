# Historial de cambios

Español · [English detallado](CHANGELOG.md)

Este historial español sigue Keep a Changelog y Semantic Versioning. Resume todas las versiones;
el fichero inglés enlazado conserva el inventario exhaustivo de cada corrección y cambio histórico.
El repositorio aún no tiene tags: 0.1.0 y 0.2.0 se reconstruyeron a partir de commits de versión y
metadata empaquetada.

## [Sin publicar]

## [0.13.3] - 2026-09-01

### Corregido

- Corregida la evidencia multi-fidelidad: resultados de seeds se agrupan por rung exacto
  `(candidato, target, maximum)`, un candidato puede generar varias observaciones y nunca se
  reetiqueta una media heterogénea como presupuesto completo.
- Eliminada la semántica residual de compromiso de cola/rondas. Cada evento replantea acciones aún
  no despachadas, registra `CANCEL_QUEUED_ACTION` con coste científico cero y rellena slots libres;
  una promoción espera evidencia comparable ya en vuelo antes de decidir prematuramente.
- Sustituida la falsa «información esperada» por un `controller_value` centralizado, acotado y
  documentado, dividido por coste incremental observado.
- Corregida la utilidad compuesta viva para combinar componentes del mismo step y conservar el
  mejor checkpoint histórico aunque salga del tail acotado.
- Permitido usar una métrica como componente y constraint; añadida agregación entre seeds `mean`,
  `worst` y `lcb` con evidencia ausente/insuficiente fail-closed.
- Centralizado el ajuste de supervivencia y publicados sus coeficientes. Pruning sigue siendo
  evidencia censurada; fallos operacionales y preemption siguen neutrales.
- Separados en `lf top`/JSON los diagnósticos marginales descriptivos del `surrogate_belief` real
  GP/k-NN, conservando el último snapshot ligero sin serializar modelos.
- Corregida la ruta CI del provider opcional y añadidas pruebas BoTorch obligatorias sin skip para
  multi-fidelidad numérica, espacio mixto/condicional, pendientes, ruido, predicción a fidelidad
  objetivo y ajuste de supervivencia.

### Cambiado

- Documentada con precisión la arquitectura como surrogate sensible a fidelidad más scheduler
  externo sensible a coste, junto con rungs exactos, cola provisional, constraints robustas,
  métricas dependientes de threshold y calibración retrospectiva del pruner.
- Ampliadas regresiones end-to-end para runner→observaciones→samplers, sustitución de cola, fallback
  controlado, utilidad compuesta viva, surrogate retenido y falso prune deliberado de late bloomer.

## [0.13.2] - 2026-09-01

### Corregido

- Unificado HPO adaptativo alrededor de una única utilidad científica alineada por checkpoint. El
  objetivo puede ser escalar o compuesto con rangos fijos y agregación ponderada/geométrica/
  Chebyshev; las constraints siguen separadas y nunca se mezclan componentes de épocas distintas.
- Rehecho el pruning a nivel de candidato usando seeds activas e históricas, incumbents terminados,
  incertidumbre conservadora, calibración retrospectiva, umbrales separados y confirmación inmune.
- Sustituida la penalización fija de censura por un modelo conjunto de supervivencia con intervalos;
  pruning aporta evidencia censurada, pero recursos, infraestructura y preemption son neutrales.
- Eliminadas barreras de ronda/startup: cada Run terminal puede elegir `START_NEW`, `ADD_SEED` o
  `PROMOTE_FIDELITY` por información esperada/coste, evitando seeds duplicadas y auditando alternativas.
- Añadida preemption conservadora y segura por checkpoint para Runs de fidelidad ya puntuados. Un
  suelo de 30 segundos, histéresis del 50 %, protección de confirmación/startup y parada cooperativa
  evitan thrashing; `PREEMPT`, `PAUSE`, `CONTINUE` y `RESUME_PREEMPTED` quedan auditados y neutrales.
- Separados presupuesto ejecutado `trials` y `proposal_pool_size`, incorporada fidelidad al modelo,
  rechazados knobs raíz ambiguos/no-op y añadidos terminaciones, confirmación incompleta y costes.
- Ampliados `lf top`/overview con componentes, Pareto diagnóstico, incertidumbre de poda por
  candidato, slots/acciones del scheduler y la probabilidad/referencia exactas que justificaron
  cada poda. Añadidos informes retrospectivos con ahorro simulado, falsos prunes, regret y
  calibración probabilística/de curva.
- Hecha útil la evidencia HPO temprana: respuesta marginal y cobertura por pares se muestran desde
  dos candidatos comparables, la ganancia predictiva empieza con tres manteniendo confianza baja y
  un `lf top` nuevo reconstruye localmente análisis remotos antiguos. Los Runs podados aportan tasas
  visibles por región y evitación suave de vecindad censurada sin fingir objetivos completos.
- Eliminados los contextos CUDA persistentes por GPU del probe del controlador HPO. La VRAM libre
  se observa en un hijo efímero que hereda el grant, por lo que monitorizar no ocupa una plaza
  científica ni retiene memoria que impida admitir el último Run de `runs_per_gpu`.
- Un Trial con todas sus seeds podadas queda ahora terminal `pruned`, no `failed`; sus curvas
  parciales se conservan como evidencia censurada sin contaminar estadísticas de objetivos
  completos, y streams métricos duplicados ya no mezclan objective actual y óptimo terminal.
- Reservadas al menos tres filas de seed con alturas comunes de terminal acotando previews de
  parámetros/métricas; el detalle completo queda a un nivel de navegación.
- Separada la evidencia actual y óptima en todo estudio adaptativo. HPO terminado clasifica ahora
  la media del mejor checkpoint de cada seed mientras las curvas actuales al mismo step siguen
  decidiendo pruning; `lf top` muestra ambos valores y la seed/época ganadora sin mezclarlos.
- Aclarado que Direct/SLURM decide cómo lanzar procesos, no el tipo de GPU, y permitida salida
  limpia desde cada pregunta del asistente mediante `0`, `q`, `quit` o `exit`.
- Hechos pickle-safe los metadatos inmutables de resultados/artefactos al cruzar workers spawn. Un
  Run completado con artefactos ya no puede fallar después con `cannot pickle 'mappingproxy'`; esa
  firma se clasifica además como fallo interno y no como configuración del usuario.
- Los estudios terminales sin índice de telemetría caen ahora a Attempts numerados y logs normales
  en `lf top`, conservando el fallo real en vez de una pantalla de datos no disponibles.
- Aislado cada Run adaptativo CPU/GPU en su propio worker: un proceso matado ya no rompe un pool
  compartido ni aborta entrenos ajenos. Workers perdidos y OOM CUDA tienen reintento acotado con
  checkpoints; si el fallo se repite queda como evidencia terminal mientras los demás continúan.
- Sustituida la comprobación fatal global `runs_per_gpu × gpu_memory` por admisión dinámica por
  Run. Las GPU temporalmente ocupadas esperan y se sondean, las disponibles continúan a capacidad
  parcial, los lanzamientos en el mismo dispositivo se escalonan y solo falla como configuración un
  umbral físicamente imposible en todas las GPU asignadas.
- Evitado el deadlock tras terminar un Run GPU: cada Run empaquetado usa ahora un proceso nuevo de
  un worker que termina con resultado o error y libera su contexto CUDA, en vez de dejar pools
  ociosos reteniendo VRAM mientras la cola espera indefinidamente.
- Evitada la carrera entre el refresco del estudio y el cierre del worker que podía devolver un Run
  terminal a `running`: las observaciones de mejor época del controlador tienen ahora un registro
  atómico separado, por lo que los contadores finales y la admisión no quedan bloqueados.
- Corregidos `lf help`, ayuda anidada en forma natural y `--help` convencional para terminar con
  éxito, también cuando otra aplicación llama directamente al entry point.

### Añadido

- Guardas de resultado explícitas mediante `objective.constraints`: se evalúan en la mejor época
  del objetivo primario y se promedian entre seeds; evidencia ausente/incumplida hace al candidato
  no factible y lo excluye del ajuste/selección sin ocultar su registro.
- Relleno de adquisición asíncrono acotado: uno o dos candidatos de anticipación del posterior
  actualizado pueden ocupar recursos antes de terminar un lote lento, qLogNEI condiciona en todos
  los pendientes y cada decisión queda auditada sin cancelación especulativa masiva.
- Detalle visual por hiperparámetro en `lf top` con curvas de respuesta, barras categóricas y tabla
  de calor de ganancia predictiva conjunta. Puntos y matriz acotados también aparecen en JSON con
  semántica explícitamente no causal.
- Añadida una consola de evidencia HPO con `i` en estudios adaptativos de `lf top`. Explica por
  parámetro dirección/posible umbral numérico o contraste categórico con cobertura, efecto,
  confianza conservadora, sugerencia de siguiente evidencia y última acción real del controlador.
  El mismo modelo acotado y explícitamente no causal está disponible en el JSON de overview.
- Toda `search` con objective activa ahora por defecto la política adaptativa completa: inicio Sobol
  scrambled, adquisición mixta dependiente de resultados, asignación probabilística de seeds
  compartidas, pruning conservador, límites de convergencia y confirmación con seeds nuevas.
  BoTorch qLogNEI mixto y sensible al ruido sigue opcional y aislado, con fallback k-NN determinista.
- Añadida promoción acumulativa explícita `search.fidelity` mediante `self.fidelity` y checkpoints
  gestionados; Lightning reanuda automáticamente presupuestos crecientes de epochs y la
  confirmación final usa fidelidad completa.
- Añadida evidencia compacta de replay/auditoría en `hpo-control/state.json` y el diario append-only
  `decisions.jsonl`, con backend/fallback, `START_NEW`, `ADD_SEED`, `RESUME`, parada y confirmación.
  Los resúmenes enlazan estos registros e informan incertidumbre/fidelidad.
- `strategy: exhaustive` es ahora un contrato de sweep exacto para `values` finitos y ramas `when`;
  se rechazan rangos continuos y límites de candidatos en vez de muestrearlos silenciosamente.
- Evidencia ampliable para Runs fallidos en `lf top`: `e` alterna el traceback persistido tanto en
  paneles de seed como en logs de Attempts, incluida fase y ruta exacta del resultado.
- Propuesta secuencial dependiente de resultados: un conjunto inicial que cubre el espacio precede
  lotes de adquisición k-NN numérica/categórica, y `lf top` solo publica parámetros al proponer
  realmente el Trial en vez de presentar todo el pool determinista como ya decidido.
- Procedencia de índice/token GPU por seed en resultados, detalle JSON y `lf top`.
- Sustituidos los sparklines de recursos/estudios de `lf top` por series temporales Unicode
  enmarcadas de alta resolución inspiradas en nvtop, sin dependencia nueva de plotting.
- Desplazamiento horizontal de logs largos y selección de curvas Lightning mediante
  `epoch_chart_include`/`epoch_chart_exclude`.
- Color semántico con soporte `NO_COLOR`, historiales compactos de clúster, parámetros completos del
  Trial seleccionado, páginas de cuatro curvas, tabla de épocas seleccionable, marcador en las
  curvas, detalle escalar por época y selector explícito de salida bruta.
- Simplificadas las tablas de estudio eliminando resúmenes truncados redundantes, añadiendo un panel
  completo de métricas de la seed seleccionada y usando `n`/`p` para páginas de curvas.
- Navegación de estudio consciente del objective: Trials nombran métrica/dirección, las seeds
  separan época última y óptima, y tablas/gráficas conservan marcador verde de óptimo junto al rojo
  seleccionado incluso tras reducir curvas.
- `lf clusters setup` y `lf clusters modify` como flujos de terminal explicados y ligeros sobre los
  comandos nativos existentes; cubren credenciales, rutas, runtimes, storage, scheduler y GPU sin
  duplicar el backend de configuración.
- Navegación con flechas/Enter y explicación enfocada de consecuencias/riesgos en las opciones del
  setup, manteniendo fallback numerado igual de documentado y salida segura en cada pregunta.
- Claim/release GPU argv atómicos por clúster. Se prefieren wrappers autocontenidos como CITIUS
  `gpu exec`; un claim persistente se libera tras éxito, fallo, cancelación o submit fallido y se
  rechaza con SLURM.

### Cambiado

- Las propuestas bayesianas usan ahora mejora esperada conjunta con ruido, error estándar entre
  seeds y miembros pendientes del lote. La confianza del panel HPO exige tamaños de muestra
  conservadores, separa explicación marginal de decisión conjunta e informa podas censuradas. El
  pruning requiere por defecto dos steps comunes desfavorables distintos.
- La telemetría distingue el pico de tensores CUDA vivos de la caché actual/máxima del allocator,
  prioriza tiempos de época/validación y libera caché CUDA no usada entre Runs empaquetados. La
  admisión HPO sigue usando memoria por Run explícita y memoria libre del driver, no la caché
  reservada observada.
- Los Runs activos muestran duración viva y tiempos medidos según llegan, usando una estimación
  media por época claramente etiquetada solo mientras no exista telemetría explícita.
- En command/scheduler, `CUDA_VISIBLE_DEVICES` heredado se trata como grants opacos del centro
  (incluidos UUID/MIG), solo se estrecha por hijo y se falla cerrado en vez de inventar o ampliar
  una asignación ausente/insuficiente. Los probes CUDA usan el mismo wrapper del centro.

## [0.13.0] - 2026-08-30

### Añadido

- Observabilidad viva y acotada de estudios: `lf top` avanza de Work a Trials de parámetros y Runs
  de seed con logs aislados autorrefrescados, curvas, resúmenes escalares/de tiempos y fallos.
  `overview --json`, `show WORK --run CLAVE --json` y `logs WORK --run CLAVE` exponen la misma
  lectura sin duplicar checkpoints ni outputs pesados; `LightningRunner` emite automáticamente
  curvas escalares y tiempos de época/validación.
- Modos de acceso GPU por clúster: reserva del scheduler, lease exclusivo directo, host compartido
  deliberado y wrapper de claim del centro expresado como argv.
- `retain_internal=True` para conservar expresamente una segunda copia física publicada,
  `retention.json` con bytes recuperados y `Work.stop_requested` para loops adaptativos propios;
  `LightningRunner` aplica automáticamente el mismo contrato.

### Cambiado

- Las búsquedas con objective y varias seeds son adaptativas por defecto: successive halving asigna
  seeds progresivamente con media/error estándar conservador y early stopping cooperativo por step;
  `strategy: exhaustive` conserva el estudio completo explícito.
- `runs_per_gpu` empaqueta procesos de entreno independientes por cada GPU de la reserva externa,
  exige límite de memoria por Run cuando es mayor que uno y reparte CPU/RAM/storage entre hijos.

### Corregido

- Corregido el cierre de `lf top` al deserializar listas inmutables anidadas del snapshot en segundo
  plano. La intención de estudio se persiste antes del envío, por lo que un Work con `search` o
  varias `seeds` abre el panel de estudio durante la preparación en vez de caer en el log
  del Attempt ordinario.
- Se deja de tratar como entreno cualquier Work con un fichero de telemetría o varios Runs de
  workflow. Solo búsquedas/seeds repetidas publican el índice de estudio; preprocesado, composición
  y Works con map abren coherentemente su log de Attempt normal desde la vista principal.
- La cancelación de Work pasa a ser jerárquica en vez de afectar solo al Job primario: intenta cada
  Job activo, el supervisor directo termina y verifica grupo más workers reparentados/con sesión
  nueva mediante la identidad heredada exacta, y la salida normal recoge hijos supervivientes.
- Los Jobs terminales ya no acumulan outputs gestionados parciales ni duplicados pesados:
  Attempts fallidos/interrumpidos compactan artefactos conservando evidencia ligera, y una copia
  interna publicada solo se elimina tras verificar hash y tamaño del destino; `lf clean` aplica la
  misma política, con vista previa, a Jobs históricos conservados.
- Los entornos gestionados sustituidos se podan después de bootstrap y de preparación automática,
  protegiendo el entorno activo y todas las referencias de Jobs vivos.

## [0.12.1] - 2026-08-25

### Añadido

- `project_root` remoto configurable mediante `clusters add --project-root` o `clusters set`, con
  comprobación en `doctor` y documentación de snapshots pequeños, mirrors compartidos y datasets.
- Progreso por fases y latidos periódicos en bootstrap humano, y refresco automático que conserva
  el scroll en los logs de Attempts de `lf top`; JSON máquina permanece limpio.

- `ManagedFile`, `Work.cache.file/fetch/rate_limit`, checkpoint files y
  `outputs.file/directory` con validación, huella y promoción atómica.
- Reanudación selectiva de `Work.map` mediante dependencias lógicas, keys por campo, validador y
  retries/backoff por elemento.
- `Work.tools.require/run` con argv sin shell, logs vivos acotados, control de threads del hijo,
  probes de versión y procedencia única en `environment.json`.
- Familia opcional `lambdaforge.clustering` con KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN,
  Agglomerative, `ClusteringResult`, compatibilidad Distance explícita, guard O(N²) y métricas de
  evidencia.
- Manual y política de seguridad completos en español, sincronizados con README y guía de agentes.
- `lf jobs clear [--apply]` y borrado individual definitivo de Job, incluyendo workspace, eventos y
  registro detached exactos.
- `Work.cache.put/get` para bytes, texto y JSON estricto con escritura atómica, sin rutas ni pickle.
- `publish_to`/`overwrite` opcional en outputs gestionados: el artefacto verificado sigue siendo la
  autoridad y se publica además una copia atómica local o del host remoto; se rechazan enlaces
  simbólicos, cambios de tipo y sustituir el propio directorio o uno que lo contiene.
- `attempt_history` en la vista máquina de Work para compartir Attempts numerados con TUI y wrappers.
- Dependencias nativas opcionales declaradas por proyecto: solve Conda exacto con micromamba
  verificado, identidad por plataforma/inventario, prefijo único inmutable, procedencia de
  ejecutables, `bootstrap --project` explicable y locks/cache offline SHA-256 sin afectar pip-only.

### Cambiado

- `lf run` local usa el mismo handoff durable y asíncrono que remoto; `lf top` navega con flechas
  adelante/atrás, usa `d` para borrar una selección terminal y `D` para limpiar historial terminal;
  `Work.log` emite mensajes con fecha y flush.
- Work cache es una categoría real de `lf clean`, coordinada con una lease del Work activo.
- Paths, fingerprints y escrituras atómicas se consolidan; dataset cache y Work cache usan
  directamente `CrossProcessFileLock`.
- `Work.map` queda como concurrencia ordenada sin persistencia oculta; `resume_map` expresa
  checkpoints por elemento y las llamadas antiguas con `key` siguen siendo compatibles.
- `lf top` navega de Work a Attempts numerados y logs sin mostrar IDs largos; `lf jobs` y
  `overview --json` conservan los identificadores para diagnóstico y automatización.
- El manual descubre el catálogo neuronal ya existente en vez de duplicarlo con aliases MLP/GNN o
  una factory adicional.
- La `storage.cache_root` configurada se propaga a procesos Work directos y planificados para que
  cache simple y de ficheros usen realmente la ubicación reutilizable/limpiable elegida.

### Corregido

- Las publicaciones `publish_to` relativas ya no terminan en workspaces de Job con hash: conservan
  el layout del YAML en local y en el mirror remoto persistente. Entradas tipadas mayores que el
  bundle pueden usar ese mirror, pero antes del scheduler deben coincidir exactamente en tipo,
  bytes y SHA-256; contenido ausente, viejo, parcial o con symlinks no ejecuta código científico.
- Las transferencias HTTP chunked/gzip truncadas, incluido `IncompleteRead`, se reintentan desde un
  temporal privado limpio en cada intento limitado; los HTTP permanentes fallan pronto y nunca se
  publican bytes o registros parciales.
- Los logs remotos, `show` y `lf top` recuperan el fallo científico estructurado persistido incluso
  con `--tail`; JSON/debug conserva la evidencia y evita duplicar un traceback ya visible.

- Se recupera el subdir de paquetes nativos instalados desde los registros regulares `conda-meta`
  cuando micromamba 2.8 lo omite en `list --json`, manteniendo verificación exacta de versión,
  build, canal y subdir antes de publicar, después de pip y al reutilizar. Los fallos muestran ahora
  un diff acotado por campos en vez del prefijo completo.
- Los nombres de paquete como `libssh2` ya no se confunden con fallos del transporte SSH; los errores
  tipados de preparación nativa tienen prioridad y se clasifican como entorno.
- El supervisor SSH ya no copia un workspace staged sobre sí mismo y crea stdout/stderr antes de
  lanzar el cálculo; logs antiguos sin stream muestran la causa durable.
- Los Jobs locales ya no pasan a `unknown` por resolver storage relativo desde distintos
  directorios: los nuevos guardan raíces absolutas y los antiguos se recuperan desde su YAML fuente.

### Eliminado

- La subclase vacía `CacheFileLock`; el lock común es la única implementación.

## [0.12.0] - 2026-08-23

`lambdaforge.Work` y `Work.run()` pasan a ser el único contrato ejecutable. YAML queda reducido a
parámetros normales, entradas tipadas, secuencia/paralelismo, seeds y search; se añaden resultados
Execution/Run/Attempt, logs, métricas, outputs, datasets, fingerprints, retry/resume y Workspaces
remotos propios. Se eliminan runners, Tasks y campos de compatibilidad anteriores.

## [0.11.0] - 2026-08-22

Introdujo authoring function-first, marcadores file/dataset, runtime global, búsqueda simple,
operaciones semánticas `show/delete/clean` y publicación streaming, como paso previo al contrato Work
único. Se reforzó ownership de artefactos y borrado exacto.

## [0.10.1] - 2026-08-21

Añadió reconciliación segura de placement de datasets, estados físicos explícitos, discovery
acotado y eliminación idempotente verificada contra manifest e identidad.

## [0.10.0] - 2026-08-21

Consolidó la UX de investigación alrededor de revisiones, Runs y una vista semántica de Work en
overview/top, con agrupación y comparación de resultados.

## [0.9.2] - 2026-08-21

Persistió eventos detallados del ciclo de vida de Jobs y progreso de workflows/preprocessing;
mejoró logs y diagnóstico de preparación frente a progreso científico.

## [0.9.1] - 2026-08-20

Hizo `lf top` responsivo frente a proveedores lentos, limpió entornos superseded tras bootstrap y
centralizó la versión de release.

## [0.9.0] - 2026-08-20

Introdujo envío remoto asíncrono durable y el primer `lf top` interactivo con recursos de clúster,
Jobs seleccionables y estado observable por comandos máquina.

## [0.8.1] - 2026-08-17

Corrigió el lanzamiento remoto para usar el Python exacto del entorno gestionado y no interpretar
incorrectamente `-m` mediante implementaciones antiguas de `env`.

## [0.8.0] - 2026-08-15

Unificó diagnósticos, categorías, códigos de salida, JSON/debug y registros locales redactados para
fallos de configuración, entorno, ejecución, datos, recursos y seguridad.

## [0.7.2] - 2026-08-14

Añadió resolución `auto/existing/managed` de Python remoto, reutilización Conda y fallback
micromamba verificado en espacio de usuario para clústeres con Python antiguo.

## [0.7.1] - 2026-08-14

Corrigió bootstrap desde instalaciones editables o wheel sin buscar `pyproject.toml` dentro del
venv consumidor y consolidó documentación y tests de comportamiento.

## [0.7.0] - 2026-08-14

Incorporó lifecycle completo de datasets: recetas por etapas, índices JSONL, miembros/assets
genéricos, contenido inmutable, placements, materialización, verificación y CLI de inspección.

## [0.6.0] - 2026-08-13

Añadió acciones post-run, control de Jobs directo/SLURM, recursos, logs durables, monitorización,
cancelación segura y multiplexación SSH con fronteras Transport/Scheduler.

## [0.5.3] - 2026-08-13

Implementó selección de PyTorch/CUDA a partir de Python, driver y compute capability reales y
políticas `pytorch.channel/require_cuda`, sin modificar drivers ni toolkit del sistema.

## [0.5.2] - 2026-08-12

Introdujo catálogos de clúster por capas, credenciales password seguras mediante prompt/keyring/env
y personalización por clúster de scheduler, storage y bootstrap.

## [0.5.1] - 2026-08-11

Estabilizó preprocessing real con modos secuencial/I/O/CPU/GPU, aliases de configuración,
validación de outputs y runtime multi-clúster más completo.

## [0.5.0] - 2026-08-11

Añadió configuración amigable compilada a contratos estrictos, TaskContext, inputs/outputs con
nombre, control plane multi-clúster y una CLI más sencilla para principiantes.

## [0.4.1] - 2026-08-10

Endureció HPO adaptativo con surrogate mixto multi-fidelity, curvas bayesianas, incertidumbre de
seeds, memoria explícita y decisiones de pruning/admisión auditables.

## [0.4.0] - 2026-08-09

Introdujo HPO adaptativo asíncrono con Sobol, BoTorch opcional, continuation por checkpoints,
pruning conservador y scheduling consciente de coste/memoria/GPU.

## [0.3.0] - 2026-08-09

Añadió tareas genéricas, preprocessing componible/reanudable, cache, artefactos y tracking de
datasets, además de extensiones de resultados y componentes científicos.

## [0.2.0] - 2026-07-22

Amplió modelos, losses, métricas, componentes graph/vision/sequence/tabular/tree/generative, esquema
de experimentos, migraciones, seeds/grids/ablations, resultados y documentación de agentes.

## [0.1.0] - 2026-07-16

Versión inicial de infraestructura POO para entrenamiento PyTorch reproducible y experimentos YAML.
