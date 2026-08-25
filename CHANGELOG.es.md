# Historial de cambios

Español · [English detallado](CHANGELOG.md)

Este historial español sigue Keep a Changelog y Semantic Versioning. Resume todas las versiones;
el fichero inglés enlazado conserva el inventario exhaustivo de cada corrección y cambio histórico.
El repositorio aún no tiene tags: 0.1.0 y 0.2.0 se reconstruyeron a partir de commits de versión y
metadata empaquetada.

## [Sin publicar]

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
