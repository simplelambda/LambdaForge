[English](AUDIT_0.6_TO_0.7.md) | [Español](AUDIT_0.6_TO_0.7.es.md)

# Auditoría de LambdaForge 0.6 a 0.7

Esta auditoría registra el estado del `main` real antes de implementar 0.7. No es un roadmap
deducido de la documentación. Cada conclusión se contrastó con implementación y tests el
2026-08-14.

## Clasificación

| Área | Clasificación | Evidencia y decisión 0.7 |
|---|---|---|
| Publicación desde preprocessing | CONFIRMED | `PreprocessingTask.run()` siempre escribe `dataset-artifact.json` y usa el nombre de la task si falta `dataset_name`. La publicación será explícita; `dataset_name` se conserva como opt-in legacy. |
| Frontera Task/dataset | DESIGN ISSUE | `TaskRunner` registra cualquier manifest que encuentra. Se mantiene por compatibilidad, pero preprocessing normal deja de fabricarlo. |
| Receta de dataset | CONFIRMED | No existe documento `kind: dataset` ni entidad recipe/build. Se añade una receta tipada que compila sus stages al DAG Workflow existente. |
| Reutilización de stages | PARTIALLY SOLVED | `TaskFingerprint`, artifacts verificados y bindings Workflow ya dan reuse content-addressed. Se añaden decisiones de receta, `required`/`reuse`, force downstream y recibos durables sin otra cache key. |
| Builds durables | PARTIALLY SOLVED | `JobService`, schedulers 0.6 y `job_type` ya persisten jobs. Los builds usarán `job_type=dataset-build`. |
| Identidad v1 | DESIGN ISSUE | `DatasetArtifact.create()` mezcla nombre, versión, fingerprint, source y metadata con los bytes. v2 separa `content_id` y `build_id` y sigue leyendo v1. |
| Miembros lógicos | CONFIRMED | v1 sólo tiene counts agregados y artifacts de task. Se añaden `DatasetMember` y `DatasetIndex` JSONL streaming sin imponer layout. |
| Partitions y targets | CONFIRMED | v1 expone un único mapa `splits` y ningún schema de targets. v2 deriva partitions arbitrarias y conserva targets genéricos con schema explícito opcional. |
| Placements | ALREADY SOLVED | `DatasetPlacement` ya es independiente del ID, multiclúster y reconciliable. Se conserva. |
| Alias inmutables | ALREADY SOLVED | `DatasetRegistry.register()` rechaza un `name@version` con otra identidad. Se conserva con error tipado. |
| Registry/DataCatalog | CONFIRMED | Registry conoce placements gestionados, pero resolver, TaskInput y bundles exigen `DataCatalog.locations`. Se crea un `DatasetResolver` único; DataCatalog queda para loaders, alias y datos externos. |
| Referencias versionadas | PARTIALLY SOLVED | Se parsea `dataset:name/subpath` y Registry acepta `name@version`, pero los bindings no fijan ambos consistentemente. Se amplía la referencia y se persiste content ID/placement exactos. |
| Lockfile | NOT WORTH CHANGING | Bindings materializados más IDs inmutables resuelven el pin sin otra autoridad mutable. |
| `materialize BUILD` | CONFIRMED | Con `apply=True` sólo se indica ejecutar manualmente el producer. Se conectará a build local o submission durable. |
| Publicación atómica | PARTIALLY SOLVED | El registry es atómico, pero publicar bytes no es una transacción staging/validate/rename. Se añade una frontera de publisher. |
| GC de stage cache | PARTIALLY SOLVED | GC ya se limita a cache y excluye datasets. La cache de stages irá bajo cache root y declarará referencias activas/publicadas. |
| Validación | PARTIALLY SOLVED | v1 verifica checksum y paths. Se amplía a IDs únicos, índice, assets, partitions y schema de targets. |
| Profiling | PARTIALLY SOLVED | Hay stats del filesystem y profiler classification explícito. El perfil base debe leer `DatasetIndex`; profiler de proyecto remoto aún se rechaza. |
| Members y diff | CONFIRMED | No existen APIs de listado/detalle/diff. Se añaden consultas streaming acotadas y diff por identidad. |
| Lineage | PARTIALLY SOLVED | Es una tupla plana. Se conserva lectura y se añade provenance estructurada de recipe/build/stage/input. |
| Entry point `lf` | CONFIRMED | Sólo se instala `lambdaforge`. `lf` apuntará al mismo callable. |
| Gramática/aliases CLI | PARTIALLY SOLVED | Ya hay namespaces y shortcuts, pero opciones/salidas divergen. Se añaden aliases moderados y render compartido. |
| Run por nombre | ALREADY SOLVED | `ProjectConfigService` ya resuelve nombres inequívocos en comandos root y de recursos. Se conserva. |
| Cluster por defecto | CONFIRMED | No se aplica preferencia de proyecto/usuario. Se añade de forma explícita y visible; `--on` gana. |
| `plan` root | CONFIRMED | `plan` de entidad reenvía a inspect, pero falta shortcut uniforme y data readiness. Se añade sin retirar `run --dry-run`. |
| Errores tipados | CONFIRMED | La CLI suele mostrar `KeyError/RuntimeError`. Se añaden errores de datasets/control plane y render accionable. |
| Arquitectura CLI | DESIGN ISSUE | `CommandLineInterface.py` mezcla parser y ejecución de todos los dominios en más de 2300 líneas. Lo nuevo de datasets/render/errors/completion irá a módulos pequeños; no se reescribe lo estable sólo por estética. |
| Salida humana/JSON | DOCUMENTATION DRIFT | Varios comandos aceptan `--json` pero siempre imprimen JSON; jobs no tiene headers. Se añaden renderers humanos conservando JSON estable. |
| Active/dry-run | CONFIRMED | `CREATED` no es terminal y un dry-run puede persistirlo indefinidamente. Se añade `PLANNED` y breakdown explícito, leyendo `CREATED` legacy. |
| `top` | PARTIALLY SOLVED | Ya existen snapshots de recursos/jobs; la vista los reduce a un active ambiguo. Se mejora sin inventar telemetría desconocida. |
| Selectores job | CONFIRMED | Sólo acepta IDs exactos. Se añade `latest` y nombre inequívoco. |
| Completion | CONFIRMED | No hay comando. Se generan scripts bash/zsh/fish sin dependencia pesada. |
| Authoring | DOCUMENTATION DRIFT | Los campos concisos top-level se aceptan, pero el IR los guarda en `extensions.authoring` y algunos docs lo exponen. Se mantiene el IR y se documenta sólo authoring conciso. |
| Artifacts/results | ALREADY SOLVED | Ya hay servicios, sync de evidencia y fetch pesado explícito. Sólo se añadirán aliases, no otro registry. |
| Versiones | CONFIRMED | El commit v0.6.1 aún declara `0.6.0` y templates antiguos. Se añade test de coherencia y se actualiza 0.7. |

## Frontera arquitectónica conservada

```text
DatasetRecipe (cómo) -> DatasetBuild (ejecución) -> DatasetVersion (qué)
                                                     -> DatasetPlacement (dónde)
```

Workflow gobierna el DAG, Task la identidad y outputs verificados de cada stage, JobService el
scheduling durable, DatasetArtifact/Index el contenido inmutable, DatasetRegistry los placements
gestionados y DataCatalog los alias/loaders/datasets externos. Esta auditoría no justifica daemon,
base de datos, scheduler DAG distribuido ni placement científico automático.
