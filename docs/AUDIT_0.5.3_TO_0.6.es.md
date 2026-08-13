[English](AUDIT_0.5.3_TO_0.6.md) | [Español](AUDIT_0.5.3_TO_0.6.es.md)

# Auditoría del plano de control LambdaForge 0.5.3 a 0.6

Esta auditoría se realizó sobre el código 0.5.3 antes de implementar 0.6. Separa hechos observados
de peticiones de la especificación.

## Capacidades existentes reutilizables

- `ClusterCatalog` ya fusiona perfiles de usuario, proyecto y explícitos con procedencia.
  `ClusterService`, `ControlPlane` y `ControlPlaneFactory` separan servicios de los contratos
  `Transport`, `Scheduler` y `EnvironmentProvider`.
- OpenSSH y Paramiko con verificación de host evitan `shell=True`; Paramiko reutiliza un cliente
  durante una invocación. `SlurmProfile` permite adaptar comandos/recursos y los jobs SLURM ya
  sobreviven al PC controlador.
- `JobStore` persiste JSON atómico legible hacia atrás. `JobService` lista, refresca, lee logs,
  cancela y reintenta. La ejecución científica ya posee limpieza recursiva, parent-death guards y
  Windows Job Objects reutilizables.
- Bundles y entornos managed están identificados por contenido. 0.5.3 resuelve Torch/CUDA remoto,
  verifica CUDA e incorpora la política exacta a la identidad.
- `ResourceRequest` distingue CPU/RAM/GPU/storage/tiempo/procesos solicitados y ya existe monitoreo
  de proceso/CUDA durante ejecución científica.
- `DataCatalog`, `DataService`, `DatasetArtifact`, providers de identidad y transferencia ya cubren
  identidad lógica, placements declarados, manifiestos, inspección y réplica explícita. El
  preprocesado produce `dataset-artifact.json` por contenido.
- `ResultService`, sincronización remota pequeña, artefactos, `ExperimentRegistry`, retention,
  plugins/configuración y workflows ofrecen APIs reutilizables.

## Limitaciones confirmadas

- `LocalScheduler.submit()` ejecuta síncronamente, guarda estado/logs sólo en memoria y no puede
  reconectar, pausar ni imponer runtime. En SSH sin SLURM mantiene la conexión durante todo el job.
- `JobService.submit()` escribe el registro después de que termine el submit; no hay CREATED/STAGING
  durable durante una operación larga o un fallo intermedio.
- `SshTransport` crea un proceso `ssh`/`scp` por operación. Puede aprovechar un `ControlMaster` del
  usuario, pero 0.5.3 no lo activa ni limita. Paramiko reutiliza dentro del proceso, pero
  `ssh_timeout` gobierna conexión, banner, auth y `exec_command`.
- `Transport.run()` no acepta timeout de comando independiente. Conexión, comando y runtime
  científico no son políticas separadas.
- Para jobs directos el store local es la única verdad: faltan supervisor remoto, heartbeat,
  identidad PID segura, leases GPU, reconciliación e inventario global concurrente.
- Faltan STAGING, PAUSED y TIMEOUT, capacidades pause/resume y la advertencia de VRAM retenida.
- No hay `ResourceService`/probes/top que separen solicitado, asignado, observado y disponible.
- Dataset exige catálogo explícito: faltan registry automático, índice remoto, profiler, borrado
  seguro, lineage/materialización y auto-registro del preprocesado.
- Configs siguen orientadas a paths; faltan índice de proyecto, ejecución por nombre y job groups.
- Bundle cache, work remoto y outputs están físicamente acoplados. No hay GC global por referencias,
  roots de storage por clúster ni budgets.
- `CommandLineInterface` supera 1.700 líneas y aún concentra dispatch de dominio.

## Suposiciones falsas de la especificación

- `DataService.inspect()` sí está expuesto: `lambdaforge data --catalog C inspect NAME` lo usa.
- SLURM ya es asíncrono y durable; la brecha es `scheduler: local`/SSH directo y la vista global.
- `ResultService` ya resuelve resultados locales por nombre/fingerprint/path y existe sync pequeño.
- Los entornos ya verifican un marker antes de reutilizar; falta publicación por directorio temporal
  y GC/ref tracking, no toda la verificación.
- SQLite no está justificado aún: JSON atómico con locks es más simple y portable al lado remoto.
- Separar runtime de wheel consumidor sería ambiguo sin lock exacto; primero conviene compartir
  descargas, reutilizar y hacer GC.

## Riesgos de compatibilidad

- Los JSON de jobs 0.5 deben seguir leyéndose; campos nuevos necesitan defaults/versionado.
- `scheduler: local` sigue válido pero pasa de bloqueo a submit durable. Quien interpretaba el
  retorno como finalización debe consultar `jobs` o usar un foreground explícito.
- `ssh_timeout` sigue aceptado como alias deprecado sólo de connect timeout, nunca de comando/job.
- Layouts `bundles`/`environments` antiguos se reconocen y clasifican, nunca se borran en silencio.
  `data` queda como alias mientras `datasets` pasa a ser la interfaz principal.
- Paths, YAML y resultados existentes siguen válidos. No se reparten HPO adaptativos ni DAGs entre
  clústeres y no hay placement automático.

## Arquitectura 0.6 propuesta

No se requiere daemon/servidor. SLURM manda sobre su job; el supervisor remoto manda sobre jobs de
proceso; `DatasetArtifact` más un placement atómico manda sobre datasets; `result.json` manda sobre
evidencia científica; markers/bytes mandan sobre caches; YAML materializado manda sobre configs.
Los registros locales son índices reconciliables.

`ProcessScheduler` lanza un `ProcessSupervisor` desacoplado por job con estado, logs, heartbeat,
usage, identidad de proceso, timeout y leases GPU. `JobService` escribe CREATED antes de enviar y
reconcilia inventarios. Pause/resume se expresan como capacidades.

`SshConnectionPolicy` separa connect/auth/banner/keepalive/comando. OpenSSH usa sockets privados
`ControlMaster`/`ControlPersist` con expiración por inactividad para reutilizar autenticación entre
operaciones e invocaciones CLI. Paramiko mantiene cliente por invocación y keepalive; el timeout de
comando es independiente y opcional.

`ResourceService`, `DatasetService`, `ProjectConfigService`, `StorageService` y
`EnvironmentService` son APIs de objetos. Lecturas multi-clúster tienen concurrencia limitada y
estado UNREACHABLE/último conocido. Toda destrucción es preview inmutable y exige `--apply`.
Datasets/resultados nunca son cache.

## Plan de migración

1. Políticas versionadas de conexión/storage y transports con timeout independiente.
2. Supervisor/estados/capacidades y submit durable local/directo con registro previo.
3. Reconciliación, grupos y consultas globales; SLURM sigue siendo autoridad.
4. Snapshots/probes de recursos, overview y top.
5. Registry dataset atómico, auto-registro, profilers e inventario remoto.
6. Planes seguros verify/remove/delete/replicate/materialize con `--apply` para movimiento grande.
7. Descubrimiento de configs y fachadas experiment/task sin duplicar runners.
8. Separar work mutable de cache bundle y añadir status/GC por referencias sólo sobre cache/tmp.
9. Conservar aliases/layouts, documentar EN/ES y validar con tests focalizados, integración POSIX,
   wheel y suite CI completa.
