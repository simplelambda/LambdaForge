# Admisión de almacenamiento y limpieza segura

Español · [English](STORAGE.md)

## Consultar y limpiar

```bash
lf clusters storage CLUSTER --json
lf storage status --on CLUSTER --json
lf storage reconcile --on CLUSTER --json         # mide; no modifica metadatos
lf storage reconcile --on CLUSTER --apply       # actualiza solo el inventario diagnóstico
lf clean --on CLUSTER              # previsualización; no elimina
lf clean --on CLUSTER --apply      # comprueba de nuevo y limpia candidatos seguros
```

En la consola, abre **Clusters → tu clúster → Clear storage…**. La acción muestra categorías
recuperables y referencias protegidas, pide confirmación, trabaja fuera del bucle de interfaz y
comunica espera/éxito/error en el panel de actividad. **Clear output** solo limpia el texto visible.
Ninguna olvida el historial científico. Usa `lf delete` o `lf jobs clear` para borrarlo expresamente.

La consulta explícita de almacenamiento inventaría directorios propios. Los sondeos habituales
solo consultan capacidad del filesystem y pequeños registros de leases, sin recorrer checkpoints
ni assets de datasets. Los informes separan categorías y volúmenes físicos; los bytes aparentes
no equivalen al espacio físico libre.
`reconcile` compara medidas profundas con `state/storage-ledger.json`, informa cambios de bytes/
archivos y permite actualizar atómicamente ese inventario versionado. Nunca elimina archivos ni
cambia registros científicos. Las categorías solapadas no se suman. El inventario no sustituye
al espacio físico ni a las referencias; si está corrupto se informa sin sobrescribirlo en silencio.

## Admisión

```yaml
resources:
  storage: 100GiB
```

Declara un compromiso futuro para la asignación externa del Job, no una cuota del filesystem ni
su ocupación medida actual. No impide que código consumidor arbitrario escriba más. Los Jobs de
LambdaForge que cooperan en el mismo host comparten leases específicos por filesystem. Nunca
se suman capacidades libres de volúmenes diferentes.

Se exige `libre físico - otros compromisos activos - nuevo compromiso >= seguridad` y disponibilidad
de inodos cuando el filesystem los informa. El consumo medido no reduce el compromiso declarado;
se libera al terminar el supervisor o al verificar positivamente la muerte del propietario.
Propiedad ilegible, corrupta o ambigua bloquea nuevos compromisos. La existencia de un PID por sí
sola no demuestra que su propietario siga vivo. Se compara el nacimiento registrado: uno distinto prueba
que el propietario original ya no tiene ese PID; un proceso zombi/muerto no puede usar su compromiso.
Esas leases antiguas del mismo host se retiran solo durante admisión bajo lock, nunca al consultar
estado. Un proceso vivo del mismo nacimiento conserva su compromiso aunque cambie argv (exec);
autorizar señales sigue exigiendo el match estricto de nacimiento/comando. Propietarios inaccesibles
o de otro host siguen protegidos. Workers y ofertas Fleet no duplican una asignación propia.

Un bloqueo de ownership genera `StorageOwnershipError`, no ENOSPC. El diagnóstico incluye lease,
host y propietario; `lf storage status --json` muestra incidencias de ownership acotadas. Inspeccionar
el host registrado, restaurar permisos de inspección o esperar a salida verificada antes de
reintentar publicación. No borrar leases vivas/remotas ni resetear historial científico. Incluso
un reflink de cero bytes verifica propiedad, margen físico e inodos; las transacciones anidadas
tienen identidades independientes de liberación exacta.

Si falta espacio, los Jobs directos esperan antes de adquirir CPU/GPU. El estado informa bytes
solicitados/libres/reservados/de seguridad y motivo. La misma autoridad puede ejecutarse dentro de
workers nativos gestionados por scheduler. Antes de seguir esperando se intenta GC automático
seguro; no se mata ciencia sana para liberar disco ni se reinician Runs mediante limpieza.

Ejemplo de perfil de clúster compartido, antes de aplicar el ámbito de proyecto:

```yaml
storage:
  state_root: /durable/lambdaforge/state
  cache_root: /scratch/lambdaforge/cache
  run_root: /scratch/lambdaforge/jobs
  dataset_root: /durable/lambdaforge/datasets
  lease_root: /durable/lambdaforge/host-leases
  cache_max_size: 500GiB
  cache_max_age: 30d
  safety:
    min_free: 20GiB
    min_free_percent: 5
  terminal_jobs:
    grace_period: 2d
```

Se aplica el mayor margen absoluto/porcentual. Por defecto se conserva un 5% libre; la retención
de checkpoints exitosos es de dos días. El ámbito de proyecto cambia las raíces operativas,
no la raíz de leases del host. El editor de la consola y `clusters set/unset` exponen estos campos.

La presión se informa como NORMAL, SOFT_PRESSURE, HARD_PRESSURE o CRITICAL a partir del espacio
físico, margen, leases e inodos. Las reservas son cooperativas: no impiden escrituras de otros
usuarios ni representan las cuotas externas de disco.

SLURM no tiene una directiva de scratch portable asumida. Las solicitudes positivas generan un
aviso de omisión salvo que se configure `resource_mapping.storage` para el sitio, por ejemplo
`{option: tmp, value: "{storage_mib}"}`. La admisión del worker no significa que SLURM haya reservado
ese espacio. Fleet exige una capacidad de almacenamiento reciente y conocida para solicitudes
positivas; una capacidad desconocida no autoriza el despacho.

## Qué se puede eliminar

La limpieza respeta referencias a Jobs activos, entornos exactos, runtimes, bundles y cachés Work.
Un build protege su propio prefijo y categorías de paquetes en mutación, no cachés Work ajenas.
La propiedad desconocida se conserva. Solo propietarios de build del mismo host, muertos y
obsoletos positivamente identificados permiten limpiar temporales huérfanos; los marcadores
históricos o de otro host permanecen protegidos.
Un controlador local muerto no basta si pip/Conda aún referencia el prefijo temporal exacto.
La evidencia de procesos ilegible también lo conserva. Bootstrap usa un intérprete existente del
host para metadatos de propiedad; no tiene que cumplir la versión Python del Work científico.

La limpieza automática aplica cuota/edad, presión y orfandad comprobada; sin cuota explícita usa
un techo del 10% de la capacidad del filesystem de caché. Primero selecciona categorías
reconstruibles más baratas y después por último uso. Si las referencias protegidas impiden
cumplir la cuota, informa del exceso pendiente en vez de eliminarlas. `lf clean` incluye además
cachés Work inactivas reconstruibles. Abarca paquetes pip/Conda/nativos/runtime, managers,
entornos, runtimes, bundles y temporales.

La eliminación de caché registra intención, renombra la entrada exacta a basura propia y reanuda
idempotentemente su eliminación interrumpida. Nunca aplica una intención sin confirmar a contenido
original que ahora podría estar referenciado. `state/storage-gc.jsonl` audita las limpiezas.
Resultados, logs, métricas, procedencia, datasets publicados, réplicas del investigador, artifacts
exitosos no publicados y Jobs activos/desconocidos no son candidatos de caché.

## Checkpoints y publicación

Las ejecuciones fallidas/interrumpidas conservan checkpoints de recuperación. La compactación de
artifacts sigue eliminando artifacts parciales fallidos y duplicados publicados verificados.
Una ejecución **exitosa** puede liberar checkpoints no fijados tras su retención. Guarda modelos
científicos duraderos como outputs; fija expresamente el estado resumible que deba conservarse:

```python
self.checkpoints.pin("best/model.ckpt", reason="retain-for-follow-up")
self.checkpoints.unpin("best/model.ckpt")  # solo metadatos; no elimina inmediatamente
model = self.outputs.from_checkpoint("model", "best/model.ckpt", release=True)
```

`from_checkpoint` sella un artifact registrado independiente y comprueba su identidad. La publicación
de datasets puede leer directamente de un directorio propio de checkpoints, sin una copia intermedia
en el Attempt:

```python
self.outputs.dataset(
    name="example", version="1", members=members,
    source_checkpoint="prepared-records",
    release_checkpoints=["prepared-records"],
)
```

Las rutas de assets son relativas a ese directorio. La liberación expresa solo es elegible después
de finalizar correctamente la ejecución y verificar una publicación independiente. Un destino
cambiado o desaparecido invalida la prueba. Las copias usan copy-on-write si está disponible y
copias regulares independientes si no; nunca hard links modificables. No se garantiza zero-copy.
Las copias de snapshots gestionados reservan bytes adicionales en el filesystem de **destino**,
incluidos `publish_to` externos y assets de datasets. Si falta margen se intenta una limpieza segura
de caché en el mismo volumen antes de mostrar un diagnóstico compacto; GC nunca recorre ni elimina
ese directorio externo. Una copia no comprometida conserva los checkpoints originales. Estos leases
incrementales se liberan tras éxito/error y no reducen el compromiso declarado del Job. Un reflink
comprueba margen/inodes sin contar dos veces el payload físico; si no es posible, la copia normal
debe reservar primero todos los bytes.

La limpieza toma el lock de escritura de Execution, comprueba otra vez pins/identidad y registra
la eliminación de checkpoints; no compite con un escritor de recuperación activo.
`checkpoint-retention.json` audita eliminaciones sin perder resultados, logs, métricas ni historial
de Attempts. Los metadatos de retención ausentes/corruptos se tratan conservadoramente.

## Límites actuales

Están implementadas la admisión de run_root y las reservas de copias de publicación gestionadas.
Staging de bundles y preparación de dependencias todavía no tienen reservas independientes de bytes
en sus volúmenes de destino. Generar índices/manifests de datasets no es una cuota de disco.
Los entornos managed siguen usando identidad v2 con hashes exactos de wheels: **todavía
no se han separado** el entorno de dependencias y las capas de código. Los marcadores de build se
protegen conservadoramente y la orfandad entre hosts sigue sin inferirse. Los markers de runtime Python
y entorno comparten la adquisición atómica con GC y heartbeat acotado, también al verificar reuse.
Un prefijo completo inválido se conserva, no se reemplaza bajo Jobs que podrían referenciarlo.
`lf storage reconcile` ofrece un inventario diagnóstico de medidas, no contabilidad completa de cada
escritura. Esta guía no promete protección frente a todo ENOSPC durante preparación.
