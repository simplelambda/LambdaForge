# Studies coordinados: estado y contratos

[English](COORDINATED_STUDIES.md)

## Estado: CPU preparada integrada; ejecución distribuida pública pendiente

Este documento registra la implementación solicitada. **Aún no existe una ruta soportada
`lf run --on-fleet`.** Los envíos independientes no son un HPO coordinado. Los planners nativos
ya comparten un punto interno de despacho; la ruta pública no se ha redirigido a Fleet.
Se conservan los Studies de un cluster y
`MultiClusterSubmissionService` independiente.

Implementado y probado:

- `Fleet`/`FleetMember`: roles, miembros y límites operacionales en `ClusterCatalog`.
- `FleetResourceService`: reutiliza `ResourceService`, distingue observación física de admisión
  acreditada por el executor y muestra disponibilidad de miembros opcionales/obligatorios.
- `ExecutionEquivalence`: identidad exacta de código, entorno, inputs, numerics y hardware.
- `GlobalRun`: identidad candidato/seed/fase/fidelity inmutable, independiente del placement.
- `GlobalPlacementBroker`: filtros de frescura, preparación, equivalencia, memoria, límites y
  checkpoint local. Orden por terminación estimada/carga cuando se conocen; los tiempos desconocidos
  permanecen desconocidos y no se inventan probabilidades. Desempate reproducible por nombre.
- `StudyCoordinator`: leases atómicas por Attempt, shards únicos, fence previo al envío, estado
  remoto desconocido, retry limitado de pérdida confirmada, reconciliación tras reinicio,
  validación de digests y cuarentena. Recibe propuestas científicas; no inventa otro optimizer.
- `ShardExecutor`: contrato de envío idempotente y observación factual. `PreparedCpuShardExecutor`
  usa ahora Jobs reales y desacoplados de `ProcessScheduler` para CPU local fresca ya verificada.
  Exige el intérprete existente actual, aplica límites y prevalidación antes del envío, y conserva
  el fence ante aceptación ambigua. Remoto, SLURM, command-GPU y Attempts recuperados siguen pendientes.
- `work.shard.execute_concrete_shard`: prueba interna de worker CPU fresco que reutiliza el
  dispatcher aislado, sin otro planner. Valida una cola finita de leases, persiste resultados,
  aísla fallos científicos y no reejecuta una reentrega terminada. Rechaza GPU, Attempts recuperados
  y continuación desde checkpoint hasta conectar sus garantías de proveedor/identidad. Exige
  equivalencia previamente verificada por el llamador; no la acredita esta capa por sí misma ni
  está conectada al envío público.
- `WorkRunner(dispatcher=...)` conserva los mismos planners fixed, paired sweep automático y
  adaptativo. Seeds, candidatos, convergencia y algoritmos científicos no pasan al dispatcher.
  `CoordinatedCpuDispatcher` integra CPU fixed con leases, invocaciones durables, reconciliación y
  el callback original de resultados/refill, incluida la retirada exacta de la cola no iniciada
  sin revocar workers residentes. Las pruebas reales con dos destinos CPU directos cubren seeds
  fijas y refill de bloques pareados automáticos completos, con un Study/Execution y un análisis
  final. Rechaza HPO distribuido hasta integrar métricas/pruning central.
- Estado coordinator v2: `pausing`, `paused`, `resuming`. La pausa congela aceptar propuestas/envíos,
  deja terminar workers propios, ingiere resultados y espera por propietarios desconocidos.
  Leases anteriores al envío conservan identidad. El reloj original puede importarse una sola vez
  y no se reinicia. Reconciliar no convierte leases aún no enviadas en propietarios desconocidos;
  reanudar no permite enviar una lease cuyo presupuesto temporal original caducó. v1 válido
  migra sin reescribir evidencia. Son operaciones internas:
  **aún no existen comandos públicos `lf pause/resume`**.
- `lf fleets list/show/offers/drain/disable/enable`: inspección y cambios de catálogo preview-first.
  Los controles aún no gobiernan Studies de producción activos. `offers` observa, no declara
  que exista capacidad concedida ni que esté lista la ejecución.

Pendiente antes de habilitar ejecución coordinada:

1. Ampliar el dispatcher integrado de CPU fixed preparada a preparación remota y streaming central
   de métricas/pruning. Mantener `PairedSweepSequentialAnalyzer` y bloques completos; la aceptación
   local CPU de bloques ya está probada, pero falta integrar transporte/recovery de producción.
2. Implementar workers de shard usando `ControlPlane`, bundles/entornos, `JobService`,
   `ProcessScheduler`/`SlurmScheduler`, política GPU y ARI local. Solo reciben Runs concretas.
3. Preflight de datos/entornos/numerics/hardware verificados y offers dentro de grants exactos;
   aplicar el límite GPU localmente. `nvidia-smi` no otorga permiso de ejecución.
4. Conectar métricas/checkpoints/artifacts remotos con telemetría, resultados, recovery y export.
   Por ahora un checkpoint no local bloquea placement; no se finge una réplica automática.
5. Añadir policy científica primary/predictive versionada al planner **existente**: adquisición
   pending-aware, presupuestos, retirada de propuestas no iniciadas e información de capacidad
   ociosa. La base exige revisiones del planner/evidencia/modelo y motivo/policy para aceptar una
   Run predictive, pero aún no genera tales propuestas.
6. `StudyDecision`, dependencias explícitas entre decisiones sin lenguaje de workflow arbitrario,
   lanzamiento/reconcile nativo, TUI y HTML científico único con procedencia/utilización.
7. Completar adaptadores y aceptación del HPO predictive. Pruebas reales de proveedor CPU local
   no acreditan ejecución remota/GPU/SLURM/command.
8. Integrar launch/pause/resume/reconcile públicos, adopción desde un cluster, expansión en vivo,
   alojamiento durable del coordinador, export distribuido y transferencia de artifacts/checkpoints.

LambdaForge 0.17.0 incorpora el catálogo Fleet y las bases de coordinación probadas descritas aquí.
Todavía no ofrece ejecución completa de Studies multi-cluster; siguen pendientes las integraciones
indicadas arriba.

## Catálogo de Fleet

Configuración operacional en `lambdaforge.clusters.yaml`, nunca en el YAML científico:

```yaml
clusters:
  a: {transport: ssh, host: a-login, user: USER, scheduler: local}
  b: {transport: ssh, host: b-login, user: USER, scheduler: local}
  queue: {transport: ssh, host: queue-login, user: USER, scheduler: slurm}
fleets:
  research:
    coordinator: local
    members:
      - {cluster: a, max_gpus: 2, max_runs: 4}
      - {cluster: b, max_gpus: 4}
      - {cluster: queue, max_inflight_jobs: 2, required: false}
```

Los perfiles conservan launcher/claim, espejo del proyecto y data environment. Todas las
referencias deben resolverse antes de sondear. Precedencia usuario < proyecto < explícito; un
override sustituye la lista completa de miembros, no mezcla listas accidentalmente. Se mantienen
perfiles de cluster, referencias de credenciales y execution profiles. Límites: enteros positivos
o null. online/draining/disabled son controles del operador, no resultados de un sondeo.

```bash
lf fleets list --json
lf fleets show research --json
lf fleets offers research --json         # Solo observación explícita, no allocation.
lf fleets drain research b               # Preview del cambio de catálogo.
lf fleets drain research b --apply
lf fleets disable research b --apply     # No envía señales a Jobs activos.
lf fleets enable research b --apply
```

`lf fleets --catalog RUTA ...` selecciona un catálogo explícito; sin esa opción los cambios se
guardan en el proyecto. No se guardan secretos ni se cambia identidad científica del Work.

## Propiedad y persistencia

El futuro driver posee un directorio de coordinator del proyecto bajo su Execution.
`coordinator.json` versión 2 (con lector v1) se publica atómicamente con fsync usando el publicador JSON y lock
cross-process existentes. `leadership()` protege el bucle de planificación. Las transacciones
cortas no mantienen locks durante E/S remota. El estado contiene inicialización/presupuestos,
Runs, leases/historial de Attempts, motivos de placement, shards, controles y cuarentena.
`results/` guarda envelopes inmutables con nombre por digest; `quarantine/` contradicciones.
Un estado ausente/corrupto/de versión no soportada falla cerrado, nunca comienza de cero.
Un reinicio no puede cambiar silenciosamente la autoridad coordinadora. Las acreditaciones de
preparación requieren booleanos explícitos y las propuestas científicas son JSON estricto y finito
antes de persistirlas.

La clave global usa `(study_identity, candidate, seed, phase, fidelity)` y la primitiva existente
de identidad científica. La lease añade Attempt, ID opaco y cluster. Dos drivers no reservan
el mismo Attempt. Las offers excluyen residentes ya reconocidos localmente; el coordinator
resta leases adicionales no reconocidas para evitar sobreasignación entre preparación y envío.

El fence de intención se persiste **antes** de llamar al proveedor. Si se pierde la respuesta,
se consulta por ID inmutable de shard, no se vuelve a enviar ni se crea otro Attempt. El contrato
del executor también exige envío idempotente. Los shards nunca contienen credenciales.
Las confirmaciones tardías se vinculan al Attempt exacto: no sobrescriben resultados terminados,
no devuelven una Run activa a queued ni asignan el Job ID antiguo a un Attempt sustituto.

Offers frescas pueden permitir trabajo nuevo; las caducadas no. Un fallo de transporte marca
leases activas `unknown_remote`, con backoff exponencial limitado a cinco minutos: no cancela
workers, no fabrica fallos y no libera identidades. Una observación ausente también es desconocida.
Retry requiere prueba positiva del scheduler/registro de procesos/shards propios que identifique
la lease y confirme que no queda owner vivo, y está limitado. Una excepción del consumidor no se
reintenta automáticamente. Shards antiguos no consumen el límite de Jobs del Attempt sustituto.

El resultado debe coincidir en Study/Run/Attempt/lease/target/parámetros/equivalencia. Duplicados
exactos son inocuos; contradicciones y resultados tardíos de una lease liberada van a cuarentena
con diagnóstico compacto sin sobrescribir evidencia válida. Reentrega repara una caída entre
publicación del resultado e índice. Lecturas comprueban el digest. Se rechazan symlinks de raíz
y evidencia. Los snapshots muestran recuentos, no descargan todos los resultados.

La reserva consume conservadoramente el límite de dispatch `max_runs`; si el operador la retira
antes del fence de envío se devuelve esa reserva no gastada, conservando el historial del Attempt.
El wall-time parte de
la creación inmutable y atraviesa reinicios. La contabilidad física y liberación reversible de
reservas no iniciadas necesitan aún la integración con el dispatcher de producción.

## Pruebas reproducibles de esta base

No ejecutan WISDOM ni contactan clusters reales:

```bash
python -m pytest -q tests/controlplane/test_coordinated_study.py tests/controlplane/test_fleets.py
python -m pytest -q tests/work/test_concrete_shard.py
python -m pytest -q tests/work/test_study_dispatch_boundary.py tests/controlplane/test_coordinator_pause.py
python -m pytest -q tests/work/test_coordinated_cpu_dispatch.py
```

El diseño fijo simulado tiene 56 candidatos × 4 seeds = 224 Runs y capacidades A=2, B=3, C=1.
Toda la evidencia entra una vez y los objetivos sintéticos coinciden exactamente con una
referencia serial. La partición desconecta B, carga otra instancia del coordinator, dirige trabajo
nuevo a A/C y reconcilia un residente, un completado y una pérdida confirmada de B: solo esta
última crea Attempt 2. También se prueban drivers concurrentes, acknowledgements ambiguos,
frescura, límites, checkpoint local, incompatibilidad de entorno/hardware, procedencia,
presupuestos, drain e identidades inmutables. No son aún pruebas de ejecución distribuida
end-to-end ni calidad del optimizer predictive.
La prueba del shard CPU ejecuta cuatro Work reales en subprocesos a concurrencia dos, conserva
evidencia y rutas nativas de los éxitos/fallos y verifica que reentregar no crea otro Attempt.
No envía un Job a un proveedor, no otorga una GPU y no arranca un optimizer.

Las pruebas del proveedor preparado ejecutan cuatro identidades fijas o tres bloques pareados
completos de dos candidatos en dos destinos CPU directos con Jobs desacoplados; ingieren evidencia
centralmente y producen un análisis final único.
Usa ProcessScheduler real local, pero aún no acredita `lf run --on-fleet`. Las pruebas de inyección
ejercitan el planner adaptativo nativo sin crear otro optimizer. La pausa cubre particiones,
ingestión al terminar, carreras previas al envío, reinicio y presupuestos inmutables.

Los sweeps authored fijos mantendrán desactivada la poda de rendimiento. Los automáticos
mantendrán bloques pareados completos; el HPO adaptativo podrá podar según su contrato actual.
El trabajo predictive futuro pertenece al Study y a sus presupuestos: capacidad libre no autoriza
dependencias downstream no resueltas ni propuestas aleatorias sin información útil.
