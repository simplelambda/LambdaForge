# Studies coordinados: estado y contratos

[English](COORDINATED_STUDIES.md)

## Estado: lanzamiento Fleet adaptativo nuevo y fijo; recovery/transferencia pendientes

`lf run CONFIG --on-fleet FLEET` encola un coordinador local durable para Runs repetidas,
sweeps fijos, sweeps pareados automáticos y HPO adaptativo nuevo sin continuación de fidelity.
El selector de destino de Run Work en la Consola ofrece
la misma ruta `fleet:NAME`. El YAML científico no cambia. Es una **capacidad pública limitada**,
no la finalización de la petición completa de HPO adaptativo coordinado.

La ruta de un clúster y `MultiClusterSubmissionService` independiente se conservan.
Los envíos independientes no son HPO coordinado.

### Implementado

- Un único `WorkRunner` nativo con los planners adaptive, fixed y `PairedSweepSequentialAnalyzer` originales.
  El dispatcher ejecuta propuestas exactas y los callbacks de resultado/refill existentes; no crea
  candidatos, seeds ni otro optimizer en los workers. Una Execution y un análisis final.
- Un Job de proveedor persistente por miembro, preparado por `ControlPlane`/`JobService`.
  El runtime de miembro recibe oleadas finitas con leases, reutiliza procesos Run aislados y ARI,
  y deja terminar lo aceptado antes de liberar su allocation. No crea otro claim GPU ni otro Job
  del scheduler por cada oleada.
- Heartbeats frescos del Job propio acreditan código/entorno/inputs file/numerics y hardware
  homogéneo real. Las cinco identidades deben coincidir entre miembros antes de crear leases
  científicas. Los placeholders de preparación nunca son evidencia de placement.
- Admisión GPU baseline con tokens opacos heredados, probes nativos de vida corta, límites por
  miembro y memoria física libre. Nunca amplía grants. Una oleada ocupada ofrece cero slots:
  **faltan co-location GPU y refill ARI incremental dentro del miembro Fleet**.
  Los offers caducan; las observaciones físicas de `lf fleets offers` no autorizan despacho.
- La ruta preparada conserva políticas direct/site-command/SLURM, TLS, bundles y entornos managed
  inmutables. Hay aceptación CPU loopback y CUDA real mínima; no acredita aún SSH, site-command
  ni SLURM de extremo a extremo.
- Los markers file conservan parámetros científicos authored y verifican bytes/tamaño canónicos
  durante preparación y binding del worker. Los grandes siguen usando el espejo del proyecto.
  La acreditación distribuida de datasets se rechaza: NAME@VERSION igual no basta.
- Leases v2, fences de envío, propietario unknown, cuarentena, límites, reloj/presupuestos originales
  y reconciliación siguen siendo autoridad. El tamaño global de oleada respeta el paralelismo
  nativo. Una pérdida de conexión por sí sola nunca autoriza otro Attempt.
- Jobs miembros visibles en `lf jobs`, pero no como Works semánticos independientes. El padre
  aparece como `fleet:NAME`; Runs activos y terminados incluyen clúster/Job/shard/lease/Attempt
  exactos. Los registros nativos del worker proyectan los últimos/mejores escalares acotados en
  la vista Study existente del coordinador. `lf show STUDY --run KEY --json` y
  `lf logs STUDY --run KEY` leen curvas/logs vivos o terminales en su miembro verificado,
  bajo demanda. Nunca se abre localmente una ruta remota. Los resúmenes visuales no sustituyen
  al stream científico durable descrito a continuación.
- `metrics.jsonl` y `training-metrics.jsonl` nativos se transportan incrementalmente como registros
  completos verificados con SHA-256: máximo 32 KiB por canal/Run/lectura y compresión sin pérdida
  opcional. Cada lectura acredita Study/Run/Attempt/lease/miembro/shard exactos; los offsets preservan
  orden y deduplican retransmisiones. El cursor durable recupera appends no confirmados tras una
  interrupción; pérdida/corrupción de bytes confirmados, saltos y replay contradictorio fallan cerrado.
  El worker conserva evidencia durante desconexiones. No se transportan logs ni artifacts pesados
  por este stream. El planner recibe resultados terminales solo tras completar sus escalares.
- El mismo planner adaptativo central consume evidencia terminada y todas las propuestas leased/
  queued/running entre miembros. Conserva adquisición pending-aware nativa: qLogNEI de BoTorch si
  está disponible y fallback mixed-kNN determinista. Capacidad admisible libre invoca la frontera
  científica nativa acotada, nunca propuestas aleatorias de otro optimizer. Repriorizar Runs aún
  sin lease conserva identidad científica y registra la invocación/prioridad anterior.
- La poda central reutiliza utility/historial/calibración nativos y envía solicitudes de parada
  idempotentes con lease exacto. Evidencia required/startup/confirmation sigue protegida. La parada
  es cooperativa; desconexión no significa ACK ni evidencia pruned. Espejos escalares locales
  verificados alimentan calibración histórica sin abrir rutas remotas localmente. Continuación/
  recovery de checkpoints sigue bloqueado, no se deduce de rutas locales con nombres similares.
- `lf cancel STUDY` detiene primero el coordinador y después sus miembros propios. El borrado
  semántico previsualiza la familia completa y rechaza miembros activos. La limpieza individual
  protege evidencia referenciada por la familia. La compactación pesada se hace en su host,
  nunca escribiendo rutas remotas sobre el coordinador.

### Comandos disponibles

```bash
lf run study.yaml --on-fleet research --dry-run --json  # Sin grant, upload ni Job.
lf run study.yaml --on-fleet research                  # Envío asíncrono durable.
lf show STUDY --json
lf show STUDY --run trial-00001-seed-4 --json
lf logs STUDY --run trial-00001-seed-4
lf cancel STUDY --dry-run
lf cancel STUDY --apply
lf delete STUDY                                       # Preview de familia terminal.
lf delete STUDY --apply
```

Coordinador `local`; miembros managed, distintos del destino local incorporado. Dry-run valida
fuente/firma/diseño, roles/credenciales, muestra caps y verificaciones diferidas sin consultar ni
adquirir GPU. El request durable captura perfiles y referencias de credenciales, no sus valores.
`--on`/`--on-fleet` son excluyentes. Se rechazan `--rerun`, `--restart`, flags de recovery internos
y `--wait-for-submit` para esta ruta.

### Pendientes

Stream de checkpoints/continuación y recovery distribuido; predictive/lookahead;
co-location; pause/resume/reconcile públicos y adopción; recovery tras reinicio del coordinador;
expansión/drain de miembros en vivo; equivalencia de datasets; transferencia de checkpoints/
artifacts, compresión/reanudación y export distribuido; TUI/HTML completos de placement/utilización;
StudyDecision y dependencias. Los controles internos de lifecycle/catálogo no sustituyen esas
operaciones públicas. Retry Fleet rechaza reiniciar como un solo clúster; export rechaza generar
un paquete del coordinador que omita silenciosamente la evidencia de los miembros.

La petición **no está terminada**. Esta ruta limitada no es HPO adaptativo distribuido listo para
producción completo; las pruebas loopback/fake no acreditan todos los proveedores.

### Validación incremental (2026-10-05)

Regresiones del stream cubren lecturas acotadas de varios chunks, compresión, replay, registros
incompletos, interrupción del receptor, leases ajenas y symlinks. Regresiones del miembro cubren
poda dirigida, idempotencia, órdenes inmutables y evidencia protegida. La integración pública CPU
loopback ejecuta el planner adaptativo nativo sobre dos allocations con un único estado HPO/análisis.
La poda histórica se prueba con espejos verificados y rutas del propietario deliberadamente ausentes.
No acredita aceptación adaptativa remota/GPU real ni recovery tras reinicio del coordinador.

### Validación de esta implementación (2026-10-04)

- `ruff check .`: correcto.
- `mypy src/lambdaforge`: correcto, 558 archivos fuente.
- `pytest` completo: 1.246 pruebas correctas; cuatro avisos de Lightning en pruebas CPU.
- Una wheel recién construida e instalada pasó los smoke tests de packaging, la verificación
  del import instalado, `run --help`, scaffolding, validación y dry-run de solo lectura.
- La integración pública Fleet probó dos miembros CPU loopback, incluidas métricas/logs Run en
  vivo; pruebas separadas del proveedor preparado ejecutaron cargas CUDA locales reales mínimas.
  No son pruebas de aceptación SSH/site-command/SLURM reales. No se modificó ni ejecutó WISDOM
  ni clústeres reales.
- No se ejecutó CI en GitHub. Las comprobaciones locales no acreditan las capacidades pendientes
  enumeradas arriba.

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

El driver local posee un coordinador del proyecto bajo `fleets/PARENT_JOB/coordinator` del
JobStore durable, junto a metadata Execution nativa con rutas propias.
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
python -m pytest -q tests/controlplane/test_shard_preparation.py tests/controlplane/test_prepared_provider_dispatch.py
python -m pytest -q tests/controlplane/test_fleet_study_service.py tests/controlplane/test_member_allocation.py
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

La aceptación preparada también prueba staging real, JobService, supervisor directo desacoplado,
workers CPU y una operación CUDA mínima con ARI nativo. Resolver/instalar el entorno son fixtures;
el transporte es loopback, no SSH. El test GPU utiliza acceso local shared explícito e hereda el
grant del supervisor; se omite sin CUDA. No acredita SLURM/gpu exec ni instalación managed real.
Los casos persistentes verifican offers baseline frescos, Job exacto reutilizado y drain sin
liberar el grant entre oleadas. Inputs/grants opacos tienen regresiones separadas.

Las pruebas del proveedor preparado ejecutan cuatro identidades fijas o tres bloques pareados
completos de dos candidatos en dos destinos CPU directos con Jobs desacoplados; ingieren evidencia
centralmente y producen un análisis final único.
Usa ProcessScheduler real local. `test_fleet_study_service.py` prueba además el servicio público con
dos allocations reales, análisis único, lectura de Run en su propietario, cancelación semántica,
request asíncrono capturado y gramática/preflight read-only. Instalar/resolver sigue siendo un
fixture: no acredita SSH real ni aceptación Fleet adaptativa completa. Un fixture vivo mantiene
abierto el Run hasta que el coordinador lee su log aislado y métrica del step 7 en su propietario
exacto. Rechaza seeds/Trials/Attempts/rutas ajenos; los escalares visuales remotos no se convierten
en evidencia completada ni provocan lecturas locales. Las pruebas de inyección
ejercitan el planner adaptativo nativo sin crear otro optimizer. La pausa cubre particiones,
ingestión al terminar, carreras previas al envío, reinicio y presupuestos inmutables.

Los sweeps authored fijos mantendrán desactivada la poda de rendimiento. Los automáticos
mantendrán bloques pareados completos; el HPO adaptativo podrá podar según su contrato actual.
El trabajo predictive futuro pertenece al Study y a sus presupuestos: capacidad libre no autoriza
dependencias downstream no resueltas ni propuestas aleatorias sin información útil.
