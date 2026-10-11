# Salud operativa, monitorización y recuperación deliberada

Español · [English](OPERATIONAL_RELIABILITY.md)

Estado lógico, evidencia científica y salud operativa son conceptos distintos. Un registro antiguo
`running` no demuestra que hoy existan sus procesos. `UNKNOWN` expresa propiedad u observación
sin resolver: no demuestra fallo científico ni autoriza duplicar el cálculo. Se reutilizan el
JobStore del proyecto, los proveedores y el runner nativo. El
[registro de aceptación](OPERATIONAL_RELIABILITY_PROGRESS.md) detalla la reforma aún pendiente.

## Inspección sin cómputo

```sh
lf health --json
lf health JOB --refresh --json
lf incidents --json
lf jobs show JOB --json
lf jobs logs JOB --tail 100
lf monitor diagnose --json
```

Sin `--refresh`, health lee observaciones guardadas. Con él consulta al proveedor. Los tiempos de
intento/éxito del observador permanecen separados del reloj remoto y del progreso científico.
Un fallo conserva estado/fecha de última confirmación y aumenta su contador. Leer caché o escribir
metadata no renueva confirmación. Salida 2 indica incertidumbre/degradación/pérdida operativa, no
el resultado científico. Salud: `healthy`, `degraded`, `unreachable`, `stale`, `lost_confirmed`
o `unknown`. Un fallo de consulta nunca equivale a un inventario vacío válido.

Una entrada ausente sigue desconocida hasta verificar propiedad. En ejecución directa se comprueba
identidad de vida y marcadores de Job del mismo usuario, también en hijos reparentados. Propiedad
inaccesible impide inferir muerte. Observar no envía señales ni modifica state.json del host.
El inventario directo aísla registros individuales corruptos/inaccesibles: los otros Jobs mantienen
su estado y salud actualizados. Sólo el Job conocido afectado queda degradado, conservando un
resultado terminal ya guardado. Las entradas de error no crean Jobs ni acreditan pérdida o permiso
de recovery. Fallos de raíz/transporte siguen invalidando la consulta del host. La siguiente
observación válida limpia el error anterior sin editar historial ni reiniciar ciencia.
Las sondas enviadas seleccionan definiciones/auxiliares por nombre en una sola lectura de fuentes,
no por posiciones de funciones cargadas que pueden quedar obsoletas al actualizar una instalación
editable. Tras actualizar LF reinicia la consola/monitor abierto para cargar la implementación
corregida: no reinicia Jobs científicos ni exige cambiar sus entornos worker inmutables.
SLURM conserva origen queue/accounting, estado bruto, motivo, exit code, nodo y fechas disponibles.
Retraso de contabilidad no es fallo confirmado; `NODE_FAIL` es infraestructura y `OUT_OF_MEMORY`
recursos. Referencias oficiales: [sacct](https://slurm.schedmd.com/sacct.html),
[estados](https://slurm.schedmd.com/job_state_codes.html).
Si `squeue` agota el timeout, una consulta acotada a accounting aún puede aportar evidencia positiva.
Accounting vacío/fallido nunca acredita un RUNNING de caché; se descartan detalles de consultas previas.

## Monitor independiente del proyecto

```sh
lf monitor config --file monitor.json
lf monitor start
lf monitor status --json
lf monitor once --json
lf monitor stop
```

Start separa de la terminal un observador del catálogo nativo, no otro runner científico. Un lock
de propietario evita duplicados. Stop sólo termina ese monitor, no cancela Works. Status verifica
identidad y edad de heartbeat por separado, mostrando degradación ante fallos repetidos. Reiniciar
conserva incidentes e intentos de notificación. No descarga epochs/artefactos, ajusta HPO, reserva
GPUs ni modifica YAML. Configuración/estado/log están en `.operational/monitor/` del JobStore e
incidentes compactos en `.operational/incidents/`; no son otro almacén científico.

Ejemplo distribuido: [monitor.json](../examples/monitor.json).

```json
{
  "interval_seconds": 60,
  "observation": {
    "freshness_seconds": 180,
    "heartbeat_seconds": 45,
    "grace_seconds": 120,
    "progress_seconds": null,
    "unreachable_failures": 2
  },
  "smtp": {"enabled": false}
}
```

Umbrales en segundos, salvo el contador. Progreso se desactiva por defecto: algunos Works válidos
no emiten métricas durante horas. Frescura, gracia y progreso son independientes. Un fallo muestra
conectividad perdida; los repetidos escalan salud sin emergentes en cada sondeo.

Cerrar Textual no detiene el monitor separado. Logout/reinicio requiere servicio del SO:

```sh
lf monitor install --json
lf monitor install --apply
```

El instalador Linux systemd-user no sobrescribe unidades ajenas ni activa lingering ni cambia
políticas administrativas. Revisa unidad y `loginctl show-user "$USER"`. La
[vida del servicio](https://github.com/systemd/systemd/blob/main/man/org.freedesktop.login1.xml)
depende de sesiones/lingering; separar un proceso no garantiza sobrevivir a limpieza del centro.
En macOS/Windows usa `lf monitor run` con launchd/Task Scheduler u otro gestor. Activación real y
supervivencia a logout/reinicio no están certificadas por las regresiones CPU locales.

## Correo opcional e incidentes sin spam

SMTP está desactivado inicialmente. Exige host, remitente, destinatarios y TLS `starttls` o `ssl`.
Credenciales: `env:NOMBRE` o `keyring:CLAVE`, nunca contraseña en configuración. Deben estar
disponibles sin preguntas para el servicio. Un relay TLS sin autenticación también requiere
configuración y autorización explícitas del operador.

```json
{
  "interval_seconds": 60,
  "smtp": {
    "enabled": true,
    "host": "smtp.example.org",
    "port": 587,
    "tls": "starttls",
    "sender": "research@example.org",
    "recipients": ["research@example.org"],
    "username": "research@example.org",
    "credential": "env:LF_SMTP_PASSWORD",
    "cooldown_seconds": 900
  }
}
```

```sh
lf monitor test-alert --json
lf incidents --json
```

Se marca envío tras aceptación SMTP, no al intentarlo. Correo fallido no cambia el Job. Máximo
tres intentos automáticos por transición, con espera persistida. IDs estables deduplican sondeos;
reapertura, escalado y resolución quedan en eventos nativos. Resolver exige observación positiva,
no desaparición. Aceptación SMTP no demuestra entrega en bandeja; una interrupción durante el envío
impide garantizar exactamente una entrega y Message-ID ayuda al receptor a deduplicar.
Un retry auditado resuelve incidentes del origen solo tras progreso científico fresco del Job exacto
aceptado y posterior a su petición. Aceptación, preparación, admisión y heartbeat no bastan. El
resultado original no cambia; `recovered_by` enlaza nuevo Job/petición/progreso. Recovery fallido no
resuelve el origen. SMTP no mantiene el lock de incidentes; una confirmación antigua no marca como
enviada una transición de escalado o resolución ocurrida durante el envío.

## Consola

Bare `lf` → **Operational health…**, o Ctrl+P → Operational Health System: salud guardada,
detalle de incidentes y configuración. Refresh providers es explícito; Start/Stop sólo controlan
el observador. Save settings comparte validador CLI. Send test email muestra aceptación o fallo.
Durante una petición se impide repetirla. Cerrar el panel no cancela ciencia. Overview/Work/Studies
separan salud y estado; los imports son históricos de sólo lectura, sin proveedor activo esperado.

Monitor y consola coordinan sondas por proyecto/proveedor sin espera larga. Una consulta concurrente
conserva registros nativos y relojes originales; receipts no prueban frescura del proveedor. Jobs
SLURM distintos mantienen carriles independientes. Morir el observador libera el lock del SO y no
deja una barrera permanente para consultar.

## Recuperación y YAML trasladado

```sh
lf retry STUDY --dry-run --json
lf retry STUDY --config experiments/trasladado.yaml --dry-run --json
lf retry STUDY --config experiments/trasladado.yaml
# Sólo tras revisar validez de métricas/checkpoints anteriores:
lf retry STUDY --config experiments/trasladado.yaml --accept-code-change
```

**Resume Study…** ofrece ruta histórica, explorador YAML y Validate recovery plan. Falta de YAML
original no provoca replay completo. Se compara identidad científica persistida tras restaurar
namespace de seeds. Comentarios/formato/orden de claves no la definen. Deben coincidir parámetros,
bytes de inputs, datasets, seeds, objetivos, diseño y contratos. Inputs relativos se resuelven
respecto al YAML elegido: conservar sus bindings exactos es imprescindible. Aceptar código nuevo
no permite saltarse identidad.

Se fija hash entre preview y preparación; editar el archivo invalida el plan. Preview no crea
Jobs, consume seeds/presupuesto, reserva GPUs ni descarga checkpoints. Apply verifica de nuevo
propiedad dentro del lock. Propiedad inaccesible o procesos potencialmente vivos bloquean
duplicación. Pérdida verificada reciente permite recuperar sin inventar un resultado failed.

La aceptación genera `recovery_request_id`, enlazado desde eventos del original y metadata/eventos
del nuevo Job. `lf jobs show NEW_JOB --json` expone fase: aceptada, preparación, admisión, runtime
observado, progreso científico observado o resultado terminal. Admisión/epoch cero no son progreso.
Fallos conservan fase/categoría/log. Consola muestra ID y fase observada.

### Historial guardado, incluidos rechazos

```sh
lf jobs recovery JOB --json
lf jobs recovery JOB --offset 100 --limit 100 --json
```

La consulta lee únicamente Jobs y eventos nativos guardados: no contacta al cluster ni descarga
checkpoints/artefactos. Funciona durante una desconexión. Enlaza identidades exactas de retry,
dependencia y ciencia, nunca nombres visibles; muestra dependencias ausentes sin inventarlas.
Los eventos se paginan desde los más recientes; `next_offset` permite consultar los anteriores.

En el Study, **Recovery history** abre la misma vista con Jobs relacionados, IDs, fases y detalles.
Apply y **Validate recovery plan** de consola registran un ID antes de validar: rechazos por
propiedad, estado ausente o configuración incompatible permanecen auditados aunque no exista un
nuevo Job. No cambian Runs válidos ni evidencia científica. Si falla escribir la auditoría, se
informa sin afirmar que quedó registrada. CLI `--dry-run` no modifica ni siquiera esta auditoría.

Un fallo de recovery abre un diálogo persistente con fase, ID, causa acotada y accesos a Logs,
Saved history y Back to plan. Cerrarlo no reintenta y el sondeo no genera notificaciones repetidas.
Un preview válido no es admisión; apply vuelve a comprobar el plan bajo el lock nativo.

Cargar Runs reutilizados establece una referencia, no progreso nuevo. Cambiar totales, reiniciar
contadores o repetir sondeos tampoco resuelve incidentes. Después debe avanzar un contador común
de completados/pasos/epochs con runtime positivamente observado; pérdida o fallo no lo certifican.
El preview nativo conserva el número de Runs lógicos válidos: una primera observación que lo supera
puede acreditar una finalización nueva. Releer el mismo contador nunca renueva la fecha de progreso.

Fixed/repeated recuperan diseño/Run/Attempt propios; adaptive exige estado real de controlador.
Runs válidos se reutilizan, fallidos/interrumpidos crean Attempts, y lo gastado no se devuelve.
No se sobrescriben Execution/Attempts originales. Recuperar Study no garantiza retomar epoch: el
Work debe implementar restauración compatible. Checkpoints gestionados, pins de retención y
transporte exacto siguen siendo autoridad nativa. Recovery Fleet sigue pendiente; no se convierte
silenciosamente a un host.

## Política y evidencia de checkpoints gestionados

En un límite seguro del algoritmo, el Work puede declarar cadencia y retención terminal:

```python
self.checkpoints.configure(
    interval_steps=5,
    interval_seconds=1800,
    retention_seconds=7 * 86400,
    storage_durability="shared",
)
if self.checkpoints.should_save(step=epoch):
    self.checkpoints.save_json("state.json", algorithm_state, step=epoch)
```

No se impone frecuencia ni se serializan algoritmos automáticamente. Cualquiera de los intervalos
declarados puede pedir guardado; avanzan por escritura confirmada, no lecturas/sondeos. JSON y
`checkpoints.file(..., step=...)` registran checksum, propietario Execution/Run/Attempt/identidad,
fecha y paso opcional. `file` reutiliza contenido válido: usa nombres versionados o validación para
reconstruirlo. Escritos mediante `path()` son avanzados y no se registran automáticamente.

`inventory()` pagina metadata; `inventory(verify=True)` comprueba bytes en el host propietario,
sin certificar deserialización. JSON registrado corrupto rechaza carga; legado sigue legible sin
certificado inventado. Antes de cargar/sustituir estado registrado se comprueba Execution/Run
propietario: registro corrupto o propietario ajeno se rechazan sin modificar bytes. Attempts
anteriores del mismo Run siguen permitidos. Compatibilidad científica/código sigue exigiendo
validación nativa y reconocimiento explícito si procede; comprobar propietario no permite saltársela.
Preview/consola muestran último descriptor con integridad **no comprobada**,
sin descargar pesos ni prometer continuar epoch. El campo histórico `resumed_from_checkpoint`
expresa disponibilidad al iniciar Attempt, no restauración confirmada; progreso es otra señal.

Durabilidad (`unspecified`, `ephemeral`, `persistent`, `shared`) es una declaración del operador/
proyecto, no garantía del montaje. Bytes locales al nodo no protegen de su pérdida. Retención se
integra con cleanup preview/apply; fallidos/interrumpidos y pins quedan protegidos. Aumentarla
después del preview se comprueba antes de borrar. Rotación en vivo, deserialización específica y
aceptación Fleet/entre hosts siguen pendientes.

Diagnósticos conservan categoría nativa, fuente, confirmación, momento, impacto y recomendación.
OOM de SLURM no prueba CUDA OOM, ni exit 137 prueba OOM del host. CUDA/dtype genéricos siguen siendo
consumidor. Respuesta malformada es observación degradada, no host desconectado. Una excepción
genérica de consulta conserva conexión desconocida aunque se repita; tipos de conexión/autenticación
se distinguen mediante la autoridad diagnóstica compartida. Un terminal legado
sin request sigue visible, pero no acredita propiedad ni recovery seguro.

## Emergencia

1. Conserva Job ID. Inspecciona health actualizado, detalles, logs acotados e incidentes. No borres
   checkpoints/historial ni mates procesos para ocultar una fila desactualizada.
2. Distingue conexión perdida, procesos perdidos confirmados, scheduler y fallo del consumidor.
   Un host inaccesible puede seguir calculando; retry no autoriza duplicarlo.
3. Restablece conectividad, inspecciona/reconcilia evidencia y pide preview. Elige YAML trasladado
   si procede; revisa reuse/retry/pending y compatibilidad de código.
4. Confirma con propiedad e identidad verificadas. Sigue preparación, admisión, heartbeat y
   progreso del **nuevo** Job conservando el ID original.
5. Conserva diagnósticos/eventos; para infraestructura contacta al centro. Éxitos históricos se
   reutilizan; excepciones arbitrarias del consumidor no se reintentan automáticamente.

El registro de aceptación distingue lo probado de lo pendiente. No acredita SMTP real, CUDA,
un sitio SLURM ni activación systemd. La prueba local nativa de dos Studies/Textual pasa con fallos
simulados del proveedor/admisión; preparación desacoplada real y caos multi-host completo siguen
fuera de esa aceptación demostrada.
