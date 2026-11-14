# Estado técnico de la integración nativa de agentes

[Registro detallado y matriz de requisitos](IMPLEMENTATION.md) ·
[Guía de uso](../MANUAL.es.md#guide-integration).

La reforma integral **no está terminada**. LambdaForge 0.19.0 empaqueta los fundamentos
implementados y las correcciones operativas, no una promesa de capacidades inexistentes.
El punto de partida y las pruebas históricas de 0.18.0 se conservan como tales; la nueva versión
no cambia los requisitos pendientes ni habilita mutaciones científicas no soportadas.
No se ha modificado WISDOM, iniciado trabajos en clusters ni cambiado evidencia científica.

## Resolución de bibliotecas del entorno seleccionado — 2026-11-14

Conda gestionado vincula `native-prefix-libraries-v1` al prefijo de dependencias, también para
venvs de código. Bootstrap aislado `-I -S` hace exec con selección del cargador antes de imports y
site hooks; sin exportar globalmente, rebajar versiones, cambiar imports ni editar prefijos/binarios.
Conserva rutas absolutas heredadas de drivers después del prefijo, omite vacías/relativas y copia
concesiones/visibilidad GPU sin cambios. Supervisor directo arranca con la política antes de Torch;
SLURM recibe el mismo argv y los hijos normales heredan el entorno. Herramientas aisladas retiran
rutas inyectadas del consumidor; `tools.run(env=...)` no sustituye el cargador con política activa.

Preparación/reutilización prueban ambos órdenes, operaciones CPU pequeñas y orígenes C++/GCC
realmente cargados. Procedencia en bootstrap, Job, recibos nuevos y Run. No se reescriben prefijos
inmutables ni evidencia antigua: se reutilizan bases compatibles con una atestación actual y
wheels de código exactos actuales.

Regresiones con ELF locales y versiones CXXABI incompatibles reproducen el error de primera carga
y comprueban la solución. Torch/SciPy instalados se prueban en procesos solo CPU y supervisor/hijo
desechables; `/proc` acredita rutas cargadas. Sin instalar/descargar dependencias ni modificar
WISDOM/datos/Jobs remotos. Bootstrap/retry real del 16 queda al operador, no se afirma validación
del clúster. Una primera suite amplia encontró un timeout de admisión en un fixture GPU ajeno;
la verificación final restringe CUDA a vacío en la invocación de tests. Tres fixtures manuales de
supervisor incluyen ahora la metadata de almacenamiento que ya exige el contrato nativo, sin
cambiar lógica de colas/backend para ocultar su ausencia. Se mantienen APBS 13 y PDB2PQR estable.
Verificación final acotada a CPU: 236 casos pytest correctos y 2 exclusivos de CUDA omitidos;
Ruff y mypy focalizado (12 archivos) correctos. Sin afirmar validación GPU/clúster 16 en vivo.

## Publicación gestionada en prefijo estable — 2026-11-14

Reproducido localmente: renombrar un entorno instalado rompe un console script real de pip y
lanzadores shell/Python (código 127). Dependencias y capas ejecutables se instalan ahora en su
ruta definitiva y solo se publica atómicamente el recibo de listo tras verificarla. Sin sustituir
bytes de binarios/textos ni introducir parches propios del consumidor. El lease de construcción/GC
cubre todo el proceso; micromamba recibe un directorio vacío. Recibos con contrato/ruta y fallos con
diagnóstico redactado; un prefijo sin sellar nunca es activo/reutilizable.

Identidad de entorno v4 y contrato de dependencias v2 usan otro prefijo, sin tocar instalaciones de
Jobs antiguos. Migran en lectura recibos congelados válidos, conservando pins exactos y rechazando
corrupción/incompatibilidad. Builds fallidos conocidos solo se eliminan por GC si no tienen
referencias activas/de recuperación. Interrupciones con propiedad no resuelta siguen protegidas.
Sin cambio de versión de release. Manual §10 documenta bootstrap/retry y limpieza preview/apply.

Pruebas con wheels locales, procesos console/shell reales, inventario/metadata nativa, concurrencia
y locks de construcción/GC; descarga/solve Conda e import de Torch son fixtures. Sigue la regresión
APBS. La prueba de heartbeat fresco toma ahora el reloj al ejecutarse, no al recopilarse, para no
volverse obsoleta durante suites largas. Bootstrap real de micromamba/PDB2PQR y retry de WISDOM no
ejecutados: necesitan autorización del operador. Sin cambios en WISDOM/datos, prefijos remotos,
Jobs ni evidencia científica. Verificación: 202 casos pytest de publicación, sondeo, entornos
nativos/capas, bootstrap/runtimes, almacenamiento y propiedad; Ruff y mypy focalizado correctos.
Comandos documentados de bootstrap/retry/limpieza comprobados con el parser, sin ejecutar backend.

## Sondeo de versiones nativas gestionadas — 2026-11-14

El sondeo común verifica paquete propietario y banner de APBS; únicamente APBS 3.4.1 `--version`
admite código 13. Otros códigos no cero siguen fallando. Bootstrap, envío y reutilización comparten
la comprobación sin omitir contención/pertenencia del ejecutable ni inventario Conda exacto.
Recibos y errores conservan argv, código y diagnósticos acotados. Regresiones con ejecutables locales
desechables cubren éxito/fallo, fallback ordinario, procedencia, escape del prefijo y cambio de build.
Sin instalar paquetes, lanzar Jobs de WISDOM ni modificar prefijos remotos. Despliegue y preparación
real de APBS quedan a cargo del operador tras actualizar LambdaForge en el controlador.

## Recuperación encadenada de Works acotados — 2026-11-12

Corregido el inspector del árbol actual: solo Attempt 1 fija los bindings originales. Los posteriores
verifican sus propios Job/bundle/roles/bytes y parámetros; un propietario distinto debe acreditar
request del mismo emplazamiento hacia la Execution original e historial nativo aceptado. Nunca se
omite un Attempt contradictorio porque otro pase. Se conservan recibos v3/v4 e identidades legadas
compatibles sin reescritura. Evidencia original o linaje ausentes/corruptos siguen bloqueando, con
Attempt/propietarios/rol y causa concreta, no una recomendación genérica de cambiar ciencia.

Preview, consola y worker comparten helper; antes de enviar ciencia se exige capacidad
`bounded-role-chain-v2` en el intérprete preparado. Inspeccionar con helper nuevo no actualiza ni
certifica el runtime antiguo. Preparación gestionada usa wheel exacto actual o bloquea un entorno
existente obsoleto con diagnóstico de actualización/bootstrap.

Pruebas offline de tres recuperaciones reales del fixture entre Jobs distintos, seis roles,
staging renombrado, checkpoints y manifests intactos. Rechazos por bytes, roles, propietario,
linaje, symlinks, traversal, recibos corruptos, parámetros y helper obsoleto; consola usa preview
nativo real y muestra su diagnóstico sin ejecutar un Job.

Los dos dry-run reales (Job original y `job-20261112091733-4cb30fdc`) pasan: Execution
`execution-95fe53f00a13002a`, Run `run-3bed97f306afd66f7dbe`, siguiente Attempt 3. No prueban que
la ciencia vaya a terminar correctamente. No se lanzó recuperación ni se editó WISDOM, historial
o checkpoints. Pendiente aplicar por el investigador con LF actualizado y preparación nativa.
Inventario actual: 450 NPZ geométricos, 450 anotaciones Zn y 449 vistas de superficie; son recuentos
de existencia, no validación científica ni de todos los hashes. Verificación: 82 pruebas focalizadas
y 62 regresiones adaptativas/fijas/retry (con solapamiento). Ruff/mypy en tres interfaces y smoke
del wheel con contrato/helper actual sin importar Torch.

## Identidad histórica de inputs renombrados por staging — 2026-11-12

Contrato `bounded-role-v1`: reconstruye hashes históricos desde bundle/roles/Attempt propietarios
y bytes actuales exactos, sin cambiar globalmente hashes, reescribir manifests o adivinar prefijos.
Preview, dependencias y worker comparten bindings operativos inmutables; el reconocimiento de
código es independiente. Los cambios reales de contenido/labels/configuración siguen bloqueados.

Inspección de `job-20261111182119-e5e63cda`: los seis TXT coinciden con sus recibos originales.
Preview válido para Execution `execution-95fe53f00a13002a`, Run `run-3bed97f306afd66f7dbe`, Attempt 2.
Metadata confirma 323 fuentes, 450 NPZ geométricos, 450 anotaciones y 4.301 archivos estructurales;
5.398 archivos registrados sin faltantes/tamaños inválidos. No se afirman verificadas las decenas
de GB de payloads: acceso nativo/validadores WISDOM deben comprobarlos. Dependencias: 323 presentes
no verificados, cero ausentes y 1.223 registros sin fuente ligada; no prueban compatibilidad universal.

Regresiones con roles/basenames duplicados, YAML trasladado, segundo staging, manifests originales
intactos, cambios de código/contenido/recibos, GC y conservación de productores. No se ha aplicado
retry, desplegado entornos ni editado WISDOM o checkpoints. Pendiente recuperación real autorizada
con LF actualizado. Inventario y límites completos en el registro inglés.

## Dependencias de adquisición y cooldown HTTP — 2026-11-11

Implementados namespaces duraderos por Attempt, protección GC de Runs retenidos entre revisiones,
recomprobación al borrar, bytes/cadenas protegidos y presión explícita. La granularidad es el
namespace, no cada archivo sobrante. Se conservan artifacts recuperables de Works fallidos con mapas.
Referencias de mapas se resuelven solo en namespaces propietarios con digest exacto. Observaciones
HTTP fijan contrato/digest en el Run; su ausencia bloquea la consulta de una URL potencialmente nueva.
Los inputs ManagedFile también se registran bajo spawn; scientific_policy identifica reglas
científicas, no frecuencia de logs/workers. El I/O oculto de subprocesos requiere referencias explícitas.

`lf jobs recovery-dependencies JOB` inspecciona sin payloads pesados; `--apply` verifica/copía solo
bytes exactos desde artifacts propietarios bajo locks nativos, sin symlinks, mover checkpoints o
consultar red. Se corrige la validación con basename temporal sin cambiar hashes persistidos.
Cooldown compartido en filesystem para Retry-After numérico/fecha/hints JSON explícitos; no sustituye
pacing local ni coordina hosts sin volumen compartido. Regresiones locales y smoke de wheel sin Torch.

Caso real inspeccionado: 28.727 referencias ausentes, 28.213 registros sin fuente ligada, cero
presentes/candidatos en el inventario propietario. No se pueden declarar restaurables los bytes
ya eliminados. No se ha reparado ni lanzado ciencia ni editado WISDOM. El bloqueo de negativos
distintos pertenece al consumidor y no se modifica.

Pendiente: despliegue y recuperación real autorizados. Supervisores inmutables antiguos conservan
su GC antiguo; no se actualizan en caliente. Para el snapshot perdido hacen falta bytes con digest
original o una adquisición nueva explícita; nunca mezclar inspecciones legadas con coordenadas nuevas.
No se afirma descubrimiento automático de archives externos ni de dependencias de I/O oculto.

## Recuperación de Works secuenciales — 2026-11-11

CLI y consola conservan Execution, Runs, Attempts y checkpoints de secuencias sin semillas.
Los predecesores restauran outputs persistidos; el worker verifica integridad antes de escribir.
Se mantienen identidad, recursos, costes, locks, propiedad y protección ante limpieza. Paralelos
y composiciones con Studies siguen explícitamente no soportados. Se prueban fallos iniciales y
tardíos, reutilización de mapas/sesión, corrupción, interrupciones, código, presupuestos,
concurrencia, preview sin mutación y confirmación de consola.

El preview real del Job `job-20261110225702-865a3c13` propone Attempt 2 de discover y cuatro pasos
pendientes: 234 checkpoints registrados sin faltantes/tamaños inválidos; sesión verificada con
fingerprint nativo. Mapas: 28.167 descargas, 28.167 inspecciones, 13 metadatos y 5 mapeos negativos.
No se afirman verificadas todas sus dependencias ni su compatibilidad científica: se comprobarán
al restaurar. No se ha ejecutado retry real ni modificado WISDOM o desplegado entornos.
Pendiente: recuperación real autorizada por el investigador. Detalles y límites en el registro inglés.

## Contraseña explícita de transferencia directa — 2026-11-09

CLI --ask-password y campo oculto de consola autorizan entregar la contraseña destino al origen
por el transporte de control SSH cifrado. No se reenvían claves privadas ni secretos guardados
implícitamente. JobService/Scheduler/Supervisor entregan stdin efímero acotado mediante pipes,
nunca solicitudes/argv/entorno/logs. PasswordSshTransport nativo mantiene el mismo motor binario/
comprimido/publicación atómica; el Job del origen no depende del PC tras el acuse. Preview no copia
assets. Se bloquean huellas desconocidas/cambiadas, sin fallback/reintento/sentido inverso ocultos;
retry genérico rechaza esos Jobs porque la contraseña no se conserva.

Verificación: seis pruebas con SSH local real de solo contraseña, datasets diminutos y host keys
temporales, usando un intérprete ya existente con la dependencia opcional, sin instalar nada.
Incluyen copia completa, no persistencia del secreto, fallo/cambio de autenticación, confianza
ausente y prompt CLI. Tres pruebas nativas de stdin efímero y prueba de modal/confirmación/limpieza
superadas. Suites previas: 86 superadas / una omitida opcional; cola/proveedor/consola/CLI: 47 / cuatro;
regresión final retry/input/consola: 27 / tres opcionales. Pasaron Ruff, mypy en doce interfaces,
ayuda CLI y comprobaciones de fuentes/API del wheel construido.

Consulta de solo lectura en ctgpgpu16: Paramiko disponible, runtime inmutable activo aún sin
entrada efímera. **Pendiente:** autorización explícita para crear/desplegar un runtime nuevo del
origen y autorización separada para una transferencia real. No se parcheó ningún prefijo, instaló
dependencia, lanzó Job real ni copió dataset. Sin UI MFA/keyboard-interactive, guardar contraseña,
recuperar secretos o política de descarga inversa con esa contraseña. El PC sigue siendo necesario
hasta entregar el secreto y recibir el acuse del Job en el cluster.

## Recibos compartidos de validación — 2026-11-09

Work.validate_dataset delega en DatasetValidationCache, fuera de la caché por semilla. El runner
aporta su identidad de código ya calculada sin cambiar huellas científicas. La clave fija contenido/
ubicación, código, contrato/versión/ajustes y runtime. Primera llamada o cambios exigen verificar
bytes nativos y callback científico; reutilizar solo inventaría metadata POSIX acotada. Lock y
lease GC nativos evitan repetir trabajo concurrente o eliminar una validación activa. Fallos,
cambios, permisos/rutas inseguros y recibos corruptos no son éxito; force revalida, no ignora hashes.
Solo assets locales v2 con checksums.

Verificación: 17 regresiones, incluido ciclo Work real con dos semillas, concurrencia, corrupción
con mtime restaurado, cambios de código/contrato, reubicación, fallos, leases GC, límites, rutas y
rechazo de URI/legacy. Suite cola/recuperación fija/validación: 78 superadas antes de los últimos
casos de frontera; otras 26 de runtime/storage perezoso/caché HTTP superadas. Ruff, mypy en cinco
interfaces, diff y API pública del wheel sin importar Torch correctos. Solo fixtures temporales.

Pendiente de adopción: WISDOM debe mover la validación estructural profunda al callback y usar
vinculación ligera después del recibo; LF no intercepta llamadas Python arbitrarias. Sin cambios
en fuentes/datos consumidores, runtime actual ni Jobs reales. No acelera en caliente el Study
actual ni evita caducar reservas durante una primera validación larga. Es caché local de detección
de cambios, no prueba criptográfica ante un filesystem hostil.

## Colas de ejecución secuenciales — 2026-11-08

Implementados `lf run --queue`, consulta `lf queue` y selección/tabla de cola en consola. Tickets
FIFO en el host del proyecto/destino; preparación detached normal y el mismo ProcessScheduler /
ProcessSupervisor, sin otro runner. El orden de preparar no adelanta al de registro. La espera
precede a compromisos CPU/RAM/GPU/almacenamiento, exige liberación de propiedad y avanza tras
cualquier resultado terminal. Tras acuse del proveedor no depende de la consola. Duplicados,
runtime antiguo incompatible, metadata insegura/corrupta/ausente y propiedad no verificada bloquean.
Recibos y retención de predecesores conservan el orden tras GC. Recuperar no repite tickets viejos;
un fallo de liberar metadata no sustituye ciencia completada.

Pruebas con metadata temporal, registro concurrente, procesos CPU detached nativos, cancelación y
fallos de preparación, orden inverso, fallo seguido de sucesor exitoso, ausencia de compromisos de
recursos durante espera, CLI/consola, guard de capacidad y retención. Sin clusters reales, WISDOM
ni datasets modificados. Límites: hosts POSIX directos, no SLURM/Fleet; PC durante preparación;
sin recuperación automática/prioridades/colas entre proyectos ni renovar reservas. Manifiestos
acotados y sin compactación administrativa del historial aún. Uso exacto en Manual §10.

Aceptación cerrada 2026-11-09: suite amplia 161 superadas / dos casos CUDA reales omitidos;
después 90 pruebas enfocadas de cola/proveedor/recuperación/storage/consola superadas / dos CUDA
omitidas tras el refuerzo final. Pasaron Ruff, mypy en 13 interfaces, ayuda CLI, empaquetado/imports
del wheel y RPC de metadata distribuido. Fixtures de almacenamiento compartido separan las colas
por cluster; la retención no confunde preparación pendiente con recuperación científica ausente.

Aclaración de alcance 2026-11-09: etiqueta de consola y ambos manuales distinguen la cola de
Studies completos del paralelismo interno de Trials/seeds. Una regresión CPU detached nativa
exige que dos Runs con semillas se solapen antes de iniciar el Job siguiente y verifica que no
cambian los recursos. La suite cola/consola/plan de evidencia pasó 42 pruebas. La inspección de
solo lectura de V14d confirmó ausencia de ticket de cola, tres Runs activas y dos aún validando
entradas consumidoras antes de reservar CUDA. No se cambió ningún Job, grant ni fuente de WISDOM;
el coste de validación inicial no se corrige cambiando la cola ni eludiendo la admisión segura.

## Aislamiento de imports del preflight — 2026-11-08

El fallo de bootstrap SLAE en ctgpgpu16 reveló un conflicto pip/setuptools provocado por LF,
reproducido localmente sin CUDA ni código consumidor. El probe usa metadata stdlib antes de
importar módulos científicos; LF interpreta requisitos y valida versiones con PEP 440 fuera de
ese intérprete. Bootstrap y Work conservan checks comunes de VCS, inventario, ABI, CUDA/grants e
identidades. Sin variables forzadas de distutils, degradar dependencias ni reescribir caché.
Regresiones de import real setuptools sin pip, pins/builds locales, metadata corrupta y preflight
fallido que no queda validado. No se ejecutó bootstrap remoto, reinstalación SLAE ni ciencia real.

## Runtimes científicos aislados — 2026-11-07

Declaración nativa en pyproject, bootstrap/envío preparado y tools del Work permiten prefijos Python
independientes e inmutables sin instalar LF/consumidor, hooks shell ni otro runner. Se implementan
pins/fuentes/ABI explícitos, imports en preflight, scripts pequeños copiados y recibos de resolución.
Se conservan grants GPU, streaming de logs, límite de hilos y cancelación del árbol propio. GC
protege aliases, snapshots Job, propietarios de recuperación y leases de preparación/uso de código
y entorno. El pruning del entorno principal no los borra. Dry-run describe resolución pendiente,
no instala para simularla. Provisioning POSIX; Windows y disponibilidad offline hermética no están
aceptados como capacidades verificadas.

Validación Linux CPU temporal autorizada: SLAE commit
`078b94e436b8eceb02251a254c3c224358ec6ee2` importó y cargó los 882.525.854 bytes del checkpoint
base tras verificar SHA-256 `ac5bf9613209fc310145e58f1aa7ddc9111d833a131488fe30ea738d51f7bb99`.
Receta y versiones observadas en Manual §4.5 / `examples/isolated_tools.toml`. No se instaló en
WISDOM, modificó su código/datos ni lanzó ningún cluster. **Pendiente:** adaptador de extracción
independiente en WISDOM, receta/driver CUDA objetivo y extracción pequeña coordinada con validación
científica de archivos/checkpoints. Provisioning LF no sustituye su import SLAE en proceso actual.

Verificación: 160 pruebas dirigidas superadas y dos casos CUDA reales omitidos explícitamente;
111 pruebas repetidas tras reforzar fuentes/TLS. Pasaron Ruff, mypy (12 interfaces), construcción
del wheel, comprobación de bytes de módulos/receta e imports aislados. La aceptación opt-in
`test_slae_tool_acceptance.py` usa la API pública real con binding temporal y checkpoint explícitos;
CI normal la omite, nunca descarga paquetes/pesos implícitamente. El recibo local ignorado conserva
las 115 versiones resueltas. No se ejecutó extracción, clusters ni modificación de WISDOM,
sus entornos, datasets o checkpoints.

## Capas de dependencias y limpieza recuperable de caché — 2026-11-06

Se amplían los servicios nativos de entornos/storage, sin otro runner ni editar el consumidor.
Los prefijos v3 instalan código exacto sobre una base congelada de dependencias, con inventario y
wheels iniciales de bootstrap conservados. El contrato de dependencias/Python/nativo/Torch/offline,
no cambiar código/YAML/seed, determina la base pesada. Los prefijos antiguos no se reescriben.
Recibos sellados fuera de GC conservan versiones exactas para reconstrucción y rechazan cambios
antes de publicar; entorno preparado/Job registran la identidad resuelta real.
Leases de preparación protegen referencias desde staging hasta el hand-off durable; procesos,
recuperaciones y series retienen código y base correspondientes.

`clean --environments-only` explícito no exige cuota. `storage repair` retira referencias operativas
exactas con auditoría, muerte del controlador probada en su host y descendientes comprobados en el
host instalador; no reescribe ciencia. Estado ausente con request legible protege referencias
concretas; propietario no atribuible permanece protegido y diagnosticado. Preview hipotético no
autoriza borrar. La consola permite seleccionar/revisar/confirmar y explica no-ops. El refresco usa
metadatos cacheados; la medición profunda distingue bloques únicos/enlaces conservados y tamaño
aparente. La cuota sigue siendo de bytes aparentes; el delta físico observado se informa aparte.

Caso real consultado en solo lectura en ctgpgpu16: 91 prefijos (87 completos), 793.224.586.015 bytes
aparentes frente a 695.754.940.416 bytes de bloques únicos de archivos. Contratos/políticas declarados
forman 12 familias, sin certificar igualdad histórica de binarios/transitivas; 49 prefijos contienen
LF 0.19.0/WISDOM 0.13.0. Tres referencias Job sin estado/solicitud y tres builds del portátil pasan
preview con prueba de muerte en ese host y comprobación remota. Plan hipotético: **82 candidatos**,
**631.804.428.288 bytes físicos estimados**, nueve prefijos protegidos. El marker/temp ilegible
b0b58dcb se conserva. Se conserva íntegro `job-20261106115040-ce45a8b3`, su entorno
`env-31506119e900d58a32cd6e97`, bundle, resultados/checkpoints y datasets. JSON local con rutas,
bytes y comandos bajo `.lambdaforge/control/plans/` ignorado, no otro registro científico.
**No se aplicó ninguna reparación, borrado, cancelación, instalación real ni lanzamiento científico.**

Pruebas: fixtures CPU mínimos de wheels, transportes loopback y suites nativas de almacenamiento,
builds, runtime/CUDA/entornos/bootstrap/proveedor/staging/consola. Cubren imports/entry points,
reutilización inmutable, corrupción/host extranjero/PID reutilizado, hand-off, recuperación,
enlaces duros, previews sin mutación y confirmaciones. Ruff/mypy de interfaces cambiadas deben
acompañar la entrega. Límites honestos: no se fuerza la liberación de propietarios ilegibles,
no se deduplican retrospectivamente bytes ni se promete reserva completa de provisioning.

Verificación: la última suite de 156 pasó 154; los dos casos CUDA reales locales agotaron espera
en `peer-startup-future-envelope-headroom` (unos 3 GiB físicos libres de los 4 GiB del portátil).
No ejercitan instalación de dependencias ni se ha debilitado admisión para hacerlos pasar.
Suite acotada CPU/proveedor/storage verificada: **154 pasadas en 53,31 segundos**, incluidos
expulsión/reconstrucción con versiones fijadas y rechazo de corrupción/cambio de resolución.
Ruff/formato de 20 archivos, mypy de 15 interfaces y `git diff --check` correctos. Consultas a
clusters solo lectura; no se afirma aceptación CUDA completa.

## Recuperación de Work único con archivos durables — 2026-11-06

Se inspeccionó el Job remoto fallido en solo lectura: una Execution única sin HPO y una Run
fallida por guarda preventiva de vóxeles, no evidencia de OOM del sistema. El CLI público
`lf retry ... --config ... --dry-run --json` reconoce su diseño resuelto `single` y devuelve
Execution/Run originales, Attempt 2, 1176 recibos (1175 NPZ e informe de fallos; 1.433.721.912
bytes registrados), cero ausentes/tamaño inválido y `payload_hashes: not_checked`. Verifica
identidad científica original, requiere confirmar código y muestra RAM efectiva 96 GiB en vez
de los 16 GiB originales. Sin preparar entornos, enviar Jobs, instalar remoto, descargar
payloads, editar consumidor/evidencia ni publicar datasets reales.

Se amplía el inventario/dispatcher/retry nativo, sin otro motor: misma Run/colección de
checkpoints y nuevo Attempt, último outcome sin borrar fallos/costes. Se verifican proveedor y
lock existente en solo lectura; apply conserva locks atómicos y referencias de protección GC.
Solo se normaliza el namespace visible del single que staging renombró; parámetros físicos,
entradas/dataset, objetivo, políticas y publicación permanecen comprobados. Cambios compatibles
de código requieren aprobación; CPU/RAM/tiempo revisados llegan al Job, al Work y a la auditoría.
Se fija el hash del YAML. **Recover Work…** inspecciona/confirma con feedback antes de enviar.
Presupuestos Run/tiempo de ejecución agotados bloquean recuperación sin devolver costes anteriores.

Regresiones: `work/test_single_work_recovery`, `controlplane/test_single_work_retry` y
`tui/test_single_work_recovery_console`, además de suites previas adaptive/fixed/grants. Fixtures
de Work nativos temporales demuestran reutilización sin reconstruir productores sanos, retry
de fallidos/interrumpidos, corrupción no usada como evidencia válida y publicación solo completa
mediante source_checkpoint. Solo se simula acknowledgement del proveedor para no lanzar cargas.
Pendiente: preview no verifica bytes ni adivina productores pendientes por claves; acceso
gestionado/validadores del consumidor mandan. No promete epochs, Fleet ni migración de cluster.
Apply real necesitará preparar normalmente el runtime LF/consumidor corregido; no se realizó.

## Backward disperso de Entmax15 — 2026-11-04

La consulta remota de solo lectura de los últimos Attempts de V11d confirma 20 Runs correctas
y cuatro fallidas del caso entmax15 (211, 223, 227, 229), rechazadas por pesos no finitos en
SchNet. Hubo además dos Attempts OOM anteriores recuperados: son historia distinta del fallo
numérico terminal. Se reprodujo en CPU que `[0, 0, -2, -2]` produce pesos correctos y gradiente NaN.

Entmax15 mantiene exactamente el forward y calcula el VJP analítico del simplex desde las
probabilidades guardadas, con sensibilidad cero fuera del soporte. La raíz segura del backward
permite derivadas superiores en regiones de soporte fijo, sin afirmar C2 en sus transiciones.
Sin cambios de constructor/state-dict, dtype ni ceros exactos; sin dependencia nueva, softmax,
epsilon, reparación silenciosa de gradientes ni relajar las comprobaciones de SchNet. Regresiones
de forward anterior, Jacobiano, CPU/CUDA y cuatro dtypes, ejes no contiguos, máscaras -inf válidas,
fronteras de soporte, diferencias finitas y optimización sintética mínima. Catálogo regenerado
desde la clase nativa. No se modificaron consumidor, datos, evidencia ni runtime remoto; no se
lanzaron/recuperaron Studies. Falta despliegue gestionado explícito para adoptar el build y un
checkpoint que ya contenga NaN no queda reparado automáticamente.

## Autoridad de evidencia de recursos en vivo — 2026-10-30

La inspección remota de solo lectura del Study reportado encontró inicialización de splits del
consumidor de unos 44 minutos por Run antes de CUDA, no un coordinador detenido. Había nuevas
Runs admitidas; una sin asignación CUDA ocupaba el escalón de packing sin validar y se protegía
una GPU hermana. Memoria libre no demuestra demanda futura del worker que aún inicializa.
No se modificaron consumidor, datos, entorno remoto, Jobs, concesiones ni registros científicos;
no se iniciaron ni detuvieron cargas reales.

El defecto de LF era realimentar al analizador con pronósticos anteriores como si fueran
historia independiente. Ahora solo resultados compatibles registrados aportan residuos
históricos; tampoco se presenta el prior de cold start como historia. El mismo predictor de
placement conserva los prefijos físicos censurados, excluye inicialización sin asignación y
publica una predicción puntual mediante mediana ponderada sin convertirla en un pico completo
ni en un límite OOM obligatorio para vecinos. Se mantienen colas, admisión física, packings
fallidos y exploración protegida. Se añaden conteos de residentes materializados/no materializados,
duración y causa real de espera sin otro scheduler ni más notificaciones.

Regresiones: `tests/hpo/test_live_resource_evidence_authority.py` y suites nativas ARI/dispatcher,
recovery de grants, compromisos compartidos y replay, con fixtures CPU temporales. **Pendiente:**
la inicialización científica del consumidor puede seguir dejando GPU ociosa; LF no omite sus
validaciones ni promete saturación sin evidencia de asignación. Aplicar el cambio a Jobs actuales
requiere despliegue gestionado/recuperación explícitos, no hot patch del entorno inmutable.

## Conectores, tablas y contratos de preview — 2026-10-29

Fuente de verdad: árbol local modificado/no seguido y políticas nativas inspeccionadas, no
GitHub ni payload antiguo. Todo el cambio está en LF. No se modifican/contactan consumidores,
datos del investigador, clusters, concesiones, Jobs o bases de datos reales. Sin otro runner,
HPO, intérprete de grafos, scheduler o registro de datasets.

- CSV/TSV incremental con proyección/límites, opciones explícitas de dialecto/cabecera/encoding/
  BOM/escape/comentarios/saltos/decimales/errores y gzip acotado decodificado. Conserva la ruta
  materializada/path interno; registros malformados de ancho incorrecto fallan salvo skip explícito.
- Column/TableSchema versionados, JSON Schema/ejemplos, casting/nulos constantes/predicados
  tabulares/selección/orden/dedup/concat/agregación con identidad derivada explícita/lineage.
  Materializaciones acotadas, no ordenación externa de disco. No imputer aprendido nuevo,
  tokenizer ni Parquet/Arrow anunciados; conserva el aislamiento train/fold del scaler existente.
- SQLite read-only/transacción/authorizer/timeout/cursor real con fixtures locales. PostgreSQL
  opcional Psycopg read-only repeatable-read/cursor servidor/TLS/referencia secreta/parametrización.
  Su límite driver/redacción se prueba con mock; integración TLS efímera real es opt-in omitido
  al faltar fixture. No se afirma certificación remota ni otros dialectos.
- Escritura SQL bajo política de operador, autoridad insert/update/replace/DDL separada,
  schema/claves/idempotencia y rollback ante fallos/cancelación. No auto-crea DB, raw DDL,
  borrado arbitrario, serialización JSON implícita o retries infinitos. No sandbox Python.
- Preview con presupuesto compartido filas/bytes/fuentes/plazo/retención, raíces/cancelación/
  staging propio y procedencia honesta de prefijo. NPY mmap y NPZ C-order parcial, sin identificar
  contenido no leído. Guardas cooperativas, no cuotas RSS; driver/I/O pueden asignar/bloquear antes
  del chequeo. Rechaza object/estructurados/NPZ Fortran multieje. Sellar un input Work tipado aún
  puede leer su contenido completo: no se debilita identidad científica.
- Catálogo con efectos versionados, schemas de parámetros y referencias secretas, preview nativo
  conservador sin plugins arbitrarios, validación contra firmas/defaults/clases abstractas y
  auditoría de completitud. Inventario: 93 modelos, 1 abstracto/15 estructuras declaradas/77 solo
  firma. Completar ecuaciones simbólicas generales sigue pendiente, sin inventar compatibilidad.
- Reutiliza modelos existentes; contratos y llamadas CPU de ensembles/multitarea/mixtures/
  NeuralODE/calibración describen módulos/estado/salidas múltiples y fit held-out, sin duplicar
  arquitectura ni fingir que sklearn es nn.Module.
- Schema ClusterProfile instalado/versionado, generado de defaults nativos, resolve efectivo
  mediante la misma clase y CLI --kind work/cluster. Fixtures gpu ejecutable/gpu exec/probe
  explícito. No modifica protocolos GPU ni permite claim/release sin propiedad por Job. Otros
  schedulers siguen no soportados y requieren contrato nativo íntegro, no renombrar sbatch.
- Wheel del árbol real con guardas de catálogo/schema actualizado, recibo hash/HEAD/dirty,
  dos builds iguales e instalación no editable validando CSV/SQLite/Work/dataset/sweep/HPO/CLI.
  No se actualiza bundle cliente ni versión de release, no se instalan extras automáticamente.

Manual EN/ES `guide-tabular-connectors`, schemas y ejemplos reales delimitan migración y alcance.
Tests nuevos: preview tabular, SQL, contratos de efectos y receta Work; wheel instalado más
regresiones nativas catálogo/ETL/muestras/Work/authoring/concesiones. PostgreSQL real y skops
ausentes se informan como skips, no como integraciones verificadas; cifras finales en la entrega.

## Lock huérfano y streams binarios — 2026-10-28

Inspección de V10e solo de lectura: fallo anterior a cómputo, marcador de entorno de un proceso
ausente del mismo portátil y ningún instalador remoto del prefijo ni procesos ilegibles.
CacheBuildLease ahora demuestra muerte/reutilización del propietario y ausencia de hijos antes
de retirar exclusivamente metadatos huérfanos por CAS bajo el GC lock. Conserva estados vivos,
ajenos, inciertos/cambiantes, symlinks y entradas inesperadas. No se ha desbloqueado/reintentado
ningún Job real ni modificado entornos, datos, registros científicos o reservas.

Las sondas directas 16↔12 ya fallan por autenticación, no fingerprints: no hay identidades SSH
predeterminadas ni agente en los hosts. No se han instalado claves ni reenviado secretos. El último
fallo relay sí era LF: readline de Paramiko en modo texto frente a receptor binario; se corrige con
lectores ChannelFile públicos en modo binario, preservando los handles durante el cierre.

Pruebas: **177 passed, 1 skipped** en preparación/storage/locks/transferencias/confianza SSH.
La prueba Paramiko omitida por faltar el extra en desarrollo se ejecutó con el intérprete ya
disponible, contra código/tests LF y fixtures temporales: **10 passed** binarios. Sin instalar
dependencias ni copiar datos reales. Ruff/mypy/diff finales completan la entrega.

## Operaciones científicas para clientes visuales — 2026-10-28

Se inspeccionaron árbol local modificado/no seguido, exports/pruebas y adaptadores/guía del
cliente antes de implementar. Se reutilizan las familias de grafos existentes, no se añade
otro runner, HPO, registro ni intérprete de nodos. No se modifican WISDOM/AIStudio, datos o
Jobs reales, no se instalan dependencias externas ni se cambia la versión de release.

- Implementado: catálogo SDK/CLI JSON versionado, schemas distribuidos, plugins por metadatos,
  resolución selectiva, alias/errores y fábricas clase/kwargs con procedencia/dimensiones del receptor.
- Implementado: muestras NPZ/NPY/JSON/CSV multiactivo/sidecars/targets/metadatos sobre el índice
  nativo, límites/preview perezoso, opcionales/presencia, collate stack/ragged/pad/grafos disjuntos.
- Implementado: operaciones ETL Python, joins estrictos/particiones/validación compartida,
  scaler solo train/fold y descarga con checksum/cancelación/límites/referencias HTTPS del host.
- Implementado: estimadores sklearn opcionales, capacidades/semillas/hilos/procedencia y API de
  persistencia skops sin pickle. La prueba round-trip se omite explícitamente si falta skops;
  el diagnóstico de dependencia ausente sí se comprueba. No se afirma haber probado ese backend.
- Verificado: ejemplo Work ETL con checkpoints/publicación atómica, dos entradas, variantes de
  campos/componentes, sweep y HPO nativo; interrupción sin dataset parcial y reutilización nativa.
  Construcción offline doble con mismo hash e instalación no editable temporal: SDK/CLI y
  prepare/sweep/HPO fuera del checkout. Ruff/mypy y regresiones de interfaces/modelos existentes.
- Pendiente explícito: ecuaciones machine-readable completas para **todos** los operadores
  neuronales, validación simbólica general, índices paginados masivos, joins many-to-many y codecs/
  tokenizers arbitrarios. Los contratos incompletos se marcan sin inventar compatibilidad.
  Límites temporales cooperativos no son preempción de I/O; labels train los declara el consumidor.
  No se despliega ni se modifica el bundle de Studio: requiere instalar/fijar el wheel revisado.

El capítulo de authoring científico del manual EN/ES, los JSON/YAML/Python reales y el registro
detallado inglés delimitan la implementación. Tests: catálogo, muestras, ETL, Work/download,
wheel instalado y regresiones nativas HTTP, plugins, componentes/grafos/imports.

## Arquitectura conservada

Instrucciones de repositorio y consumidor, Skills procedimentales empaquetadas, fachada Python
fijada al proyecto, MCP oficial opcional y memoria Markdown son capas distintas. Todas las
consultas operativas/científicas delegan en servicios nativos; no existe otro runner ni un backend
que interprete salida humana de CLI. AGENTS inglés breve es canónico; sus contratos detallados
siguen en FRAMEWORK_CONTRACTS.md y la explicación española no define una segunda política.

Las 24 Skills tienen referencias y ejemplos nativos, y se han validado con skills-ref oficial.
Init/sync trabaja con recursos del paquete instalado, mantiene personalizaciones y no cambia
la configuración global del cliente. Sin flags, init conserva el comportamiento científico anterior.

## Permisos y aprobación humana

READ/PLAN es el valor predeterminado. La disponibilidad de una operación no significa que esté
habilitada en la sesión. Las acciones aprobadas son crear/editar/enlazar notas y exportarlas
en ZIP, bajo WRITE_PROJECT. Preview no escribe; request guarda una propuesta sin aprobarla.
Una persona revisa los valores exactos desde CLI TTY o Console y escribe el identificador completo.
Solo después puede el agente aplicar ese plan por separado.

Los registros firmados vinculan proyecto, argumentos, revisiones, hash y caducidad. Claim previo
bajo lock, auditoría y revalidación impiden replay concurrente. Un crash tras el claim no permite
repetir automáticamente. MCP no expone aprobar; texto recuperado y `approved=true` no autorizan.
Esto protege la frontera del protocolo, no un programa con acceso shell independiente al mismo
usuario del sistema operativo. La continuación 2026-10-12 añade cancelación de Job exacto;
resto de SUBMIT/OPERATE/DESTRUCTIVE científicos siguen pendientes.

## Transporte, dependencias y memoria

SDK oficial 2.3: stdio predeterminado, protocolo limpio y planificación importadora aislada con
timeout/cancelación. HTTP local es opt-in, READ/PLAN, token privado, loopback/Host/Origin exactos,
comprobación por petición y límites. No es OAuth ni permite exposición de red pública.
Dependencias opcionales requieren consentimiento; solo extras declarados en virtualenv/Conda,
con modo wheelhouse sin red. Los fallos no borran archivos/notas.

Markdown de once tipos es canónico; SQLite FTS incremental es reconstruible. Ediciones externas,
revisiones, notas corruptas y duplicados se diagnostican sin convertir interpretaciones en ciencia.
Consultas relacionadas usan enlaces/tags/evidencia exacta. Verificar referencias consulta registros,
no interpreta resultados ni descarga pesos. Obsidian headless funciona sin escritorio o plugins;
búsqueda semántica y vault externo todavía no están implementados.

Exportación de notas ZIP con bytes exactos, manifiesto checksum y revisión del inventario completo:
rechaza corrupción, duplicados, symlinks, cambios y sobrescrituras; nunca incluye ciencia/plugins.
Auditoría nativa de intención/finalización con ID/revisiones, sin títulos/cuerpos y paginada por
bytes; una intención interrumpida no acredita éxito. Trece prompts MCP y recursos de contratos
instalados, índices y resultados/productos exactos respetan toolsets y no autorizan ejecución.
Ayuda Obsidian distingue diagnóstico pasivo de prueba CLI/apertura local explícita, sin instalar
plugins ni alterar preferencias. Headless/ausencia/fallo/timeout se probaron mediante fixtures;
no se acredita una sesión de escritorio real ni configuración de vault externo.

## Verificación realizada

- 67 pruebas focalizadas: protocolo real, HTTP en socket, aprobaciones, aislamiento, pip real
  fallando sin red en venv temporal, CPU Work/evidencia/memoria, Console, estructura y versión.
  Se comprueba también el cambio externo entre validación nativa y escritura antes de commit.
- Codex 0.162.0-alpha.17.2 real descubre herramientas y llama project_info desde otro directorio;
  configuración temporal aislada, sin inferencia ni credenciales y sin tocar ajustes existentes.
- Todos los modelos públicos resuelven declaraciones/docstrings sin importar Torch. Consultas
  seleccionadas de Runs, productos, incidentes y recursos usan los servicios nativos y límites.
- Ruff sobre src/tests y mypy sobre 622 archivos pasan. Wheel 0.18.0 instalado y probado fuera
  del checkout; no depende del MCP para uso normal. CI configurada, no afirmada como ejecutada.
- Pytest completo (iniciado antes de las últimas regresiones): 1930 correctas, 1 fallo y 4 avisos,
  1195,10 segundos. El único fallo era la versión ausente en el enlace español AGENTS; se corrigió
  y pasaron las cuatro pruebas de versión y el gate final de 67 pruebas. No se repitió toda la
  suite tras esa corrección: esta ejecución completa no se presenta como totalmente verde.

## Lo que queda por implementar o acreditar

Operaciones científicas autorizadas; resto de herramientas/recursos/prompts y planificación nativa;
semántica local/híbrida; vaults existentes/externos autorizados y aceptación real de Obsidian;
UX de recuperación del journal ante escrituras interrumpidas; GUI ampliada; un segundo cliente real;
conversaciones completas con ciencia CPU y memoria entre sesiones; gates finales completos.
Descubrimiento MCP real no equivale a una conversación de agente ni a autorización científica.
El registro inglés conserva los siguientes pasos para continuar sin repetir trabajo verificado.

## Continuación — 2026-10-12, base `8df2a1c`

La petición nueva 1–65 sigue en curso; no se aumenta versión ni se declara finalizada.
Envelope v1 compatible, diagnósticos y aprobaciones con códigos precisos y esquemas por operación
validados; errores MCP con `isError`. El SDK 2.3 publica mediante contratos públicos de modelos,
sin cambiar managers privados. Descubrimiento técnico EN/ES determinista añade propietario,
permisos/frescura/errores/siguientes herramientas; no es búsqueda semántica de memoria.

Primer OPERATE: cancelación de Job exacto mediante ApprovalService firmado y JobService nativo.
Preview sin proveedor/escrituras, revisión humana CLI/Console, revalidación de binding/estado,
claim único y resultado guardado. Incertidumbre, cambios y Fleet bloquean; se preservan hermanos
y ciencia. Permiso de notas no autoriza cancelar. Controles de aprobación fijos fuera del scroll.
Se extraen adapters nativos de resultados y acciones Job, sin cambiar la fachada pública.

Las 24 Skills tienen procedimientos/casos específicos, ejemplos comprobados con parser nativo
y herramientas reales de fachada. CI crea venv explícito y fixtures de instalación ya no
dependen del SDK ambiente; formato normalizado sin cambios de AST.

Verificación: **97 pruebas focalizadas correctas**, incluidas todas las de agentes con SDK
oficial, cliente Codex real aislado opt-in y estructura/versión. Ruff pasa; **869 archivos**
con formato correcto y mypy sobre **628 fuentes**. Las 24 Skills pasan `skills-ref` oficial y
validador de creación; ejemplos contrastados con parser/fachada sin ejecutar acciones.
Wheel 0.19.0 construido/instalado sin dependencias en entorno temporal propio y smoke fuera del
checkout. No se afirma pytest backend completo, ciencia remota, conversación de segundo cliente
ni aceptación de escritorio real.

Siguiente frontera: resto de operaciones autorizadas y ejemplos ejecutables; memoria contextual/semántica
opcional, vault/importación/conflictos, GUI/Apps/tasks, monitor y contratos checkpoint/Fleet.
No se han cambiado WISDOM, clusters reales ni evidencia persistida, ni se promete soporte pendiente.

## Assets de reports y ownership tras reinicio — 2026-10-13

`outputs.html_bundle` conserva archivos opacos completos como artifact nativo, entrada acotada y
manifiesto atestiguado. Loader público aislado, transferencia de buffers verificados, visor loopback
de lectura con URL privada y transporte existente comprimido. Concurrencia/plazos/cancelación y
descarga de iframe al cambiar pestaña; liberar el renderer propio corresponde al consumidor.
Export/import relocatable conserva todos los bytes y usa el mismo inspector. Regeneración mediante
ResultInput/Work desde inferencia retenida, sin repetir entreno ni forward passes. Guía EN/ES,
schema y ejemplo empaquetados; `html_section` conserva su contrato.

Sondas nativas de lectura de ctgpgpu16 confirmaron reinicio posterior a los propietarios guardados.
Sesiones de acceso nuevas inaccesibles bloqueaban recovery por error. La sonda usa arranque/vida;
marcas vivas, fechas desconocidas y propietarios posteriores inciertos siguen protegidos. Preview
de v6b/c conserva 47 Runs cada uno; v6f conserva un Run completado. v6d estaba completado; v6e/g
no habían creado Execution científica. Zn conserva checkpoints, pero no es Study y no se certifica
su continuación individual por esta vía. No se inició recovery remoto ni se modificó ciencia/WISDOM.

Verificación: 132 regresiones focalizadas y 29 pruebas finales de HTML gestionado/legacy, con
Chromium real, asset de 70 MiB, 206 registros, float32/NaN/máscaras exactos, aislamiento,
corrupción/cancelación/plazos y export/import relocatable. Cuatro casos Work local/remoto pasaron
también tras ampliar parametrización; 86 pruebas de ownership/proveedor/consola correctas. Ruff y
formato (686 archivos) pasan; mypy sobre 631 fuentes. Wheel 0.19.0 instalado/probado fuera del
checkout en venv temporal reutilizando dependencias existentes, sin descargar dependencias nuevas.
No se afirma suite completa, productor remoto real de bundles ni recuperación científica remota.
La reforma anterior de agentes sigue pendiente según fronteras arriba; no se aumenta versión.

## Continuación de robustez de recursos — 2026-10-13, sin nueva versión

Observación GPU nativa con caché UID/host/boot acotada, NVML oficial opcional, helpers aislados
con plazo/recolección, calidad measured/unknown y circuitos UUID. Visibilidad CUDA numérica se
fija una vez mediante identidad del driver aislada, sin contexto de cómputo; la lectura ordinaria
no importa Torch ni inicializa CUDA. Concesiones UUID/MIG no requieren esa consulta del driver.
Admisión shared/command usa el planner existente con compromisos anteriores al spawn,
reconciliación residente, fairness acotada y perfiles operativos validados. Un sondeo fallido
pausa arranques, no termina hermanos sanos.

Capacidad física/cgroup, PSS/USS/RSS, afinidad/cuotas y límites de threads del hijo protegen
presupuestos RAM/CPU conocidos. Leases exteriores por host/boot y UUID GPU/MIG; liberación y
visibilidad usan identidad retenida, no reinterpretan ordinales. Propiedad antigua ambigua queda
protegida con explicación en cola. Descubrimiento GPU fuera del lock. Historial operativo con
ventanas comprimidas, picos muestreados y recuperación ante temporales interrumpidos, sin tocar
evidencia científica. Informe guardado sanitizado usa servicios nativos CLI/consola/fachada READ.
Claim/release genéricos inseguros siguen explícitamente no soportados, sin mutación del proveedor.

Verificación: **293 pruebas** en gate amplio focalizado; **211** en gate final solapado de recursos/
proveedores/ARI; **58** regresiones Work; **59** del gate final leases/perfiles/proveedor; **9** de
consola seleccionadas. Ruff y diff pasan; mypy comprueba **641 fuentes**. Benchmarks CPU sintéticos
y wheel 0.19.0 instalado/probado fuera del checkout, sin descargar dependencias. Protocolo MCP
oficial omitido por SDK ausente, no aceptado. No se afirma suite completa ni NVML/MIG/gpuctl/SLURM reales.

La reforma completa **no está terminada**. Continuar desde `docs/ai/IMPLEMENTATION.md#history-resource-robustness-audit`:
predictor RAM/compromisos globales sin demanda declarada, backpressure PSI/procesos, compromisos
por fases/reconciliación tras crash, reconciliación nativa de leases antiguos, informes históricos
hardware completos y aceptación autorizada de proveedores. No repetir los bloques ya verificados
ni modificar WISDOM/evidencia científica para superar esas pruebas.

## Ampliaciones incrementales de Studies fijos — 2026-10-14, incompleto

La petición adjunta **no está completada**. Estado detallado en
[STUDY_EXTENSIONS.es.md](../MANUAL.es.md#guide-study-extensions); no se aumenta versión.

Implementado y probado:

- Comparación nativa de diseño fijo: ampliar una dimensión rectangular o añadir seeds explícitas,
  IDs candidato/celda independientes del orden, ordinales originales conservados, tipos numéricos
  exactos, delta validado por WorkConfig y no-op. Se rechazan cambios mixtos/eliminaciones,
  condicionales, auto-replicación, HPO adaptativo/composición, sin usar retry como ampliación.
- `StudyExtensionService.preview`: Execution/design/Attempt propios exitosos, inputs e identidades
  científicas exactas, objetivo declarado, compatibilidad específica de código y metadata CPU
  guardada. No cambia CodeIdentity global, IDs Run, configuración original, recovery o coste gastado.
- Prueba Git de HEAD limpio original idéntico y solo YAML elegido modificado; digest nativo de
  fuentes sin Git. Cambios, symlinks, otro proyecto, propietario UNKNOWN, evidencia incompleta/
  corrupta y dependencias distintas se rechazan. Metadata CPU nativa no atestigua cada dependencia
  arbitraria/hardware/runtime; el apply amplio sigue bloqueado.
- `StudySeriesStore` interno: primera reserva en envelope atómico, hashes de revisiones inmutables,
  historial append-only encadenado, idempotencia/concurrencia y metadata propia acotada. No inventa
  Jobs/Executions. Lock nativo protege frente a borrado; ResultStore rechaza evidencia referenciada
  y autoridad corrupta/ausente. Orígenes con Job se rechazan hasta integrar su retención/cleanup.
- CLI `studies extend` hace preview por defecto; `--apply` se rechaza antes de cargar target/Work.
  show/history usa metadata propia del servicio. Ninguna operación pública crea series todavía.
  Python preview devuelve `apply_supported=false` y `series_persisted=false`.

Verificación: **89 pruebas** en el gate final diseño/preview/reservas/recovery fijo/borrado;
**13** en gate adicional solapado de borrado/runtime. Ruff y diff pasan; mypy comprueba **8 fuentes
tocadas**. Origen CPU escalar con 20 celdas usa Work/Run/Attempt nativos; solo se inlina la frontera
de procesos CPU en fixtures desechables. No se afirma scheduler real/cluster. Todavía **no se
ejecutan** las 15 celdas delta: aceptación real 20+15=35 no está superada. Se registrará abajo el
smoke de wheel cuando termine. Sin descargar dependencias, mutar clusters o cambiar WISDOM.

Continuar desde estos gates, sin repetir la base verificada:

1. Materializar delta conservando contexto/rutas nativas y atestar mejor runtime/fuentes;
   no convertir metadata coincidente en permiso para mezclar dependencias no demostrables.
2. Handoff estable request/Job por SubmissionService, reconciliación de crashes y revalidación
   apply; sin otro runner ni reenvío ciego de reservas inciertas.
3. Transiciones inmutables y ampliaciones sucesivas; proteger Job/storage delete y compactación,
   export/import y productos antes de habilitar esas referencias ejecutables.
4. Proyección acumulada/adaptador StudyAnalysis con IDs/procedencia originales, seeds emparejadas,
   costes físicos, interpretación exploratoria y fingerprint/cache nuevo de evidencia/análisis.
5. CLI apply/analyze y consola no bloqueante con selector/confirmación/historial y agrupación
   científica conservando Jobs visibles. Skill de Study e integración segura READ/PLAN.
6. Ejecución real CPU 20→35/20→32 mediante submission, recovery de delta fallido, concurrencia,
   inyección de crashes, ampliaciones sucesivas y aceptación single-cluster/proveedor autorizada.

Fleet, migración arbitraria y waiver silencioso de código siguen no soportados.

Gate de wheel superado fuera del checkout: wheel 0.19.0 construido sin aislamiento/index/
resolver dependencias e instalado únicamente en target temporal propio. Smoke importa servicio/
registro instalados, valida comparación nativa 20→35 y parser CLI, y comprueba ambas guías
empaquetadas. Se reutilizan dependencias existentes, sin descargar/instalar dependencias nuevas.
Esta comprobación de paquete no certifica ejecución del delta ni soporte remoto.

## Continuación de experiencia de desarrollo — 2026-10-15, pendiente

UX incremental sobre HEAD `8df2a1c`, conservando cambios sin commit de recursos/informes/ampliaciones.
Versión **0.19.0**, sin aumento. La reforma UX amplia **no está terminada**;
[DEVELOPER_EXPERIENCE.es.md](../MANUAL.es.md#guide-developer-experience) detalla comandos/límites. SHA-256 confirma
los siete archivos protegidos servicio/diseño/journal/CLI/tests byte-idénticos al checkpoint inicial;
config diff solo reutiliza sus helpers JSON puros.

Implementado: nombres/instrucciones/tests reales de scaffold; schema instalado, explicación estática
y diff nativo; ayuda bilingüe/completion del parser; inspect/diagnose/watch acotados con autoridad
nativa; filtros Work/Study con apertura exacta y scroll/snapshots conservados; `ResultStore.table`
JSON/CSV paginado con procedencia, selectores, último/todos los Attempts y snapshots censurados;
contexto/doctor nativos, preview oficial Codex y launch TTY; Skills Work/Study con ejemplos completos.
Registro verificado no es handshake ni conversación real. No se amplían permisos READ/PLAN.

Pendiente: elegibilidad de recovery consolidada, streams por Run, búsqueda global de entidades
separada con debounce/páginas, sorting/paging completo, extracción de Workspace, comparaciones
compatibles/emparejadas mediante StudyAnalysis/AnalysisProfile, integración remota de tablas,
editor/merge conservador, copiado de recetas, navegación/accesibilidad HTML offline, otras Skills,
metadata común de acciones/capacidades y contratos MCP tipados. Faltan aceptación real SDK/cliente/
configuración ajena, shells/editor aislados y benchmarks repetibles antes de afirmar finalización.

Verificación: **116 pasan** en CLI/cliente/assets/aislamiento/salud/regresiones de ampliación;
**50** en semántica/resultados Work; **16** en gate final solapado de tablas/inspección/ejemplo Skill.
Gates anteriores solapados: **72 pasan, 1 deseleccionada**, **9** de navegación. Ruff/diff limpios;
mypy comprueba **19 fuentes tocadas**. Ambas Skills pasan validador skill-creator; ejemplo valida
WorkConfig fixed/repeated/adaptive y todas sus líneas CLI sin ejecutar ciencia. No se afirma
suite completa de backend ni CI GitHub.

Wheel 0.19.0 sin aislamiento/index/resolver dependencias, instalado en target temporal propio;
consumer generado construido/instalado aparte sin dependencias. Fuera del checkout, comandos ligeros
instalados no importan Torch/consumer; fixture CPU escalar nativo y catálogo/tabla exacta pasan.
No se cambia WISDOM, clusters, GPU/SSH/SLURM, registro/launch Codex real, configuración global ni
versión. Tests de registro usan estado simulado aislado. Continuar desde pendientes sin repetir lo
comprobado, sin convertir estado terminal en permiso de recovery ni comparaciones en confirmación.

## Checkpoint histórico: primer delta ejecutable de Study fijo — 2026-10-16

La sección de aceptación posterior sustituye el alcance pendiente de este checkpoint histórico.

La petición de finalizar ampliación cómoda de **WISDOM remoto no está completada**. Este checkpoint
implementa primer delta **CPU local** explícitamente revisado y diálogo común de consola. Cluster/
GPU, revisiones/análisis acumulados, reconciliación completa de crashes del handoff y export de
serie siguen pendientes. Solo para ese alcance local sustituye al checkpoint anterior de lectura.
Sin aumento de versión, cambios en WISDOM ni operaciones sobre clusters reales.

Implementado: revalidación del hash revisado, revisión inmutable con checksum e historial de
handoff, ID nativo de Job preasignado, snapshot privado adyacente que conserva rutas relativas/nombre,
bytes e identidad científica fijados en preparación, supresión de duplicados/concurrencia y
`studies refresh` que contrasta las celdas delta completadas. Job/Execution/Series conservan IDs
distintos. Borrado e historial/GC protegen referencias; identidad final debe coincidir con el envío.
Falta de autoridad/handoff incierto bloquea replay; no se inventa otro runner ni un Job ficticio.

**Extend Study…** preselecciona YAML original, permite explorador, muestra conservados/nuevos/total
y valores/seeds añadidos, invalida revisión al cambiar selección y confirma una sola vez. Carga,
errores (incluido SystemExit al importar consumer) y aceptación son visibles. No modifica el YAML
original; recientes muestran el authored YAML y no el snapshot privado. Acciones Study en grid de
dos filas. Importados son de lectura; remoto/GPU se rechaza explícitamente. Deltas/series ya enviadas
no se aceptan como nuevos orígenes para repetir celdas. Procedimiento Skill actualizado mediante
skill-creator y validado.

Verificación: **46 pasan** en gate diseño/preview/apply/consola; **69** en gate solapado de apply/
storage/borrado; **82** en storage/submission/recovery/borrado; **22** en consola recovery/historial/
productos; **31** de authoring/assets. Gate final de **9** cubre además código cambiado durante
preparación, rutas relativas, nombre implícito, concurrencia/handoff y SystemExit visible. Fixture
CPU ejecuta 20 Runs originales + 15 nuevos disjuntos y conserva bytes originales; solo se inlina
frontera preparación/procesos del proveedor. No certifica envío desacoplado real ni GPU. Gate inicial
amplio: 137 pasan / 1 fallo asíncrono de historial; reruns separados pasan. Test del diálogo espera
condiciones, no presupone que una pausa de UI completa el hilo. Ruff/diff limpios; mypy comprueba
14 fuentes. Se añadirá abajo resultado de wheel instalado al completarlo.

Continuar por procedencia managed/GPU en el mismo placement y ejecución delta verificada antes de
anunciar ampliación de WISDOM en cluster. Después: autoridad/costes acumulados, agrupación/análisis
y retención/export de serie. No usar Resume, reinicio completo ni ediciones manuales del journal
para suplir esos gates. No repetir la base local de delta/consola ya comprobada.

Smoke de wheel instalado superado fuera del checkout: API/CLI apply/refresh revisados, servicio/
journal nativos, diálogo y ambas guías empaquetadas. Build/install solo en target temporal propio,
sin index/resolver dependencias ni nuevas dependencias. Certifica empaquetado, no ejecución real
desacoplada del proveedor. Validación final reutiliza checks nativos de inputs/Run de recovery y
compara el fingerprint del entorno registrado del delta.

## Aceptación de series incrementales de Studies fijos — 2026-10-16

Implementado el flujo revisado mediante CLI nativo y Research Console: editar el YAML original,
abrir **Extend Study…**, revisar Runs conservados/nuevos y confirmar el plan exacto. SubmissionService
envía solo celdas nuevas disjuntas. Las revisiones sucesivas comparan el diseño acumulado, no solo
el origen: fixtures CPU ejecutan **20→35→56** y **20→32**, conservando bytes originales e identidades
exactas de Execution/Run/Attempt.

La ejecución gestionada en un cluster inspecciona evidencia autoritativa en el host original y
reutiliza su entorno inmutable conservado mediante preparación/scheduler/lanzador nativos. Antes
de ciencia comprueba compatibilidad del runtime y tipo de GPU concedida; ordinales/cantidad no
son identidad de hardware y nunca se amplía la visibilidad del proveedor. Metadatos remotos
paginados con checksum, sin descargar modelos/checkpoints/artefactos ni historiales de epochs.
Se mantienen checks de código, inputs/datasets, objetivo, seeds, propiedad y presupuesto acumulado.

La consola agrupa revisiones opt-in en una serie con Runs/métricas acumulados, navegación a Jobs
nativos, revisiones e historial. Análisis reutiliza StudyAnalysis con procedencia exacta y etiqueta
exploratoria post hoc, no confirmación basada solo en supervivientes. Export conserva paquetes
nativos independientes, journal y JSON/HTML acumulados. Borrado/GC protege Jobs, Executions y entornos
referenciados.

Reconciliar un crash de handoff reutiliza el Job preasignado solo si no existe request/ejecución
del proveedor. Autoridad ausente o incierta bloquea; apply duplicado devuelve el Job aceptado.
Un delta fallido se recupera mediante retry nativo explícito de su Execution, conservando celdas
correctas y costes de Attempts y siguiendo su genealogía exacta. Las métricas en vivo no selladas
son una capa visual, no evidencia científica aceptada silenciosamente.

Verificación: **128 tests pasan** en el gate de ampliación/diseño/consola/retry/export/assets de
agentes; **63** en storage/propiedad/borrado; **70** en rutas remotas, envío coordinado, TLS y acceso
GPU. Ruff pasa; mypy comprueba 19 fuentes modificadas. Skill de Studies validada. Wheel **0.19.0**
construido e instalado en target temporal propio sin resolver dependencias ni acceder al índice;
fuera del checkout pasan smoke checks de CLI, servicio, consola, hooks del runtime conservado y
guías bilingües empaquetadas. Sin aumento de versión, cambios en WISDOM ni operaciones en clusters reales.

Límites: origen propio correcto, sweep fijo rectangular y seeds explícitas; cada revisión añade
valores a una dimensión existente O añade seeds. No se admiten ampliaciones adaptativas,
condicionales, compuestas o Fleet, migración, entornos remotos no gestionados, GPU local ni import
ejecutable de Series. Continuar entrenamiento desde checkpoint requiere soporte del consumer.
Aceptación remota mediante proveedor loopback aislado y procesos CPU nativos; contratos GPU con
metadatos sintéticos, sin afirmar aceptación real SSH/SLURM/CUDA ni WISDOM. Véase la
[guía de ampliaciones](../MANUAL.es.md#guide-study-extensions) para flujo y límites exactos.

## Refinamiento de ampliaciones condicionales/selectivas — 2026-10-16

La limitación rectangular anterior era real: orígenes condicionales fallaban antes de comparar.
Configuración nativa admite puntos escalares completos adicionales mediante `sweep.include` y
selecciones dispersas exactas mediante `sweep.candidates`, mutuamente excluyentes. Ambos conservan
activación de ParameterSpace; claves inactivas ausentes. Include une puntos al diseño base, no
expande sus productos cruzados. Candidates no enumera el producto cartesiano. Sin esas claves se
mantienen variantes/identidad nativas de YAML existentes.

La ampliación fija compara candidatos activos/seeds exactos, rechaza quitar candidatos históricos
o cambiar contratos de activación y genera solo su delta nativo disjunto. No repite controles.
Una nueva combinación conjunta se ejecuta aunque todos sus valores marginales ya se hayan observado.
Añadir seeds sigue siendo otra revisión. No cambian envío, autoridad de revisiones, análisis
acumulado, costes, checks de código/inputs ni registros anteriores.

Verificación: **148 tests pasan** en gate de regresión con ejecución delta condicional nativa,
revisiones sucesivas, consola, recovery y dispatch loopback con prefijo conservado. Suite dispersa
ampliada: **15 pasan** (solapados), incluyendo schema y ausencia de enumeración cartesiana. Ruff
y mypy pasan en seis interfaces fuente; schema/diff válidos. Cuatro fixtures sintéticos cubren
recuentos 60+32, 48+28, 48+32 y 96+28; no certifican los registros V6 reales de WISDOM. Sin cambios/
envíos WISDOM, pruebas GPU/SSH/SLURM reales ni aumento de versión.

Límite operativo: instalar el controlador no añade sintaxis a un entorno remoto inmutable antiguo.
Preview comprueba soporte nativo del parser antes del handoff y dispatch lo vuelve a contrastar.
Falta de soporte bloquea con diagnóstico explícito; no se implementa transición reconocida a otro
runtime compatible ni se modifica ese entorno. No prometer ampliación remota de V6 hasta que su
preview real revisado pase. Siguen excluidos adaptativo, Fleet, migración, composición y valores
estructurados.

Gate final tras consolidar validación de puntos mediante ParameterSpace: **57 pasan** entre
comparación dispersa/condicional, series acumuladas nativas y dispatch con prefijo conservado
(solapados con el gate anterior). Wheel final 0.19.0 construido sin índice/resolver dependencias;
fuera del checkout pasan include → delta disperso exacto, schema y ambas guías empaquetadas.
Instalación solo en target temporal propio; el smoke no ejecuta ciencia.

## Corrección de orden original en ampliaciones fijas — 2026-10-16

Inspección de lectura de V6a WISDOM `job-20261011132247-5c2dec2e`: las claves JSON ordenadas
regeneraban 12 de 15 posiciones de Trial distintas. El snapshot propio de la Execution tiene
valores JSON exactamente iguales y reconstruye la huella original sin discrepancias de Trials.
El lector usa ese snapshot acotado, regular y propio solo tras comparar exactamente sus valores
con la configuración congelada; siguen activos huella nativa y checks por Attempt. JSON ordenado
explícito conserva el orden al paginar. Preview/completion local y lector remoto enviado como
fuente comparten comportamiento; el YAML vivo editado no es autoridad histórica.

Regresiones: orden condicional multidimensional, edición del YAML vivo, paginado remoto loopback,
distinción int/float y snapshot ausente/corrupto/modificado/symlink. Gate preview/series/remoto:
**42 tests pasan**; gate ampliado orden/apply/diálogo: **16** (solapados). Ruff y mypy pasan sobre
tres fuentes modificadas. Sin aumento de versión.

El lector real de V6a verifica ahora **60 Runs correctos / 64 Attempts gastados** sin reescribir
evidencia ni enviar cálculo. El preview real siguiente sigue bloqueado por compatibilidad de
código: el Git original estaba dirty y su digest difiere del árbol dirty actual, aunque HEAD
coincide. No se omiten checks ni se migra código/runtime; sin cambios WISDOM ni apply. Esto corrige
el falso cambio de configuración de la captura, no certifica que V6a actual sea enviable.

Gate final combinado orden/preview/apply/series/remoto/diseño/disperso/consola: **81 pasan**.
Sin cambios adicionales en estado científico.

### Ampliaciones fijas con revisiones de código (2026-10-16)

El bloqueo posterior por código era real, pero demasiado restrictivo para ampliar deliberadamente.
Ahora `code_policy="revisioned"` por defecto conserva el origen inmutable y permite una revisión
revisada del proyecto/runtime actual. No declara equivalencia. Las ampliaciones compatibles
mantienen su comportamiento; `--code-policy strict` conserva la equivalencia obligatoria.
Parámetros fijos, contenido, datasets, objetivos, seeds, diseño y Attempts correctos siguen
verificándose. Solo se ejecutan celdas nuevas, sin devolver coste de fallos ni tiempo gastado.

El remoto gestionado utiliza preparación nativa de bundle/entorno/scheduler, sin parchear prefijos
históricos. Código y declaración de dependencias quedan fijados durante la preparación desacoplada.
Un preflight nativo en el host compara el delta realmente desplegado y registra su identidad
científica aparte del pin del controlador. No modifica coordenadas de seeds: distinto namespace
del host no autoriza cambiar seeds literales/diseño. Checksums de configuración/código, recogida
y retry usan esa acreditación. Los lectores eligen el Python científico registrado para ambos
preflights de ampliación, no el Python del login por accidente.

La serie conserva procedencia y métricas originales. Revisiones sin equivalencia acreditada no
se mezclan en medias de seeds ni análisis de efectos. StudyAnalysis opera por revisión; por defecto
se muestra la última, JSON incorpora `code_cohort_analyses` y HTML incluye pestañas nativas
aisladas **Code · revision-N**. La semántica original de métricas permanece con su evidencia.
Consola advierte antes de confirmar **Extend · current code** y muestra política por revisión.
Es un historial único, no un experimento reiniciado ni evidencia atribuida falsamente a otro código.

El preview real de lectura de V6a `job-20261011132247-5c2dec2e` permite **60 conservados + 32
nuevos = 92 celdas**, manteniendo **64 Attempts gastados**, con `revisioned-current` y despliegue
gestionado propuesto. No se modificó WISDOM/evidencia ni se envió ninguna ampliación real.
El proveedor loopback desechable ejecuta worker/preflight/Runs CPU/recogida nativos, con namespace
nuevo y código actual, y rechaza seeds desplegadas alteradas. La preparación del proveedor/entorno
se intercepta: no demuestra instalación gestionada real, SSH ni GPUs reales.

Verificación: **149 pruebas pasan** en el gate combinado ampliaciones/orden/recuperación/consola/
informes y **1 prueba adicional** de revisiones de código sucesivas pasa. Ruff y mypy pasan en
fuentes modificadas. Se comprueba el wheel instalado fuera del checkout y sin instalar dependencias.
Sin aumento de versión. Siguen explícitamente fuera HPO adaptativo, Fleet, migración, GPU local y
despliegue de código en entornos no gestionados. HTML por revisión mantiene límites 16/64 MiB;
muchas revisiones pueden necesitar informes separados, no documentos embebidos ilimitados.

### Corrección de preflight grande y retry anterior a Execution (2026-10-16)

Diagnóstico de lectura de la ampliación V6a `job-20261016125643-ce4bcdd6`: falló el controlador
en `prepare_native_delta`, no Work de WISDOM. Enviaba como argv SSH un contrato JSON compacto de
539.412 bytes (530.026 de semántica de métricas). Password SSH recibió connection reset en esa
llamada. No había ID de scheduler, request remoto del proveedor ni resultado Execution nuevo;
la revisión correcta original no se repitió ni alteró.

Preflight envía ahora compromisos SHA-256 canónicos por campo y los recalcula íntegramente en el
host. No ignora/trunca campos ni identidades; mantiene tipos numéricos exactos, nombres de campos
en discrepancias, huella científica nativa y checks de configuración/código. El preflight real
**de lectura** de la configuración ya preparada pasó con un comando SSH de **5.747 bytes**, sin
preparar guard, enviar al proveedor ni llamar a Work.run.

Retry nativo conserva propiedad de la ampliación y pin de envío aunque falle antes de existir
Execution. Reintenta solo preparar ese delta, conserva genealogía/evidencia previa y no inventa
continuación de checkpoints. La búsqueda local filtra manifiestos por Job exacto y no selecciona
otro Study original. Recovery de una Execution existente mantiene su acreditación de preparación.

Verificación: **106 pruebas dirigidas pasan**, con AnalysisProfile nativo >600 KB, rechazo de
cambios en descripciones/parámetros/seeds, tipos canónicos exactos, validación de lectura, retry
de preparación mediante WorkRunner/recogida nativos, rechazo de duplicados, cohorts/prefijo
conservado y recuperación adaptativa/fija. Ruff y mypy pasan. Sin aumento de versión, edición de
WISDOM, reescritura científica ni retry real.

Decisión pendiente para recuperar en vivo: el pin del código WISDOM actual ya difiere del enviado.
Es independiente del fallo SSH diagnosticado. No se renovó ninguna aprobación implícitamente;
retry mantiene el código revisado protegido. Aprobar código diferente para esta ampliación
pendiente o reenviar el código preparado conservado no se implementa en esta corrección focalizada
y requiere un workflow revisado explícito, no editar journals ni reiniciar el experimento completo.

## Consolidación del manual, clases de clustering y FGW genérico — 2026-10-16

Las 32 guías/auditorías anteriores se integran sin perder contenido sustancial: material público
con índice en `docs/MANUAL.md` / `docs/MANUAL.es.md`, e informes históricos archivados debajo,
sin presentarlos como validación actual. En `docs/` quedan solo los dos manuales y cuatro referencias
internas esenciales. Paquete, enlaces, metadata de agentes y las 24 Skills apuntan a capítulos
conservados. Skill-creator valida las Skills; las regresiones comprueban enlaces y anclas.
Se conservan procedimientos especializados, ejemplos y límites de autoridad.

Las estrategias y contratos/resultados de clustering tienen un módulo por clase, manteniendo
importaciones públicas y comportamiento. `lambdaforge.distances` añade `StructuredMeasure`,
`FusedGromovWassersteinDistance` y `FGWResult`: POT opcional, FGW q=2 balanceado, atributos arbitrarios
mediante distancia explícita o matriz precalculada, pesos normalizados, unidades explícitas,
inicio factible, acoplamiento/marginales y convergencia local. Reservas densas protegidas sin tensor
de cuatro índices, lógica específica de proteínas, clustering ni certificado falso de óptimo global.
Un job CI opcional comprueba el solver real.

Aceptación local: **66 pruebas dirigidas pasaron**, con objetivos numéricos/marginales reales de
POT, identidad/permutación, pesos desiguales/cero, atributos estructurados, errores/iteraciones,
imports opcionales, memoria, clustering existente, documentación/ejemplos, Skills y API neuronal.
Ruff check/format y mypy (21 módulos) pasaron; también las 24 Skills con el validador de skill-creator.
Wheel limpio construido en temporal, smoke instalado fuera del checkout y ejemplo FGW instalado
ejecutados correctamente. POT solo se instaló en el destino temporal expresamente autorizado.

La comprobación global adicional tuvo **5 éxitos y 1 fallo**: cuatro clases previas carecen de
docstring (`GPUObservationPolicy`, `ResourceSafetyPolicy`, `ExtensionReply`, `SeriesReply`). No se
modificaron esas implementaciones ni se afirma que esa prueba pasara. Sin cambios en WISDOM,
clusters reales, evidencia científica, entrenamientos o entorno normal del proyecto.
Se mantiene 0.19.0: no se solicitó un incremento de versión.

## Diagnóstico y corrección del dispatch con GPU ociosa — 2026-10-17

La inspección de solo lectura de runtimes/metadatos reales de WISDOM confirmó dos fallos distintos:

- `wisdom-v6g`, `job-20261016165002-1ab71d38`, cluster 12: 66 completadas, 54 pendientes, ninguna
  activa y unos 79 GiB libres en la H100 concedida. El planner aprobaba trabajo, pero el evento
  `RESOURCE_LAUNCH_WAIT` daba `study_ram_allocation_headroom_or_unknown`: la cuota calculada sin
  auxiliares se contrastaba después contra todo el árbol y bloqueaba la primera Run de reemplazo.
- `wisdom-v6h`, `job-20261016132142-9e6d1052`, cluster 16: 110 completadas, 104 pendientes y dos
  activas en los tokens concedidos `1`, `3`, con unos 121 GiB libres en cada H200. Las estadísticas
  conservaban rachas positivas de startup tras 39/47 avances reales sin nuevo pico material.
  Seguían como RAMPING, impidiendo promover la exploración; además una cola rara de GPU completa
  se interpretaba como compromiso determinista de toda la memoria por residente.

La cuota elástica derivada se reconcilia con la misma medición fresca del árbol que usa el
lanzamiento, sin rebajar requisitos independientes ni límites del Job/físicos/cgroup. Avances
planos reales cierran rampas antiguas; polls idénticos no. Una envolvente terminal compatible válida
puede limitar un compromiso estable, conservando muestras/pesos de cola rara para el riesgo conjunto;
sin historial no se inventa una cota, ni se borran límites OOM. El check final conserva el planner,
contexto de GPU hermanas y comparación WAIT aprobados, sin reiniciar evidencia de espera.
`launch_waits` y motivos de inactividad distinguen propuesta y arranque real; historial deduplicado.
Sin cambios en candidatos, seeds, objetivos, diseño ni concesiones GPU.

Validación: 127 pruebas dirigidas de recursos/runtime/recovery/documentación y otro conjunto nativo
más amplio de 189 pruebas pasaron (hay solapamiento). Ruff check/format y mypy de los tres módulos
modificados pasaron. Solo fixtures locales desechables y consultas remotas de lectura, sin entrenar.

Límite operativo: el código no cambia entornos inmutables ni controladores ya arrancados. El usuario
autorizó cancelar/reanudar solo el 12 y mantener el 16. Se canceló nativamente
`job-20261016165002-1ab71d38` sin Runs activas; el preview verificó YAML, ciencia y código consumidor
sin cambios: conservar 66 éxitos, reintentar 3 y mantener 51 pendientes, con 72 Attempts gastados y
costes intactos. No hizo falta forzar reconocimiento de código cambiado. El retry nativo creó
`job-20261017102915-3bb4b922`, con entorno inmutable nuevo `env-72dd487020659c94443a910f`;
se verificó por lectura que contiene el código corregido de lanzamiento/cuota. La misma
`execution-27fc9197382c151f` pasó a dos Runs activas en la H100 concedida y conservó los 66 éxitos.
Sin edición manual de WISDOM, evidencia, checkpoints o configuración del cluster. El Job original
del 16 no se canceló, reinició ni parcheó; su controlador ya arrancado necesita otra autorización
futura de recovery nativo para incorporar el fix. Identidad y controles de recovery siguen mandando;
no se promete continuación de epoch sin soporte del Work.

## Archivo histórico de auditorías

Informes conservados de iteraciones anteriores; fechas y pendientes son históricos, no validaciones nuevas.

<a id="history-architectural-reform"></a>

## Reforma arquitectónica: auditoría y registro de implementación

[English](IMPLEMENTATION.md#history-architectural-reform)

Este documento permite continuar el trabajo; **no declara terminada la reforma solicitada**.
Base: `d2f526d`, LambdaForge 0.17.0. La reforma no modifica WISDOM, trabajos de clusters productivos
ni registros científicos persistidos.

Continuación sobre `c8f2d2e`: la [auditoría científica](#history-scientific-audit) corrige orden de
evidencia pareada, recovery del compromiso anticipado y coherencia conclusión/estabilidad. No cierra
los pendientes de lifecycle, asignación host aprendida, dependencias de productos o Fleet distribuido.

<a id="history-architectural-reform-hallazgos"></a>

### Hallazgos

- `work/runner.py` reúne ejecución, coordinación, clasificación de reintentos, terminación,
  resúmenes científicos y despacho de recursos en unas 10.000 líneas. La extracción debe separar
  responsabilidades con pruebas previas, no limitarse a mover bloques.
- Los resultados describen Attempts físicos. La recuperación ya conserva los resultados lógicos
  válidos de diseños adaptativos, repetidos y fijos. Un diseño fijo no necesita un controlador
  adaptativo ficticio; uno adaptativo requiere su estado real.
- Terminación y retry clasificaban de forma distinta: un genérico `CUDA error` parecía fallo de
  recursos, aunque errores de acceso ilegal o tipos de tensor no son fallos de asignación.
- La telemetría sustituía failed por retrying antes de archivarlo. Perdía el fallo visible y
  contaba reintentos en espera como procesos activos. Ocho entradas no explicaban el coste físico
  acumulado de una recuperación larga.
- En la base faltaban contratos, ProductRegistry, dependencias e importación portable. Ya existe
  la capa de contrato/registro/transporte de productos detallada abajo; siguen pendientes
  orquestación de dependencias y export/import Fleet completo. Decisiones nativas, inputs tipados y
  publicación declarada de Study individual ya están implementados.
- Ya existen preflight, candidatos reconstruidos sellados, comparación por el proyecto y recovery
  de publicación de datasets. En la base faltaban certificados y resolución por contrato;
  certificados durables ya reutilizan productos, resolución Work explícita sigue pendiente.
- Fleet permite Studies nuevos con planificación central, miembros gestionados, leases y fencing.
  Siguen pendientes recovery público distribuido, miembros dinámicos, co-location, transferencia de
  checkpoints y exportación completa. Los controles internos no equivalen a aceptación pública.
- En esta base existían leases y limpieza con preview. Sigue pendiente admisión completa de
  provisioning; la actualización de 2026-11-06 descrita arriba separa dependencias/código
  conservando prefijos antiguos y procedencia numérica.

<a id="history-architectural-reform-base-implementada"></a>

### Base implementada

- `diagnostics/failure.py` clasifica de forma pura la terminación, reconocimiento de OOM,
  elegibilidad de retry por worker perdido y diagnósticos persistidos. `FailureDisposition` usa
  `ErrorCategory` y `RetryDisposition` existentes. Elegibilidad **no autoriza** retry: ARI,
  presupuestos, compatibilidad y dominancia del placement siguen gobernando la admisión. Mencionar
  memoria o un worker perdido en un error del consumidor no autoriza un reintento ciego.
- `work/state.py` deriva `StudyState` con `operational`, `evidence` y `health` a partir de las
  obligaciones y estados lógicos existentes. `required_evidence()` se comparte entre telemetría
  viva y terminal. Candidatos adaptativos no propuestos no crean deuda ficticia. Pruned sigue siendo
  evidencia censurada, no un objetivo final inventado ni un fallo operativo.
- `work/attempt_history.py` archiva Attempts terminales antes de transiciones lógicas. La cola
  visible queda acotada; los contadores acumulados de Attempts, fallos y duración sobreviven.
  Historias antiguas truncadas se identifican como límites inferiores (`history_complete=false`),
  no como información exacta inventada. Los resultados y presupuestos autoritativos no cambian.
- Las proyecciones compactas de Study/overview transportan `lifecycle` y la consola lo muestra sin
  pedir ficheros de Attempts. Un retry en espera no ocupa un slot de proceso activo.
- Recovery elige el mayor número de Attempt de cada celda lógica: una respuesta antigua retrasada
  no sustituye un éxito recuperado. Los distintos niveles de fidelidad siguen separados.

Se conservan `status`, versiones de resultados, decisiones HPO, placement ARI, hashes exactos y
locks. Esta primera base **todavía no** migra todos los consumidores de ejecución final, análisis,
Fleet y dependencias al contrato agregado de lifecycle.

<a id="history-architectural-reform-integración-adicional-y-correcciones-de-higiene"></a>

### Integración adicional y correcciones de higiene

Finalización y referencias a outputs usan los últimos resultados lógicos, no `all(run.ok)` sobre
fallos físicos antiguos. Resumen/objetivo no duplican Attempts de la misma celda. Agregación nativa
de evidencia comparte autoridad con telemetría, distingue fases/fidelidades, admite diseños repetidos
sin Trial parametrizado y conserva la deuda de cada Work compuesto. Los registros físicos observados
permanecen; historial omitido por recovery sigue siendo incompleto.

ResultStore adapta lifecycle antiguo en memoria; metadatos de Analysis y export lo transportan sin
cambiar fingerprints científicos. No se han migrado todavía todos los lectores de HTML, análisis
cacheado, recovery o Fleet. Faltan historia física completa, obligaciones dinámicas y overhead.

Los previews creaban raíces y locks de GC/controller; comprobar una caché podía crear un lease.
Inspección, previews de GC/entornos y compactación ya no los crean. Los leases existentes se consultan
sin escribir. Apply conserva locks y recalcula objetivos: un preview orientativo no autoriza borrar.
Propiedad imposible de inspeccionar protege la caché. El modo read-only no se ha probado en Windows/NFS.

Borrado nativo de ResultStore exige estado terminal y ownership persistido, serializa con import
Study y toma el lock del controlador hasta retirar los archivos. Preview no crea locks. Apply
retira solo el padre Work propio si queda vacío, con operaciones sobre descriptor de raíz sin
seguir enlaces. Conserva hermanos/contenido ajeno, roots y productos publicados independientes.
Snapshots importados running son evidencia borrable. Falta auditar todos los demás caminos de
limpieza de padres propios vacíos del proyecto.

<a id="history-architectural-reform-capa-de-modelo-registro-y-transporte-de-productos"></a>

### Capa de modelo, registro y transporte de productos

`products/models.py` separa significado versionado explícito, contenido exacto y provenance
original con modelos profundamente inmutables y pickle-safe. `products/registry.py` publica copias
independientes verificadas con SHA-256 crudo, nombres/contratos inmutables, locks nativos y lecturas
de metadata acotada. Lecturas/previews no crean nada. Configuración del productor es provenance,
no requisito de compatibilidad; nuevos orígenes no sustituyen al original.

`lf products list/show/provenance/consumers/verify/publish/export/import/select/decide/status/finalize` llama a esa API. Los bundles verifican
manifiestos, bytes e inventario completo de provenance antes de import idempotente. Export → borrar
original → import concurrente conserva bytes promovidos e identidad histórica. Véase
[Productos](../MANUAL.es.md#guide-products). `products/selection.py` selecciona snapshots de modelos completos, de
últimos Attempts y fidelidad completa, con métricas del artifact, grupos, constraints, top-k y
empates explícitos. Solo lee/promueve modelos seleccionados; no atribuye best/last de una Run a
pesos arbitrarios. `outputs.from_checkpoint(..., metadata=...)` registra evaluación declarada por
el proyecto; un Study CPU mínimo prueba esta ruta. StudyDecision nativa conserva selección e identidad
exacta del Analysis persistidos sin refit y con preguntas pendientes explícitas. Inputs product
tipados comprueban contratos/expectativas; bundle fija contenido, worker recibe raíz propia del
proyecto y Attempts reales registran consumo inmutable. Preflight remoto comprueba materialización
sin descargar pesos. Un Study individual declara `products`: guarda ciencia y después publica antes
de compactar. Estado acotado e historial de publicación conservan fallos; `products finalize` solo
reintenta publicar bajo locks de propiedad, sin repetir entrenos. Esto **no** implementa promoción
compuesta/Fleet, espera/replanificación ni export completo Fleet.

`lf import PACKAGE [--apply]` verifica/registra exports versión 2 de Study de un host, incluidos
productos sellados exactos. Execution/Run/Attempt y procedencia originales quedan intactos en
archivo portable; ubicación se registra aparte y nunca ejecuta ni sirve como recovery nativo.
Export local/provider incluye productos de recibos publicados. Consola Products pagina metadata y
auditorías, con verify/export explícitos e import Study con feedback, revalidación y confirmación.
No es import/export Fleet distribuido, autenticidad criptográfica ni orquestación automática.

`DatasetEquivalenceCertificate` sella la comparación existente de todo el dataset en el mismo
ProductRegistry, como ScientificReport versionado. Fija IDs exactos, contrato completo, verificador,
política/resultado/evidencia; fechas y rutas son provenance operativa. Comparaciones no resueltas,
corruptas o realmente distintas fallan de forma cerrada. La API no fusiona identidades, infiere
transitividad ni modifica resolución de inputs Work ordinarios. Informes demasiado grandes fallan
explícitamente, nunca se recorta aprobación. Véase
[Reconstrucción de datasets](../MANUAL.es.md#guide-dataset-reconstruction). Falta resolución YAML por contrato.

Constructores de ficheros/checkpoints y lecturas de caché sin lease de worker ya no crean árboles
vacíos de datos/records/locks. La primera escritura conserva publicación atómica y lock por clave.
El lease explícito de Work activo sigue siendo anticipado para seguridad frente a GC; adquisición
completamente diferida del lease queda pendiente. Publicación de raíz de checkpoints rechaza la
intención inválida antes de comprobar existencia, sin necesitar crear una carpeta vacía.

Imports públicos reutilizan `LazyExports` para Work, productos, HPO, métricas, clustering y Analysis.
Leer contratos/catálogo o ayuda CLI no importa Torch, runner ni Analysis científico. Ejecución,
resultados/export y planificación cargan servicios en sus rutas respectivas; nombres y módulos de
clases no cambian. No hay otro runner ni capa de compatibilidad. Es corrección concreta de frontera
de startup, no la extracción todavía pendiente de responsabilidades executor/ARI.

<a id="history-architectural-reform-pendiente-por-orden-de-dependencia"></a>

### Pendiente, por orden de dependencia

| Secciones solicitadas | Frontera | Estado / aceptación pendiente |
| --- | --- | --- |
| 1–3, 63–64 | Auditoría y extracción | Auditoría inicial; faltan executor, despacho/retry y persistencia con caracterización previa. |
| 4–9, 46–47 | Estado, fallos, recovery, presupuestos | Base de estado/fallos/finalización integrada; faltan lectores restantes, taxonomía, obligaciones dinámicas, coste físico/overhead y self-healing completo. |
| 10–18, 31–32, 50, 54–56 | Productos, contratos, selección, dependencias | Contrato/registro/promoción, ModelSet local/StudyDecision nativa, inputs Work, raíces/consumidores, publicación YAML de Study individual y retry solo de publicación implementados; faltan selección compuesta/remota/Fleet y espera/replanificación/DAG. |
| 19–20 | Import/export portable | Import/export nativo de Study/productos de un host, verificación e idempotencia implementados; faltan inventario/transferencia completos Fleet y aceptación distribuida. |
| 21–24 | Datasets y equivalencia | Reconstrucción conservada; certificados durables mediante comparación/catálogo nativos implementados. Faltan política explícita de representación adicional, reports escalables como artifacts y resolución Work downstream por contrato. |
| 25–30, 60 | Filesystem, storage, entornos, caché | Previews/raíces diferidas, borrado terminal con locks, capas inmutables de dependencias/código y reparación auditada/contabilidad física implementados; faltan auditoría amplia restante y admisión independiente de bytes de provisioning. |
| 33–42 | Lifecycle productivo Fleet | Pendientes pause/resume/reconcile/adoption, miembros/drain/capacidades, co-location y transferencias reanudables verificadas. |
| 43–45 | CLI, consola y HTML | Overview lifecycle, productos/auditorías e import Study confirmado integrados; faltan dashboards recovery/selección/dependencias y explicaciones completas. |
| 48–49, 51–53, 59 | Compatibilidad, lecturas acotadas, seguridad | Conservar contratos; auditar migraciones, pureza de dry-run, idempotencia, locks y stores de cada feature. |
| 57–58, 65–70 | Aceptación y entrega | Faltan escenarios A–D y validación final después del resto de implementación; no declarar finalización global. |
| 61–62 | Documentación y migración | Registro inicial bilingüe; las rutas y ejemplos finales deben corresponder a implementaciones funcionales. |

No presentar features pendientes mediante comandos ficticios, identidades adivinadas, reinicios
silenciosos, equivalencia de datasets sin certificado o export Fleet que omita miembros.

<a id="history-architectural-reform-punto-de-validación"></a>

### Punto de validación

Antes de extraer: 69 tests de caracterización pasaron (observabilidad, obligaciones y diagnósticos).
Los mismos 69 pasaron tras la integración inicial. Las nuevas regresiones cubren OOM frente a
errores del consumidor/kernel, frontera de worker, disco lleno, respuestas retrasadas, historial
acotado con coste acumulado, recuperación y obligaciones ausentes/censuradas. Registrar resultados
finales en la entrega; este documento no afirma por adelantado que la aceptación final pase.

Validación de primera base: Ruff/mypy y suite completa **1454 passed**, con cuatro warnings previos
de Lightning, antes de productos. Smoke de wheel instalada y un Work CPU pequeño pasaron fuera del
checkout con prefijo nuevo y dependencias existentes reutilizadas. Regresiones de modelo/registro/
CLI/transporte y selección: suite completa **1508 passed**, anterior a decisiones/inputs tipados.
Checkpoint de productos/almacenamiento diferido: **84 passed**; integración más reciente:
**237 passed**, con publicación, decisión, inputs, recovery, obligaciones y providers. Ruff/mypy
pasaron. Suite completa posterior: **1554 passed**, cuatro warnings previos Lightning, antes de
import Study/consola. Wheel fuera del checkout pasó publicación CPU → consumo → borrado productor →
import producto, con prefijo nuevo reutilizando dependencias existentes. Regresiones de import y
provider/consola añadidas; registrar resultado final en entrega. Loopback no equivale a aceptación
SSH/Fleet productiva ni a terminar los escenarios A–D.

Checkpoint de continuación: corrida completa **1588 passed / 3 failed** (cuatro warnings previos
Lightning). Los tres fallos eran doubles de tests que suponían WorkRunner eager en globals o la
ruta de evidencia anterior a import; corregidos, los tres pasan al repetir. Otra integración
dirigida de 48 tests cubrió esos módulos. Regresiones de certificados: **20 passed**, con rechazo
por cualquier miembro, corrupción, tolerancias explícitas, transporte inmutable/pickle, traslado,
límite de informe e integridad actual separada de aprobación histórica. Ruff/mypy pasan
(**580 archivos fuente**). Wheel actual fuera del checkout pasó Study CPU de dos seeds: export →
borrado del original → import, conservando Analysis, provenance y pesos. Pantallas Products/import
headless renderizadas e inspeccionadas. No equivale a otra corrida completa limpia ni a aceptación
final de toda la reforma.

<a id="history-scientific-audit"></a>

## Auditoría de corrección científica

[English](IMPLEMENTATION.md#history-scientific-audit)

Base: `c8f2d2e`, LambdaForge 0.17.0. Se cierran las correcciones de orden de evidencia y coherencia
de conclusiones descritas aquí, **no toda la auditoría de ingeniería solicitada**. No se modifica
WISDOM, ningún cluster real, manifests científicos históricos ni GitHub CI.

<a id="history-scientific-audit-defectos-confirmados-y-correcciones"></a>

### Defectos confirmados y correcciones

| Clasificación | Autoridad / comportamiento anterior | Corrección y evidencia |
| --- | --- | --- |
| CONFIRMED_BUG | `SequentialSweep.evaluate` ordenaba la intersección por número de seed; un bloque nuevo podía reordenar observaciones ponderadas anteriores. | Exige `seed_order` persistido, consume exclusivamente su prefijo completo y registra las seeds utilizadas. Pruebas con ProjectSeedStream real, llegadas invertidas y celdas iniciales ausentes. |
| CONFIRMED_BUG | `runner._execute_fixed_evidence_group` reiniciaba `committed_lookahead=None` al recuperar. | Restaura el compromiso de `sweep-blocks.json` mediante validación compartida de solo lectura; impide un segundo bloque especulativo. La integración conserva Runs, manifest original y coste físico tras interrupción. |
| CONFIRMED_BUG | La telemetría de bloques mostraba mínimo/máximo completo aunque hubiera huecos. | Publica únicamente el prefijo completo continuo. El inventario inicial se persiste antes de lanzar, incluso sin ningún resultado terminal. |
| CONFIRMED_BUG | `WorkRunner._summary` contaba solo obligaciones iniciales y podía declarar éxito con una celda anticipada comprometida pendiente. | Recovery y finalización comparten `recovery.fixed_requirements`: bloques creados siguen siendo obligatorios sin reescribir el diseño. Integración conserva tres Runs válidas y muestra la cuarta pendiente, sin reiniciar ni esconder deuda. |
| CONFIRMED_BUG | ScientificDesign convertía predicciones sin soporte a `UNRESOLVED` conservando la estabilidad de otra hipótesis. | La conclusión descriptiva sin soporte tiene cero realizaciones propias; predicción, estabilidad predictiva y motivo permanecen separados. Cobertura no se llama confianza. |
| CONFIRMED_BUG | `_exact_conclusion` y análisis de sweep atribuían `1-probabilidad_modal` al evento no resuelto; una distribución vacía tenía estabilidad uno. | Usan la masa real del token no resuelto o cero sin realizaciones. Un líder numérico débil no transforma equivalencia observada en un producto de probabilidades de otro evento. |
| CONFIRMED_BUG | Study Analysis interpretaba cualquier `stop=true`, incluido `INCOMPLETE`, como resolución formal. | Lector compartido exige una decisión científica; una parada operacional sigue sin resolver. Ausencia de procedimiento secuencial es `None`, no aprobación formal. |

El lector remoto de retry integra el mismo validador puro desde su fuente autoritativa, como los
lectores existentes de recovery/seeds. Una regresión lo ejecuta con Python aislado (`-I -S`),
sin LambdaForge instalado: no equivale a aceptación SSH real.

`ScientificQuestionState` separa resolución descriptiva y formal. Su accessor histórico `resolved`
conserva aplicabilidad descriptiva exploratoria cuando no existe procedimiento formal; no garantiza
cobertura formal.

<a id="history-scientific-audit-contrato-estadístico"></a>

### Contrato estadístico

La fórmula sigue siendo la secuencia empirical-Bernstein plug-in predecible de
[Waudby-Smith y Ramdas, teorema 2 y ecuación 15](https://rss.org.uk/RSS/media/File-library/Events/Discussion%20meetings/Estimating-means-of-bounded-random-variables-by-betting.pdf).
Los pesos dependen del pasado; las diferencias pareadas están acotadas y se supone media
condicional común. El generador determinista de seeds no demuestra esa hipótesis científica.
Bonferroni de familia primaria y margen declarado no cambian. Monte Carlo comprueba regresiones,
no demuestra el teorema ni calibración de modelos reales.

<a id="history-scientific-audit-compatibilidad-y-propiedad"></a>

### Compatibilidad y propiedad

- Inferencia runtime `paired-pm-eb-cs-v2`, orden `committed-acquisition-prefix-v1` y `evidence_seeds`
  exactas. Registros históricos v1 siguen consultables: no se reescriben ni se llaman v2.
- `StudyDesign.replication_policy` histórico permanece inmutable; el registro de evaluación
  determina su versión runtime. No cambian IDs científicos, Runs ni Attempts.
- Inventario de bloques sigue en versión 1, que ya guarda ordinales y compromiso. Recuperación
  rechaza duplicados, discontinuidades, compromisos contradictorios o metadatos de seed erróneos.
  Coordenadas iniciales proceden de `seed_source.resolved`, no del número de seed. Ausencia de
  evidencia autoritativa falla antes de escribir. Previews siguen siendo de solo lectura.
- Snapshot de preguntas científicas versión 4; semántica de conclusiones de parámetro versión 3.
  `confidence` significa estabilidad descriptiva del evento mostrado, no probabilidad de verdad
  ni cobertura formal. Análisis históricos no se recalculan silenciosamente.
- Seeds finitas mantienen orden declarado; sweeps fijos no adquieren pruning adaptativo.
  Retry/continuación mantienen Run lógica y crean Attempts. Evidencia importada sigue inerte.

<a id="history-scientific-audit-verificación-reproducible"></a>

### Verificación reproducible

```bash
pytest -q tests/hpo tests/work tests/analysis tests/products tests/tui
python benchmarks/sequential_sweep_audit.py
python benchmarks/hpo_scientific_design.py
python benchmarks/resource_scheduling.py
ruff check .
mypy src/lambdaforge
```

El benchmark nuevo inspecciona cada prefijo de 200 trayectorias, hasta 96 bloques, con tres
comparaciones frente a referencia bajo nulos degenerado, de varianza baja y alta. El benchmark
de recursos es una regresión sintética existente: no valida GPUs reales ni una política nueva.
El benchmark científico existente solo cubre tres paisajes pequeños, no los ocho escenarios
requeridos para aceptación científica integral.

Resultados locales de este cambio:

- Suites amplias retenidas HPO/Work/Analysis/productos/Console: **718 aprobados, 628,84 s**, antes
  de las últimas adiciones de deuda comprometida/lector aislado; después, suites backend afectadas
  finales: **144 aprobados, 43,70 s**. No se afirma haber repetido pytest de todo el repositorio.
- Ruff, mypy (**581 ficheros fuente**) y comprobación de espacios aprobados.
- Wheel instalada fuera del checkout: smoke de imports/assets, CLI help/scaffold/validación,
  sweep automático CPU con procesos spawned y cuatro Runs, más preview de recuperación aprobado.
  Dependencias reutilizadas en prefijo temporal; no se reemplaza entorno de proyectos del usuario.
- Calibración: **0/200** violaciones de familia en cada uno de tres escenarios nulos, hasta 96
  inspecciones. Benchmark de tres paisajes conserva regret medio aproximadamente **3,70e-17**
  en ambas políticas. Benchmark sintético existente de recursos: makespan adaptativo **58 s**,
  fijo **109 s**, ambos sin OOM. Son regresiones del scheduler existente, no mejora nueva.
- Hardware CUDA, SSH/SLURM real y nuevos experimentos de recursos con trazas **no ejecutados**.

<a id="history-scientific-audit-fronteras-pendientes-explícitas"></a>

### Fronteras pendientes explícitas

| Clasificación | Pendiente | Aceptación necesaria |
| --- | --- | --- |
| CORRECTNESS_RISK | Fault injection integral de lifecycle/continuación y costes físicos adicionales en todos los lectores. | Checkpoints/publicación interrumpidos, resultados tardíos, fidelidades heterogéneas y continuación sin información por todas las rutas nativas. |
| DEFERRED_FEATURE | Leases CPU desiguales aprendidos y calibración ARI adicional. | Experimentos reales de demanda/progreso y replay de trazas que demuestren mejora manteniendo grants y afinidades. No se reemplaza el modelo actual especulativamente. |
| DEFERRED_FEATURE | Valoración científica revisada de preguntas correlacionadas y costes. | Comparaciones en ocho paisajes con soporte, calibración, coste y regret; se conservan autoridades actuales. |
| DEFERRED_FEATURE | Extracción más amplia de WorkRunner, sustitución de datasets por contrato y espera de productos. | Caracterización e integración de propiedad/identidad/recovery; política explícita de certificados, sin transitividad inferida. |
| DEFERRED_FEATURE | Recovery/adopción Fleet públicos, transporte reanudable y export distribuido. | Cadena completa lease/fencing/dispatch/acknowledgment/recovery/export validada. Se mantienen límites distribuidos actuales. |

Las suites retenidas cubren productos, certificados, import/export, replay y clasificación de
fallos; son cobertura de regresión, no aceptación de sus funcionalidades pendientes.

Las auditorías de recursos y fiabilidad solo redactadas en inglés se conservan en [el archivo inglés](IMPLEMENTATION.md#historical-audit-archive).

## Ampliación de algoritmos científicos — 2026-10-18, continuación por fases

**No se acepta como terminada la petición completa.** La entrega inicial verificó P0–P4;
la continuación de abajo verifica P5/P6. Estado actual: **P0–P6 implementadas, P7–P12 pendientes**.
El [registro técnico detallado](IMPLEMENTATION.md#scientific-algorithms-expansion) conserva la
auditoría, contratos, resultados reales y siguiente paso exacto; no hay que repetir lo ya hecho.
No se modificó WISDOM, ningún cluster, evidencia persistida, entorno de dependencias ni versión.

- P0: baseline limpio `main`/`0eb0310`. Ya existían FGW/StructuredMeasure, cinco clusterizadores,
  ARI/silhouette/estabilidad básicos y conformal de regresión. Precomputed estaba anunciado como
  capacidad pero no accesible mediante su string público; faltaban los nuevos componentes.
- P1: validación pública de disimilitudes, guardia antes de convertir/transferir, precomputed en
  DBSCAN/HDBSCAN/Agglomerative compatible/silhouette. Ward/KMeans rechazan. Simetrización solo
  explícita. La regresión detectó que HDBSCAN podía mutar D; ahora copia dentro del backend.
- P2: union-find por mesetas, jerarquía de eventos, índices de subárboles, ramas esenciales
  explícitas, consultas y curvas de masa. Oráculo exhaustivo, permutaciones, transformaciones
  monótonas, empates y cadena profunda. Los IDs empatados pueden renombrarse al reindexar vértices.
- P3: intervalos estables más anchos, pertenencia uniforme exacta y discreta, reglas explícitas
  de muerte/solapamiento, masa/núcleo/soporte y motivos de rechazo. IoU/Dice binarios/difusos.
  La selección disjunta es greedy, no un óptimo global; pertenencia no es confianza probabilística.
- P4: valores-p empíricos inclusivos con corrección add-one, máximos/mínimos por objeto,
  resolución/empates y bootstrap percentil por observación/grupo con estratos explícitos.
  No se confunde con conformal de regresión, posterior o control FDR de regiones positivas.

APIs NumPy diferidas en `topology`/`stats`, contratos/resultados individuales y sin dependencias
nuevas. sklearn 1.7.2, NumPy 2.2.6, SciPy 1.15.3, Torch 2.3.1, Python 3.10, pytest 9.1.1;
POT no está instalado. La instalación en un destino temporal fue consultada, no ejecutada.
Manual público único actualizado en [inglés](../MANUAL.md#guide-graph-science) y
[español](../MANUAL.es.md#guide-graph-science), changelog sin subir versión, dos ejemplos sintéticos.

Verificación focalizada final: **128 passed, 16 skipped** (todos por POT ausente), 2,47 s.
Incluye clustering anterior/nuevo, topología, regiones, calibración, imports diferidos, FGW y
enlaces del manual. Ruff pasa y mypy pasa para **21 ficheros**. Una ejecución más amplia anterior
dio 130 passed, 16 skipped y un fallo estructural previo por cuatro clases antiguas sin docstring;
se confirmó en HEAD y no se tocaron. No se presenta esa ejecución como verde.
Benchmark de cadenas: N=1.000, 10.000 y 50.000 en 0,0497, 0,4905 y 2,4800 s;
pico Python rastreado 0,75, 7,89 y 39,39 MiB, no RSS completo. Wheel construido localmente;
smoke desde `/tmp`, importando del propio wheel y compartiendo dependencias existentes en lectura,
pasó topología, regiones, calibración, bootstrap, validación y DBSCAN precomputed. No se instaló
ningún paquete ni se simuló una integración POT. Repetición focalizada: 128 passed, 16 skipped,
3,37 s. Los otros tres tests estructurales pasan por separado; el de docstrings previo se
deseleccionó explícitamente. `git diff --check` pasa.

Pendientes, por orden: **P5** FGW cuadrado/rectangular por bloques y reanudación nativa con
identidad/integridad; **P6 obligatorio** baricentros POT reales y representación relacional
sin anular diagonal; **P7–P8** medoides/asignación/rechazo; **P9–P10** mezclas y MDS;
**P11** estabilidad alineada/representantes/vecinos y bootstrap integrado; **P12** ejemplo
completo topología→FGW→prototipos, backends opcionales reales y aceptación integral.
El ejemplo actual de grafo/calibración no sustituye el ejemplo completo solicitado.

### Continuación P5/P6 — POT real, 2026-10-18

Se instalaron, con autorización explícita, **POT 0.9.7.post1 y 0.9.6** únicamente mediante
`--no-deps --target` en dos directorios temporales. El entorno original sigue sin POT. No se
tocaron WISDOM, clusters, trabajos científicos, versiones ni otras dependencias.

- P5: `FGWMatrixBuilder`, `FGWMatrixBlock`, `FGWMatrixResult`, identidad numérica versionada,
  colecciones cuadradas/rectangulares, bloques deterministas, reflexión triangular explícita,
  threads/budgets y diagnósticos acotados. `complete` no se confunde con convergencia.
  NaN/error parcial en memoria se conserva como null/error en JSON seguro, no como éxito.
- Dentro de un Work se reutiliza **Work.resume_map nativo** sin otro runner/cache/locks.
  Se conservan bloques acabados; ausentes, mal formados, incompletos, con checksum inválido o
  fallidos se recalculan. La unidad de recuperación es el bloque: si quedó incompleto pueden
  repetirse comparaciones buenas de ese bloque, nunca se devuelve su coste físico.
  Atributos opacos y métricas callable requieren codec/revisión explícitos; no se inventa
  equivalencia de closures. El presupuesto es una estimación, no RSS/callbacks arbitrarios.
- P6 obligatorio: `FGWBarycenter` y resultado inspeccionable llaman a **POT real**, con atributos
  numéricos compartidos, soporte/masas, C/X iniciales, seeds, reinicios secuenciales y selección
  por objetivo ponderado posterior válido. Alpha extremo se rechaza aquí; FGW ordinario lo mantiene.
- `StructuredMeasure(..., relational=True)` es un opt-in keyword-only: admite la diagonal no
  nula del baricentro **sin modificarla**. El contrato anterior sigue por defecto. Geometría/masas
  complejas ahora se rechazan explícitamente, sin pérdida silenciosa de su componente imaginaria.
- POT usa parada OR entre cambios de atributos/estructura. Se conserva su historial real pero
  no se vende esa parada como convergencia completa: se exigen BOTH y solves posteriores
  convergidos. Los acoplamientos/objetivo finales se recalculan en el prototipo final y se distinguen
  de los previos al último update. `backend_objective=None` si POT no lo aporta. No se promete
  óptimo global, soporte óptimo, unicidad ni realización física del prototipo sintético.

Verificación real: **171 passed sin skips**, 4,19 s con POT 0.9.7.post1; **66 passed**, 2,89 s
para transporte/recovery con POT 0.9.6; **3 passed** de ausencia de backend/imports sin target POT.
Incluye seis regresiones de WorkRunner/checkpoints nativos, comparación numérica con POT,
permutaciones, masas, diagonal relacional, multistart/arranque inválido/no convergencia, NaN y
consumo de matrices FGW por clustering precomputed. Ruff pasa y mypy pasa para 41 ficheros.
Ejemplo `examples/fgw_barycenter.py` ejecutado realmente, con diagonal `[0.16666667,0.33333333]`
conservada y objetivo local posterior `0.11291666666666667`. No es un benchmark universal.
La configuración CI incluye pruebas/ejemplo nuevos; no se lanzó ni observó GitHub CI.
No se afirma que pase el test global anterior de docstrings, ajeno a esta tarea.

Manual y changelog EN/ES actualizados sin nueva versión. El [registro detallado](IMPLEMENTATION.md#scientific-algorithms-fgw-follow-up)
mantiene comandos, límites y smoke de wheel tras ejecutarlo. Siguiente fase exacta: **P7 medoides**;
después P8 asignación/rechazo, P9 mezclas, P10 MDS, P11 estabilidad y P12 integración completa.
No se crearon stubs de esas funcionalidades ni se modificó la lógica nativa de HPO/recursos/recovery.

Wheel final reconstruido e importado directamente desde `/tmp`, comprobando procedencia `.whl`:
baricentro POT real con diagonal no nula, matriz por bloques completa e imports diferidos correctos.
Se compartieron dependencias existentes en lectura y POT temporal; no se instaló el wheel ni otras
dependencias. Tras la guardia adicional de underflow en pesos de soporte normalizados, repetición
final: **171 passed en 4,53 s** y **66 passed en 3,41 s**, respectivamente en las dos versiones POT.

## 2026-10-19 — compromisos de Jobs eliminados bloqueaban GPU V8

Diagnóstico remoto de lectura: V8a en el 12 tenía 16 Runs pendientes, ninguna activa y unos
79 GiB libres. V8b en el 16 tenía dos Runs en el token `0` y el token concedido `1` vacío con
unos 140 GiB libres. Ambos usan ya techos automáticos. El veto real era
`peer-startup-future-envelope-headroom`: compromisos de arranque de los Jobs antiguos eliminados,
con controladores muertos y sin procesos marcados ni propiedad relevante incierta.

Se corrige la reconciliación operativa con `ProcessSupervisor.detached_owner_absent`: solo si
desapareció todo el directorio del Job, exige muerte/reutilización y ausencia nativa de procesos.
No basta antigüedad, archivos ausentes o GPU vacía; orfandad viva, propiedad incierta y directorios
existentes corruptos siguen protegidos. Se reutiliza la comprobación por propietario/ruta dentro
del sondeo, sin recrear registros ni cambiar recovery, ciencia, reservas o límites físicos.

Verificación local: **118 passed** en planificación GPU, startup compartido, ámbito de procesos
de recovery y scheduler directo. Ruff y mypy sobre los módulos operativos modificados. No se
modificó WISDOM ni ningún Job/evidencia/reserva remotos. Los runtimes inmutables actuales todavía
necesitan cancelación/recovery nativos autorizados para adoptar el arreglo; no se realizaron.

## 2026-10-20 — componentes genéricos para el diseño prospectivo V9

Lectura del capítulo V9, sus familias incluidas y auditoría en TFM/Memoria; sin modificar el informe
ni WISDOM. Se reutilizan GCN, GraphSAGE, GIN, R-GCN, PNA, GATv2, Transformer disperso, EGNN y ECMP.
Nuevos módulos públicos individuales: GINE, NNConv completa/por bases, SchNet radial, GVP/GVP-GNN,
PaiNN, interacciones DimeNet bilineales/DimeNet++ factorizadas con bases Bessel/armónicas y encoder
direccional por nodo, y Transformer estructural global inspirado en Graphormer. Se añaden primitivas
de geometría, bases, dropout vectorial, tripletas y caminos dirigidos, con límites explícitos.

Se declaran las adaptaciones (SiLU/readout/init de Torch, atributos genéricos y grado/caminos por
bloque), no reproducción de todos los modelos/pretraining originales. Gates, ramas relacionales,
protocolos, selección, controles de capacidad y ciencia de WISDOM siguen en el consumidor. El extra
opcional `graph-directional` usa SciPy solo para constantes al construir bases; sin auto-instalar,
PyG ni SymPy. Forward es Torch, con O(3) para vectores polares y lectura escalar invariante, sin
inventar vecinos ni alterar posiciones físicas. La atención densa nunca cruza elementos del batch.

Validación: **168 pruebas** nuevas/de regresión de grafos/catálogo y **6** de modelos generales.
Ecuaciones, gradientes, bf16 CPU, permutación, reflexiones/rotaciones, multiplicidades, vacíos,
state_dict, entradas inválidas, límites y dependencias. Bases contrastadas con SciPy hasta ocho
órdenes y 64 raíces distintas ordenadas por orden. Ruff pasa; mypy pasa para **60 ficheros**.
Ejemplo `graph_encoders.py --directional` ejecutado con forward/backward finitos en ocho variantes.
Wheel 0.19.0 construido sin aislamiento/instalaciones y probado desde `/tmp`, verificando que los
imports y el ejemplo proceden realmente del wheel y que incluye el extra. No se afirma instalación
en un entorno limpio, conformance/rendimiento CUDA, mejora predictiva ni ejecución de V9.
Manual único y changelog EN/ES actualizados, sin cambio de versión. P7–P12 del trabajo anterior
de algoritmos científicos siguen pendientes y no se presentan como completadas por esta tarea.

## 2026-10-20 — interfaces públicas de composición/control de grafos

Completadas solo las fronteras que faltaban: moduladores firmados GCN/GraphSAGE sin recalcular
grados; etapas públicas GIN/GINE/ECMP/PNA con una actualización común; control conjunto de bases
radiales/cutoff y entrada pública de bases/soportes SchNet/PaiNN; control por etapas de aristas
y tripletas mixtas DimeNet e inyección de bases nativas por relación. La atención global existente
no se duplica; su guardia contempla promoción de dtype de scores bajo autocast. Matriz de
capacidades, firmas y ejemplos en [el manual](../MANUAL.es.md#graph-operator-controls).

Constructores, ecuaciones por defecto y claves de estado conservados. Se distingue raíz, atributos,
relación, distancia y dirección/ángulos; controles fraccionarios se componen por etapas. Permanecen
constantes propias documentadas al cerrar mensajes. Instancias independientes conservan parámetros
de etapas no ejecutadas; no se cuentan como capacidad ejecutada. PNA conserva dtype topológico de
su forward bajo AMP mediante el argumento público degree_dtype. ECMP rechaza ahora índices
flotantes/Booleanos e IDs de relación inválidos en lugar de truncar. No se inventa equivalencia
de encodings externos ni se infieren columnas de distancia duplicadas en atributos del consumidor.

Validación real: **274 passed en 3,67 s** en diez baterías de grafos/catálogo. Incluye referencias
previas exactas, pesos con signo/cero/fraccionarios, raíz única, cierre de fuentes, duplicados,
tripletas mixtas, soportes largos, permutaciones/reversión, batch disjunto, O(3), gradientes bf16,
state_dict estricto, geometría inválida y memoria. Ruff lint/formato pasan; mypy pasa para **59
ficheros**. Ejemplo `graph_operator_controls.py --directional` ejecutado con métodos públicos y
gradientes finitos; wheel construido sin instalar dependencias y ejemplo empaquetado ejecutado
desde `/tmp`, verificando procedencia real del wheel. No se alteró WISDOM/memoria, contactaron
clusters, lanzaron Jobs, instalaron dependencias ni hicieron commits o cambios de versión.
No se acreditan rendimiento CUDA, ajuste de proteínas reales, mejora predictiva ni toda la receta
Graphormer original; una prohibición de matrices densas debe resolverla expresamente el consumidor.
Las fases pendientes anteriores de algoritmos científicos siguen independientes de esta tarea.

### 2026-10-20 — Acotar realimentación del posterior ARI y cálculo de capacidad conjunta

Inspección de lectura de `wisdom-v8d`, Job `job-20261020101604-883baff7`, cluster16:
permisos físicos `1,3`; token 1 con unos 139 GiB libres y un worker CUDA ocioso de 842 MiB,
token 3 con tres entrenos activos. La primera Run del token 1 ya persistió `succeeded`, pero
el controlador no la recogió. Proyecciones Study/admisión obsoletas pese a heartbeat supervisor
y entrenos hermanos avanzando. Soportes activos de 3.446/3.447 puntos, predicciones de 6.739;
proyección de recursos de unos 30 MiB y summary de unos 106 MiB. Pila leída con depurador nativo:
`_normalized_weights → _weighted_quantile → _joint_fit_probability →
_resource_experiment_signature → _exploration_value → place → _execute_gpu_admitted_runs`.
Depurador desconectado; ningún Job cancelado, reiniciado, reintentado ni parcheado remotamente.

Corrección nativa: mezclas y entradas de convolución acotadas a 128 puntos ponderados; masas
extremas conservadas y agrupación interior conservadora hacia arriba. Ensemble compartido recorre
eventos CDF una vez, sin ordenar cada soporte por cuantil. Actualizaciones activas conservan pesos
residuales históricos, sin convertir colas raras en equiprobables. Soportes pequeños exactos,
controles físicos, permisos, dominancia OOM y políticas científicas intactos. Sin scheduler paralelo
ni override de concurrencia.

Verificación: **175 pruebas pasan en 3,90 s** sobre liveness de distribuciones, recursos adaptativos,
invariantes v3.1, benchmark/replay, refill/recovery nativos y compromisos/observación/salud/atribución
GPU. Ocho casos nuevos verifican CDF conservadora, masas extremas, ensemble exacto por enumeración,
entradas antiguas grandes, 70 ciclos activos sin evidencia terminal y pesos de cola. Ruff lint/formato
pasa para tres ficheros Python; mypy para ambas interfaces fuente modificadas. Benchmark sintético
local de cuatro distribuciones de 6.740 puntos: cálculo anterior no termina en 3 s; corregido unos
0,042 s. No demuestra throughput real en GPU.

Limitación operativa: el runtime remoto inmutable ya ejecutándose sigue conteniendo la versión
anterior. Incorporar la corrección a ese Job exige una operación de ciclo de vida autorizada aparte;
no se realizó. Sin cambios en evidencia viva, código de WISDOM ni dependencias.

### 2026-10-27 — Replicación Dataset como Job operativo persistente

CLI `datasets replicate --apply` y Replicate en consola encolan el mismo Job nativo de tipo
`dataset-replication`. Reutilizan JobService/JobStore y schedulers/supervisores existentes, sin
Work científico, ciclo de vida alternativo ni otro motor de copia. SSH directo se ejecuta en origen
sin depender del PC. Relay comprimido usa un Job local independiente de la interfaz/lanzador, pero
necesita su PC/red. Snapshots conservan referencias de credenciales, nunca contraseñas; se rechaza
relay que requiera prompt posterior. SSH mantiene clave de host y no reenvía credenciales/agente.
No se lanzaron/reanudaron/cancelaron transferencias reales ni se instalaron runtimes remotos.

Dataset Transfers y `datasets transfers [DATASET]` / `transfer-status JOB` recuperan metadatos
acotados y logs nativos. Solicitudes activas exactas bloqueadas/deduplicadas; propiedad incierta o
falta de confirmación nunca autorizan duplicar. Se mantienen políticas CPU/almacenamiento/GPU.
Acknowledgements verificados reconcilian la colocación del controlador sin cambiar identidad;
fallos/corrupción no son copias completas. Origen/destino quedan protegidos frente a borrado incluso
si el Job está en el PC. Los perfiles derivados no reciben dos veces el scope de proyecto. Historial
terminal usa merge nativo de metadatos en caché; status explícito consulta al host. UI conserva filas
previas ante observación fallida, sin spam; Locations incorpora colocaciones completadas.

Verificación: **142 pruebas pasan en 73,72 s**: todos los tests Dataset, ProcessScheduler nativo y
consola Dataset. Jobs locales temporales reales, salida del lanzador antes de empezar, deduplicación
concurrente, compresión directa/relay, scope de proyecto, corrupción, metadatos sin assets, caché
terminal, rutas/propiedad, protección de borrado y feedback pendiente/verificado. Ruff lint/formato
pasa para 12 ficheros; mypy para 9 interfaces fuente. Wheel construido sin instalar dependencias;
módulos y parsers CLI importados desde ese wheel en `/tmp`, comprobando procedencia.
No se acredita instalación limpia, conectividad SSH/SLURM real ni rendimiento con datasets grandes.

Límites: no convierte retroactivamente transferencias antiguas; relay no sobrevive al apagado del PC;
copias interrumpidas reinician el stream, sin continuación por offset. Verificar/reconciliar un
destino ya exacto mantiene la vía síncrona existente sin transferencia nueva. La persistencia de
una nueva copia empieza al confirmar el Job nativo, no solo al pulsar Submit. Sin cambio de versión.

### 2026-10-27 — Rutas relativas de solicitudes relay y diagnóstico de fallos tempranos

Inspección exclusivamente de lectura de `stderr.log`, `request.json` y `work/replication.json`
aportados para `job-20261027090952-65861abc`: el archivo existía, pero su argumento era relativo.
El supervisor usó correctamente su work_dir absoluto como cwd; el worker buscó la ruta relativa al
proyecto debajo de ese directorio y terminó antes de copiar. No es un fallo científico/de integridad.
No se modificaron WISDOM, datasets, registros ni el Job real; no se lanzaron copias ni comandos remotos.

La replicación reutiliza el anclado local nativo antes de construir rutas/preparar/serializar el
comando. Workspace y raíces locales state/cache/run/dataset/lease e intérpretes con ruta explícita
quedan fijados a ProjectContext; el registro ya era absoluto. Snapshots conservan esas rutas. Perfiles
SSH intactos y con semántica del host: sin resolverlos en el PC, symlinks, copias auxiliares ni cambiar
el cwd del worker como parche.

Status lee como máximo 8 KiB del stderr propio en Jobs fallidos; redacta antes de exponer/guardar y
devuelve causa útil y `logs_command` incluso sin progreso/resultado. Sin log hay diagnóstico explícito,
no error vacío. Cachés antiguas fallidas se actualizan una vez. CLI muestra la causa y consola la
incluye en detalles del log. Resultado, identidad, publicación, deduplicación activa y propiedad
incierta siguen bajo las mismas garantías.

Verificación: **146 pruebas pasan en 81,62 s**: todos los tests Dataset, ProcessScheduler y consola
Dataset. Regresiones con worker real desde work_dir y raíces locales relativas/absolutas (controlador
en subdirectorio del proyecto), snapshots absolutos completos, SSH intacto, solicitud ausente sin
destino/progreso/resultado, redacción, caché antigua vacía y CLI JSON/humano. Pasan pruebas existentes
de deduplicación activa y publicación atómica/verificada. Tras acotar el tail se repitieron casos
focalizados de fallo temprano. Ruff lint/formato pasa para cuatro ficheros; mypy para tres interfaces
fuente. Sin dependencias nuevas ni cambio de versión. Reintento por `datasets replicate`, primero
preview y después apply; Job fallido y logs se conservan como historia.

### 2026-10-27 — Telemetría de recursos con JSON inmutable anidado y revisión de recuperación

Reproducido localmente el traceback generator/items aportado de V10a, sin CUDA ni datasets.
asdict reconstruye subclases dict con un generador de pares; FrozenJsonMapping solo aceptaba Mapping.
El constructor admite ahora el protocolo dict de iterables de pares y sigue congelando los valores
anidados. Las dataclasses de recursos usan un único límite campos/contenedores portables, sin
reconstrucción asdict/deepcopy: historial activo/terminal, admisión, exploración, ledgers y contratos
de valor científico de las trazas. Listas/mappings inmutables y mapping proxies se materializan en
copias JSON nativas. Valores de wire equivalentes y hashes de identidad de parámetros intactos.
Valores no JSON siguen fallando explícitamente y conservan el snapshot atómico anterior. Sin catch
genérico, desactivar monitorización/admisión ni introducir otro motor de recovery.

Verificación: **213 pruebas pasan en 72,93 s**: JSON inmutable/serialización de recursos (12 nuevas),
ARI/v3.1/liveness/replay/benchmark, IPC inmutable del monitor, compromisos GPU compartidos y recovery
adaptativo/fijo/de permisos GPU. persist_active real con paso 89/fase de evaluación sintéticos,
round-trip, valores/inmutabilidad, JSON válido, otros caminos de recursos, wire y rechazo atómico.
Recovery conserva paths/bytes de resultados válidos, compatibilidad de checkpoints, identidad
Execution/Run y costes/historia; reencola seeds interrumpidas sin inferir finalización de telemetría.
No hizo falta cambiar recovery. Pasa además una regresión de identidad/caché del entorno nativo.
Ruff lint/formato para cuatro ficheros, mypy para dos interfaces fuente. Wheel construido sin instalar
dependencias; desde `/tmp`, procedencia del wheel, asdict anidado y persist_active/recarga verificados.

Despliegue pendiente como acción explícita: usar el paquete LF corregido en controlador, bootstrap
gestionado desde el proyecto consumidor, retry --dry-run y reintento deliberado. EnvironmentIdentity
incluye bytes del wheel aunque siga siendo 0.19.0. Sin parchear prefijos antiguos ni reescribir
evidencia. Solo se leyó el YAML V10a para confirmar nombre/seeds; no se ejecutaron Runs, transferencias,
sondeos remotos, instalaciones, cambios de datos/índices ni recuperación real. No se descargó el JSON
de error remoto ni se afirman cantidades de Runs completadas/checkpoints compatibles de V10a.
No se atribuye causalidad a la transferencia Dataset simultánea. Sin cambio de versión.

### 2026-10-27 — Replicación directa en ambas direcciones, relay explícito y estado verificable

La inspección de solo lectura del Job aportado encontró EOF escribiendo al receptor y otro EOF
Paramiko durante cleanup. Demuestra un stream roto, no distingue red, salida del receptor o
suspensión del PC. Los eventos nativos registraron failed a las 11:21:33 UTC, unos cinco segundos
después de la salida 11:21:28; 11:39 fue una confirmación posterior, no la primera detección.
No está demostrada la causa exacta del desfase visual anterior. El listado de Datasets actualiza
ahora la observación nativa sin exigir transfer-status ni abrir el detalle.

El mismo servicio comprueba envío SSH y después descarga inversa mediante sondas efectivas,
acotadas y de solo lectura. Distingue autenticación, host key, timeout, red, herramientas, permisos
y causa indeterminada. Envío vive en origen y descarga en destino, usando el motor exacto existente
sin socket de datos del controlador. Auto solo propone relay: consentimiento interactivo CLI,
--route relay explícito en automatización y confirmación en consola muestran ruta, tamaño original
y dependencia del PC. Se persisten dirección/diagnóstico aprobados; un cambio previo al despacho
obliga a revisar. Sin relajar confianza, copiar claves, reenviar agentes ni otro runner/reintento infinito.

Estado usa OperationalHealth/Jobs nativos, separando resultado del proceso, antigüedad de observación,
avance de bytes y verificación/publicación observada. Observación desconocida no inventa failed
ni running confirmado. No calcula porcentajes comprimidos con tamaño original. Drena stderr acotado
y redactado; exit desconocido sigue siendo desconocido. Cleanup conserva la excepción original y
recoge los subprocesos propios. Respuestas acotadas del receptor distinguen verificación/publicación
reales en envío directo y descarga. Conserva checksums, publicación atómica, propiedad y deduplicación.

Pruebas exclusivamente con fixtures locales desechables/transportes simulados: workers persistentes
en ambas direcciones sin stream controlador, consentimiento/ruta, diagnóstico, estado incierto y
terminal, relojes separados, fallo temprano emisor/receptor, EOF original/cleanup, rechazo de
publicación incompleta y protección de copias activas. Sin sondas SSH reales, transferencias,
cancelación/reinicio, instalaciones ni cambios en WISDOM/datasets/índices. Autenticación/confianza
del centro y permisos/runtime siguen correspondiendo a operador/administrador. Las sondas no
garantizan conectividad futura. Reintento deliberado reinicia streams parciales, no continúa por
offset; reutiliza destinos exactos verificados. Sigue en 0.19.0; no se afirma conectividad real entre clusters.

Comprobaciones: suite amplia de Datasets/ProcessScheduler/política SSH/binario/consola, **179 tests**
en 108,07 s. Tras añadir fases de publicación observadas en receptor, subconjunto afectado de
replicación/binario/consola: **77 tests** en 72,17 s (dos nuevas regresiones dirección/fases).
Ruff lint/formato pasa en 16 ficheros y mypy en 12 interfaces fuente. Wheel construido sin instalar
dependencias; smoke desde `/tmp` comprueba procedencia extraída, helper y clasificación de rutas.
Son pruebas con fixtures/empaquetado, no aceptación del acceso real entre centros.

### 2026-10-27 — Política explícita sin portátil en replicación desde consola

La consulta local de solo lectura del Job `job-20261027120734-c733ed69` confirma que `host=local`
era correcto: usaba relay controller-stream. Su diagnóstico persistido indica clave ED25519 de
destino no conocida en envío directo y autenticación denegada en descarga inversa. Son resultados
de esas sondas, no garantía de conectividad actual. El preview nuevo desde esta sesión no pudo
autenticar por falta de backend de keyring disponible; no se modificaron claves, credenciales,
perfiles, Jobs o datasets reales ni se inició transferencia alguna.

La consola solo dejaba elegir extremos y convertía la propuesta relay en política relay al confirmar.
Ahora ofrece Direct only, Auto y Relay, predetermina Direct only para origen remoto y envía esa
política al preview nativo. Un fallo directo abre diagnóstico legible/desplazable, sin confirmación
relay implícita. Extremos locales explican dependencia del PC. También se rechaza defensivamente
un plan relay bajo política directa. Transfers identifica relay como `via this PC`; los Jobs ya
enviados conservan su ruta fija. Manuales EN/ES aclaran que el host deriva de la ruta, no es una etiqueta editable.

Pruebas: consola de replicación más regresiones nativas de rutas, **33 passed** en 43,99 s; suite
final ampliada de consola, **10 passed** en 17,97 s. Cubre selección/default, controles accesibles a
80×24, diagnóstico literal legible, rechazo de relay bajo política directa y etiqueta relay clara.
Ruff lint/formato pasa en tres ficheros y mypy en dos interfaces fuente. Sigue siendo necesaria
configuración autorizada de SSH entre hosts: operador/administrador verifica claves y autoriza acceso
no interactivo en alguna dirección. No se desactiva confianza ni se reenvían credenciales.

### 2026-10-28 — Aprobación explícita de huellas SSH de primer acceso

La captura identifica falta de confianza ED25519 en 16 → IP exacta del destino; autenticación
denegada inversa es otro problema. Aceptar claves en el controlador no configura el iniciador;
alias e IP pueden tener registros distintos. Se intentó un preview WISDOM de solo lectura, pero
esta sesión no accede al backend de keyring: no se afirma el estado SSH/autenticación actual.
Sin cambios reales de confianza, credenciales, Jobs, datasets, registros o WISDOM ni copias iniciadas.

El error nativo de ruta directa expone diagnóstico estructurado sin analizar tracebacks humanos.
SiteSshTrustService/helper distribuido inspeccionan de forma acotada configuración SSH efectiva,
confianza de usuario/sistema y claves ofrecidas. DatasetService, `datasets trust-ssh` y consola usan
la misma operación. Una modal de primer acceso muestra iniciador, destino/IP/puerto, archivo remoto
y huellas SHA256 y avisa que keyscan no las autentica. Aprobar fija/reconsulta todas las huellas,
revalida bajo lock nativo, conserva bytes/permisos previos y añade entradas hasheadas atómicamente
al known_hosts efectivo y seguro del usuario iniciador. Cambios concurrentes, huellas distintas,
claves previas diferentes/marcadas, symlinks y rutas inseguras se rechazan. HostKeyAlias, proxies y
expansiones especiales requieren gestión del centro, no aproximaciones. Sin desactivar comprobación
estricta, reenviar credenciales ni reemplazar claves automáticamente.

Tras aprobar repite el preview normal; la copia necesita su propia confirmación. Cancelar confianza
no cambia nada. Confiar en el servidor no concede acceso no interactivo ni resuelve una cuenta que
solo acepte contraseña. Se mantienen Jobs directos envío/descarga, consentimiento relay, identidad,
checksums y publicación atómica del dataset.

Verificación: **194 passed en 100,32 s** en tests de Dataset, confianza, binario/política SSH y consola.
Cubre huellas contra ssh-keygen, preview sin escritura, aprobación, lookup hasheado, conservación,
idempotencia/concurrencia, rotación/revocación/rutas inseguras, puertos, CLI y helper subprocess real
con binarios SSH simulados/HOME desechable; modal cancelar/aprobar/re-preview y confirmación de copia
separada. Ruff lint/formato en 11 ficheros y mypy en nueve interfaces fuente. Ayuda CLI comprobada,
wheel sin instalar dependencias y smoke de procedencia/helper/APIs desde wheel extraído. Se consultó
el manual primario OpenSSH sobre keyscan no autenticado. Sin aceptación real del centro ni nueva versión.

### 2026-10-28 — Diagnóstico de arranque CPU V10d y atribución GPU cero

Inspección de solo lectura del Job `job-20261028082934-ca20dad0` en ctgpgpu16: permisos GPU 2/3
intactos, tres workers científicos vivos y cuarta seed esperando. Cada worker consumía ~un núcleo,
leía decenas de GB y seguía en carga de splits, sin procesos CUDA ni epochs. El código del runtime
consumidor inmutable confirma que registrar entradas valida sidecars y AtomicSASA.validate vuelve
a calcular SASA durante inicialización. Es preparación CPU del consumidor, no prueba de deadlock
del launcher GPU. Sin intervención, los primeros llegaron a epochs 3–4 y LF admitió seed 229 a
09:17:10 UTC. Sin cambios en Work/YAML, datasets, evidencia persistida, permisos GPU o procesos.
No se usó attach intrusivo de stack: pausaría un worker vivo y requiere aprobación separada del
operador. Código, logs, metadatos de archivos abiertos y avance CPU/E/S bastaron para el diagnóstico.

Se reprodujo otro bug LF con fixtures: workers CPU ausentes de la tabla CUDA caían a atribución
agregada y heredaban uso global del driver/ajeno (~617 MiB en este caso). GPUObservation declara
ahora completitud del inventario, incluidos errores NVML/SMI al consultar/inspeccionar y truncación.
El runner reconoce ausencia como cero medido solo con inventario completo/medido, identidad viva
validada, observación posterior al nacimiento y sin allocator positivo contradictorio. Bytes por
PID desconocidos, observación parcial/antigua y PID reutilizado siguen siendo desconocidos/prudentes;
uso multidispositivo incierto no se etiqueta exacto. No se relajó VRAM física, reserva futura,
vía protegida, exploración, permisos de dispositivos ni resultados científicos.

Cambios acotados a GPUObservation, atribución del runner y regresión focalizada, más documentación/
changelog EN/ES. **153 tests pasan en 2,59 s**: ARI/v3.1 nativo, distribución/replay/benchmark,
atribución/observación y admisión compartida. Incluye live-update real, reserva/memoria ajena,
desconocido vs cero, nacimiento/PID reutilizado, bytes inciertos, conflicto allocator y fallos del
backend. Ruff lint/formato en tres ficheros; mypy en dos interfaces fuente. Sin instalar dependencias
ni subir versión. Jobs inmutables ya ejecutándose no se parchean en caliente: la corrección requiere
un próximo runtime desplegado explícitamente. Reducir el arranque SASA corresponde al consumidor;
no se afirma optimización WISDOM ni despliegue del cambio sobre el Job actual.
