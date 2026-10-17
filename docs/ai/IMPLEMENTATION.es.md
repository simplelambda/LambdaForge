# Estado técnico de la integración nativa de agentes

[Registro detallado y matriz de requisitos](IMPLEMENTATION.md) ·
[Guía de uso](../MANUAL.es.md#guide-integration).

La reforma integral **no está terminada**. LambdaForge 0.19.0 empaqueta los fundamentos
implementados y las correcciones operativas, no una promesa de capacidades inexistentes.
El punto de partida y las pruebas históricas de 0.18.0 se conservan como tales; la nueva versión
no cambia los requisitos pendientes ni habilita mutaciones científicas no soportadas.
No se ha modificado WISDOM, iniciado trabajos en clusters ni cambiado evidencia científica.

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
- Existen leases de almacenamiento y limpieza con preview. Falta admisión completa del
  provisioning y separar dependencias del código sin debilitar entornos inmutables/provenance.

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
| 25–30, 60 | Filesystem, storage, entornos, caché | Previews/raíces diferidas y borrado terminal con locks/padres Work vacíos de ResultStore corregidos; faltan resto de auditoría read-only/leases/caminos de limpieza, admisión de provisioning, capas de código/dependencias y economía de limpieza. |
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
