# Reforma arquitectónica: auditoría y registro de implementación

[English](ARCHITECTURAL_REFORM.md)

Este documento permite continuar el trabajo; **no declara terminada la reforma solicitada**.
Base: `d2f526d`, LambdaForge 0.17.0. La reforma no modifica WISDOM, trabajos de clusters productivos
ni registros científicos persistidos.

Continuación sobre `c8f2d2e`: la [auditoría científica](SCIENTIFIC_AUDIT.es.md) corrige orden de
evidencia pareada, recovery del compromiso anticipado y coherencia conclusión/estabilidad. No cierra
los pendientes de lifecycle, asignación host aprendida, dependencias de productos o Fleet distribuido.

## Hallazgos

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

## Base implementada

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

## Integración adicional y correcciones de higiene

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

## Capa de modelo, registro y transporte de productos

`products/models.py` separa significado versionado explícito, contenido exacto y provenance
original con modelos profundamente inmutables y pickle-safe. `products/registry.py` publica copias
independientes verificadas con SHA-256 crudo, nombres/contratos inmutables, locks nativos y lecturas
de metadata acotada. Lecturas/previews no crean nada. Configuración del productor es provenance,
no requisito de compatibilidad; nuevos orígenes no sustituyen al original.

`lf products list/show/provenance/consumers/verify/publish/export/import/select/decide/status/finalize` llama a esa API. Los bundles verifican
manifiestos, bytes e inventario completo de provenance antes de import idempotente. Export → borrar
original → import concurrente conserva bytes promovidos e identidad histórica. Véase
[Productos](PRODUCTS.es.md). `products/selection.py` selecciona snapshots de modelos completos, de
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
[Reconstrucción de datasets](DATASET_RECONSTRUCTION.es.md). Falta resolución YAML por contrato.

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

## Pendiente, por orden de dependencia

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

## Punto de validación

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
