# Manual de LambdaForge 0.19

Este es el único manual público del framework; las guías detalladas son capítulos. En `ai/` solo quedan contratos internos y el registro histórico de aceptación.

[English](MANUAL.md) · [Contratos internos](ai/FRAMEWORK_CONTRACTS.md)

## Índice

- [Authoring científico: contratos, muestras y ETL](#guide-scientific-authoring)
  - [Tablas incrementales, SQL y efectos de preview](#guide-tabular-connectors)
- [1. Modelo mental](#1-modelo-mental)
- [2. Instalación y estructura](#2-instalación-y-estructura)
- [3. API de Work](#3-api-de-work)
- [4. Ficheros gestionados, cache, map, outputs y herramientas](#4-ficheros-gestionados-cache-map-outputs-y-herramientas)
  - [4.1 Elegir almacenamiento](#41-elegir-almacenamiento)
  - [4.2 ManagedFile y cache](#42-managedfile-y-cache)
  - [Validación científica compartida de datasets](#guide-dataset-validation-cache)
  - [4.3 Map simple y reanudable](#43-map-simple-y-reanudable)
  - [4.4 Outputs y checkpoints](#44-outputs-y-checkpoints)
  - [4.5 Herramientas externas](#45-herramientas-externas)
- [5. Referencia YAML](#5-referencia-yaml)
- [6. Ficheros y datasets](#6-ficheros-y-datasets)
  - [6.1 Reutilizar, reconstruir y recuperar publicación](#61-reutilizar-reconstruir-y-recuperar-publicación)
  - [6.2 Réplica nativa comprimida](#62-réplica-nativa-comprimida)
  - [6.3 Gestión explícita de conflictos](#63-gestión-explícita-de-conflictos)
- [7. Secuencia, paralelismo, seeds y búsqueda](#7-secuencia-paralelismo-seeds-y-búsqueda)
  - [7.1 Defaults por intención y autoridad resuelta](#71-defaults-por-intención-y-autoridad-resuelta)
  - [Preguntas científicas, confianza y regiones prácticas](#preguntas-científicas-confianza-y-regiones-prácticas)
  - [7.1 Sweeps exactos sin HPO](#71-sweeps-exactos-sin-hpo)
  - [7.2 Entrenos independientes por GPU](#72-entrenos-independientes-por-gpu)
  - [7.3 Continuación multi-fidelidad](#73-continuación-multi-fidelidad)
  - [7.4 Contrato de early stopping](#74-contrato-de-early-stopping)
  - [7.5 Observabilidad viva del estudio](#75-observabilidad-viva-del-estudio)
- [8. Ejecución, identidad y reutilización](#8-ejecución-identidad-y-reutilización)
  - [Recuperar un Study interrumpido](#recuperar-un-study-interrumpido)
- [9. Resultados y metadata](#9-resultados-y-metadata)
- [10. Clústeres y Jobs](#10-clústeres-y-jobs)
  - [Colas de ejecución secuenciales](#guide-execution-queues)
  - [Paquetes nativos declarados por el proyecto](#paquetes-nativos-declarados-por-el-proyecto)
- [11. Limpieza y seguridad](#11-limpieza-y-seguridad)
- [12. Referencia CLI](#12-referencia-cli)
- [13. Clustering](#13-clustering)
  - [Disimilitudes precalculadas](#disimilitudes-precalculadas)
- [14. Componentes neuronales reutilizables](#14-componentes-neuronales-reutilizables)
- [15. Arquitectura y extensiones](#15-arquitectura-y-extensiones)
- [16. Análisis de estudios](#16-análisis-de-estudios)
  - [16.1 Semántica de evidencia](#161-semántica-de-evidencia)
  - [16.2 Efectos, cobertura y fiabilidad](#162-efectos-cobertura-y-fiabilidad)
  - [16.3 Evidencia de admisión](#163-evidencia-de-admisión)
  - [16.4 Espacio de investigación y declaraciones](#164-espacio-de-investigación-y-declaraciones)
- [17. Consola de investigación](#17-consola-de-investigación)
- [18. Aislamiento por proyecto](#18-aislamiento-por-proyecto)
- [19. Fiabilidad operativa](#19-fiabilidad-operativa)
- [20. Distancia Fused Gromov–Wasserstein (FGW)](#20-distancia-fused-gromovwasserstein-fgw)
  - [Entradas y ejemplo](#entradas-y-ejemplo)
  - [Atributos estructurados y distancias precalculadas](#atributos-estructurados-y-distancias-precalculadas)
  - [Objetivo y diagnóstico](#objetivo-y-diagnóstico)
  - [Límites y comparación de varios objetos](#límites-y-comparación-de-varios-objetos)
- [21. Resultados de Work, evidencia histórica y productos independientes](#guide-work-results)
  - [Inspección e informes](#guide-work-results-inspección-e-informes)
  - [Seleccionar dependencias históricas exactas](#guide-work-results-seleccionar-dependencias-históricas-exactas)
  - [Promoción explícita desde un Work ordinario](#guide-work-results-promoción-explícita-desde-un-work-ordinario)
  - [Portabilidad y disponibilidad remota](#guide-work-results-portabilidad-y-disponibilidad-remota)
  - [Evidencia en ampliaciones de Studies fijos](#guide-work-results-evidencia-en-ampliaciones-de-studies-fijos)
- [22. Productos científicos durables: base implementada](#guide-products)
  - [Identidades y contratos](#guide-products-identidades-y-contratos)
  - [API de publicación explícita](#guide-products-api-de-publicación-explícita)
  - [CLI nativa](#guide-products-cli-nativa)
  - [Selección post-Study de ModelSet](#guide-products-selección-post-study-de-modelset)
  - [StudyDecision nativa](#guide-products-studydecision-nativa)
  - [Publicación declarada y retry de publicación sin entrenar](#guide-products-publicación-declarada-y-retry-de-publicación-sin-entrenar)
  - [Dependencias Work tipadas](#guide-products-dependencias-work-tipadas)
  - [Export/import portable de productos](#guide-products-exportimport-portable-de-productos)
  - [Importar un Study portable sin ejecutarlo](#guide-products-importar-un-study-portable-sin-ejecutarlo)
- [23. Estudios científicos condicionales](#guide-conditional-studies)
  - [1. Una pregunta, un Study](#guide-conditional-studies-1-una-pregunta-un-study)
  - [2. Gramática de activación](#guide-conditional-studies-2-gramática-de-activación)
  - [3. Validar y entender el plan](#guide-conditional-studies-3-validar-y-entender-el-plan)
  - [4. Tiempo y capacidad del destino](#guide-conditional-studies-4-tiempo-y-capacidad-del-destino)
  - [5. Identidad y evidencia existente](#guide-conditional-studies-5-identidad-y-evidencia-existente)
- [24. Ampliaciones incrementales de Studies fijos](#guide-study-extensions)
  - [CLI](#guide-study-extensions-cli)
  - [Ejemplo de regularización](#guide-study-extensions-ejemplo-de-regularización)
  - [Ampliaciones condicionales y selectivas](#guide-study-extensions-ampliaciones-condicionales-y-selectivas)
  - [Identidad, compatibilidad y evidencia](#guide-study-extensions-identidad-compatibilidad-y-evidencia)
  - [Ejecución admitida](#guide-study-extensions-ejecución-admitida)
  - [Consola y análisis acumulado](#guide-study-extensions-consola-y-análisis-acumulado)
  - [Presupuesto, fallos y durabilidad](#guide-study-extensions-presupuesto-fallos-y-durabilidad)
  - [Verificación](#guide-study-extensions-verificación)
- [25. Datasets publicados y reconstrucción científica](#guide-dataset-reconstruction)
  - [Elegir la intención antes de calcular](#guide-dataset-reconstruction-elegir-la-intención-antes-de-calcular)
  - [Publicar una vez, replicar el contenido exacto](#guide-dataset-reconstruction-publicar-una-vez-replicar-el-contenido-exacto)
  - [Tres identidades separadas](#guide-dataset-reconstruction-tres-identidades-separadas)
  - [Comparar reconstrucciones completas](#guide-dataset-reconstruction-comparar-reconstrucciones-completas)
  - [Recuperar una publicación sin recalcular](#guide-dataset-reconstruction-recuperar-una-publicación-sin-recalcular)
  - [Conflictos entre registros](#guide-dataset-reconstruction-conflictos-entre-registros)
  - [Certificados de equivalencia durables](#guide-dataset-reconstruction-certificados-de-equivalencia-durables)
  - [Cambios necesarios en WISDOM (no realizados aquí)](#guide-dataset-reconstruction-cambios-necesarios-en-wisdom-no-realizados-aquí)
- [26. Admisión de almacenamiento y limpieza segura](#guide-storage)
  - [Consultar y limpiar](#guide-storage-consultar-y-limpiar)
  - [Admisión](#guide-storage-admisión)
  - [Qué se puede eliminar](#guide-storage-qué-se-puede-eliminar)
  - [Checkpoints y publicación](#guide-storage-checkpoints-y-publicación)
  - [Límites actuales](#guide-storage-límites-actuales)
  - [Reutilización de dependencias y reparación](#guide-storage-environment-layers)
- [27. Pestañas de informes con assets completos gestionados](#guide-managed-reports)
  - [API del navegador](#guide-managed-reports-api-del-navegador)
  - [Visor nativo remoto y offline](#guide-managed-reports-visor-nativo-remoto-y-offline)
- [28. Espacio de investigación de Studies](#guide-research-analysis)
  - [1. Abrir el espacio](#guide-research-analysis-1-abrir-el-espacio)
  - [2. Declarar significado, no política de ejecución](#guide-research-analysis-2-declarar-significado-no-política-de-ejecución)
  - [3. Familias y preguntas](#guide-research-analysis-3-familias-y-preguntas)
  - [4. Cómo funciona el descubrimiento](#guide-research-analysis-4-cómo-funciona-el-descubrimiento)
  - [5. Explorar y guardar](#guide-research-analysis-5-explorar-y-guardar)
  - [6. Arquitectura, seguridad y límites](#guide-research-analysis-6-arquitectura-seguridad-y-límites)
  - [7. Informe de implementación](#guide-research-analysis-7-informe-de-implementación)
  - [8. Informe de la segunda simplificación](#guide-research-analysis-8-informe-de-la-segunda-simplificación)
- [29. Studies coordinados: estado y contratos](#guide-coordinated-studies)
  - [Estado: lanzamiento Fleet adaptativo nuevo y fijo; recovery/transferencia pendientes](#guide-coordinated-studies-estado-lanzamiento-fleet-adaptativo-nuevo-y-fijo-recoverytransferencia-pendientes)
  - [Catálogo de Fleet](#guide-coordinated-studies-catálogo-de-fleet)
  - [Propiedad y persistencia](#guide-coordinated-studies-propiedad-y-persistencia)
  - [Pruebas reproducibles de esta base](#guide-coordinated-studies-pruebas-reproducibles-de-esta-base)
- [30. Salud operativa, monitorización y recuperación deliberada](#guide-operational-reliability)
  - [Inspección sin cómputo](#guide-operational-reliability-inspección-sin-cómputo)
  - [Monitor independiente del proyecto](#guide-operational-reliability-monitor-independiente-del-proyecto)
  - [Correo opcional e incidentes sin spam](#guide-operational-reliability-correo-opcional-e-incidentes-sin-spam)
  - [Consola](#guide-operational-reliability-consola)
  - [Recuperación y YAML trasladado](#guide-operational-reliability-recuperación-y-yaml-trasladado)
  - [Política y evidencia de checkpoints gestionados](#guide-operational-reliability-política-y-evidencia-de-checkpoints-gestionados)
  - [Emergencia](#guide-operational-reliability-emergencia)
  - [Ampliación deliberada frente a recuperación](#guide-operational-reliability-ampliación-deliberada-frente-a-recuperación)
- [31. Seguridad de recursos compartidos — implementación en curso](#guide-resource-safety)
  - [Observar no concede GPU](#guide-resource-safety-observar-no-concede-gpu)
  - [Startup entre Studies](#guide-resource-safety-startup-entre-studies)
  - [RAM y CPU](#guide-resource-safety-ram-y-cpu)
  - [Perfiles, circuitos e informes](#guide-resource-safety-perfiles-circuitos-e-informes)
  - [Reservas del centro](#guide-resource-safety-reservas-del-centro)
  - [Aceptación pendiente](#guide-resource-safety-aceptación-pendiente)
- [32. Experiencia de desarrollo — implementación incremental](#guide-developer-experience)
  - [Empezar y descubrir](#guide-developer-experience-empezar-y-descubrir)
  - [Entender YAML](#guide-developer-experience-entender-yaml)
  - [Inspección, diagnóstico y seguimiento](#guide-developer-experience-inspección-diagnóstico-y-seguimiento)
  - [Tablas pequeñas sin exportar todo el experimento](#guide-developer-experience-tablas-pequeñas-sin-exportar-todo-el-experimento)
  - [Agentes: registro no significa conexión](#guide-developer-experience-agentes-registro-no-significa-conexión)
  - [Consola y Skills](#guide-developer-experience-consola-y-skills)
  - [Pendiente](#guide-developer-experience-pendiente)
- [33. Integración de agentes fijada al proyecto](#guide-integration)
  - [Inicio rápido](#guide-integration-inicio-rápido)
  - [MCP, clientes y permisos](#guide-integration-mcp-clientes-y-permisos)
  - [Aprobación humana y dependencias](#guide-integration-aprobación-humana-y-dependencias)
  - [HTTP local opcional](#guide-integration-http-local-opcional)
  - [Memoria y sincronización](#guide-integration-memoria-y-sincronización)
  - [Python y GUI](#guide-integration-python-y-gui)
- [34. Topología de grafos y calibración independiente](#guide-graph-science)
  - [Matrices FGW acotadas y reanudables](#fgw-blocks)
  - [Baricentros FGW reales y estructuras relacionales](#fgw-barycenters)

Resultados e informes de Work, dependencias históricas exactas y materialización comprimida explícita:
[Guía de resultados](#guide-work-results). Reutilizan runner, ResultStore, registro de productos e
import/export de Studies; no añaden entrenamiento ni inferencia estadística implícita a Works ordinarios.

[English](MANUAL.md) · Español

## 1. Modelo mental

LambdaForge ejecuta clases Python y posee la infraestructura que las rodea:

```text
YAML Work -> WorkConfig -> plan -> scheduler/Job -> WorkRunner -> Work.run()
```

El investigador implementa una subclase de `lambdaforge.Work` y su método `run()`. YAML elige la
clase, argumentos normales, recursos y repeticiones. LambdaForge resuelve entradas externas,
calcula identidad, enlaza servicios, captura salidas y conserva un resultado inspeccionable. Una
función o una clase que no herede `Work` no puede ser target de YAML.

La jerarquía durable es:

- Work: operación o estudio con nombre humano;
- Execution: una invocación en un destino;
- Run: un miembro científico de seed/variante;
- Attempt: un intento de completar ese Run;
- Job: el proceso o trabajo del scheduler que ejecuta el Attempt.

## 2. Instalación y estructura

Framework y proyecto consumidor son paquetes independientes instalados en el entorno del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge
python -m pip install -e .
python -m pip check
```

Para desarrollar el framework se sustituye la primera instalación por
`python -m pip install -e /ruta/absoluta/LambdaForge`. No se comparte el `.venv` del framework, no
se modifica `PYTHONPATH` en producción ni se copia el código fuente dentro del consumidor.
`lf init DIRECTORIO` crea `pyproject.toml`, un paquete `src/`, YAML y exclusiones seguras.

```text
proyecto/
  pyproject.toml
  src/mi_proyecto/work.py
  experiments/estudio.yaml
  data/entrada-pequeña.json
  .lambdaforge/                 # estado gestionado; ignorado por Git
```

`lf --help`, `lf help`, `lf COMANDO --help` y `lf help COMANDO SUBCOMANDO` son rutas equivalentes.
`lf` sin argumentos abre la Consola en un TTY; con redirección o `lf --json` imprime ayuda.
Clusters edita el mismo catálogo y servicio de credenciales que los comandos nativos y nunca pone
contraseñas en argv ni YAML. Las rutas de 0.13 `top`, `clusters setup` y `clusters modify` están
retiradas.

## 3. API de Work

Una subclase normalmente no define constructor y tiene una única entrada científica: `run()`.
Todas las propiedades de runtime fallan si se usan fuera de una ejecución gestionada.

| API | Vida | Responsabilidad |
|---|---|---|
| `run(**parameters)` | una vez por Attempt | cálculo científico y resultado JSON primario |
| `name`, `config` | inmutable | nombre, clase, parámetros y recursos normalizados |
| `inputs` | inmutable | procedencia y rutas resueltas de file/dataset tipados |
| `outputs` | Attempt | valores, ficheros/directorios, artefactos y datasets |
| `metrics` | append-only | historia escalar y valores finales |
| `checkpoints` | Run | estado explícito para reanudar |
| `cache` | reconstruible | valores simples y ficheros reutilizables ligados a identidad |
| `tools` | ejecución | programas externos y su procedencia |
| `progress` | snapshot | avance vivo `completed/total/message` |
| `log(...)` | stream | diagnóstico humano con fecha y flush |
| `resources`, `seed`, `trial` | inmutable | reserva y miembro del estudio |
| `run_dir` | Attempt | raíz durable avanzada |
| `temp_dir` | Attempt | temporales eliminados al finalizar |
| `source_dir` | inmutable | contexto del paquete consumidor |
| `resuming` | inmutable | existía checkpoint compatible al comenzar |
| `map(...)` | Job | concurrencia ordenada sin persistencia |
| `resume_map(...)` | Job | reanudación explícita por elemento y clave estable |

`outputs.value` acepta una vez valores JSON. `outputs.file/directory` es la vía recomendada para
artefactos nuevos; `outputs.artifact` importa de forma avanzada una ruta ya creada. `metrics.log`
acepta escalares finitos y `step/split` opcionales. `print()`, stdout, stderr y `logging` se capturan
en los logs del Job y en `work.log`. `self.log()` añade nivel y fecha; `progress.update` representa
avance, mientras `metrics.log` conserva evidencia científica.

## 4. Ficheros gestionados, cache, map, outputs y herramientas

### 4.1 Elegir almacenamiento

| Necesidad | Servicio | Sobrevive retry | Lo elimina `lf clean` | Resultado publicado |
|---|---|---:|---:|---:|
| bytes/texto/JSON pequeño reconstruible | `cache.put/get` | se reutiliza | sí | no |
| fichero descargable/calculable | `cache.file/fetch` | se reutiliza | sí | no |
| estado secuencial para reanudar | `checkpoints.file/save_json` | sí | solo cuando deje de ser requerido, según retención | evidencia de resume |
| fichero/árbol científico final | `outputs.file/directory` | pertenece al Attempt | solo duplicado publicado verificado | sí |
| intermedio desechable | `temp_dir` | no | automático | no |

`cache.path`, `checkpoints.path` y escrituras directas bajo `run_dir` son escapes avanzados. El
usuario que los elige también asume validación, atomicidad y registro.

Los checkpoints son recuperación, no modelos permanentes. `checkpoints.pin(name, reason=...)`
protege estado propio nombrado; `unpin(name)` solo cambia metadata. Publica modelos con
`outputs.from_checkpoint(name, checkpoint, release=True)` o datasets desde `source_checkpoint`.
Liberar exige publicación independiente verificada y finalización correcta de Execution.
La [guía de almacenamiento](#guide-storage) contiene ejemplos y retención.

### 4.2 ManagedFile y cache

El caso normal no exige rutas ni callbacks:

```python
self.cache.put("resumen", {"aceptadas": 1842, "schema": 3})
resumen = self.cache.get("resumen")
fallback = self.cache.get("ausente", {"aceptadas": 0})
```

`put(clave, contenido)` admite `bytes`, texto UTF-8 o JSON estricto y reemplaza atómicamente el
valor. `get(clave, default=None)` devuelve el tipo soportado original o el default si no existe una
entrada íntegra. No se admite pickle ni serialización implícita de objetos. El perfil de storage
elige la ubicación física; el Work normal no necesita hacerlo. `cache.path` queda como escape
avanzado explícito.

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work \
  --cache-root /scratch/USER/lambdaforge-cache
```

La raíz elegida se propaga a cada proceso Work, también a Jobs directos detached; no depende del
estado del shell. La ejecución local usa por defecto `.lambdaforge/cache` del proyecto.

Cuando una librería necesita una ruta o la construcción es pesada se usa `ManagedFile`:

```python
estructura = self.cache.file(
    f"estructuras/{identificador}.cif",
    build=lambda destino: crear_estructura(identificador, destino),
    validate=lambda fichero: fichero.size_bytes > 0 and estructura_valida(str(fichero)),
)
```

`build(destino: Path)` escribe un temporal real; su retorno se ignora. En un hit se comprueban
registro, SHA-256, tamaño y validador semántico. En un miss o entrada inválida se adquiere un lock
exclusivo entre procesos para esa clave, se vuelve a comprobar, se construye junto al destino, se
valida, se hace `fsync`, se promociona atómicamente y se actualiza el registro. Un fallo no publica
bytes parciales. Las claves son relativas; se rechazan rutas absolutas, `..` y symlinks.

`ManagedFile` es de solo lectura y compatible con `os.PathLike`: ofrece `str`, `path` como escape,
`open/read_text/read_bytes`, `exists` y metadata inmutable `key`, `sha256`, `size_bytes` y `scope`.
Los checkpoints guardan clave y evidencia de contenido, nunca la ruta física.

```python
limite = self.cache.rate_limit("archivo", requests_per_second=4)
fichero = self.cache.fetch(
    url,
    key=f"archivo/{identificador}.json",
    retries=5,
    retry_backoff=0.5,
    timeout=30,
    decompress="gzip",
    validate=registro_valido,
    rate_limit=limite,
)
```

`fetch` hace un GET HTTP(S) cacheable, no sustituye a un cliente HTTP general. Añade timeout,
reintentos después del intento inicial, backoff exponencial y gzip opcional. Los cortes de conexión,
lecturas chunked incompletas —también durante gzip—, estados 408/425/429 y 5xx se reintentan;
errores permanentes como 401 o 404 fallan de inmediato. Cada intento pasa por el limitador y empieza
con un temporal privado vacío. Al agotarlos, el error conserva URL, número de intentos y causa final.
Solo se publica atómicamente un fichero completo y validado, nunca bytes parciales. El limitador es seguro
entre threads de esa instancia de Work; no coordina Jobs distribuidos. `lf clean` muestra cada cache
de Work y `--apply` lo elimina solo cuando obtiene el lock exclusivo; un Work activo mantiene una
lease compartida.

**Dependencias de recuperación y observaciones externas.** Los mapas y recibos de un Run
conservado protegen sus namespaces de caché tras fallo, cancelación y cambios de código aceptados,
no solo durante una lease activa. La granularidad actual es el namespace de Work, no cada archivo
sobrante. GC muestra `protected_bytes`, `reference_chains` y `retained-recovery-dependency`, y avisa
de presión no resuelta sin destruir esos inputs. Borrar historial propietario libera referencias
solo si lo permiten sus dependientes nativos. Se conservan artifacts de Attempts de Works fallidos
con mapas para posible reparación exacta; las cachés no referenciadas siguen siendo limpiables.

`fetch` dentro de un Work guarda digest y contrato de petición hasheado en el Run original.
Reutiliza solo bytes exactos; si faltan, falla en vez de consultar una URL que podría haber cambiado.
Las adquisiciones científicas duraderas deben sellarse además con `checkpoints.file` y un recibo
de primera observación/digest declarado por el proyecto. Aceptar código permite localizar bytes
registrados entre namespaces, nunca demuestra equivalencia científica.

En `resume_map`, pasa/devuelve ManagedFile explícitos: las dependencias de entrada se conservan
también con `executor="process"`. En threads se registran además accesos a caché del callback;
el I/O oculto de subprocesos no puede inferirse automáticamente. Declara ecuaciones/configuración
relevantes en `scientific_policy={...}`, que se comprueba al restaurar; no metas frecuencia de logs
ni número de workers. Un resultado legado identificado solo por PDB ID sin dependencia de fuente
no acredita compatibilidad con coordenadas nuevas: migra fuentes originales exactas o exige una
adquisición nueva. Los logs distinguen restaurados y reconstrucciones/misses; el bloqueo por una
observación ausente incluye su digest esperado.

`Retry-After` numérico o fecha HTTP impone cooldown compartido por servicio en el mismo volumen
de caché, entre threads y procesos; por defecto agrupa el origen HTTP. No coordina máquinas sin
almacenamiento común ni sustituye el pacing local. Para hints JSON del proveedor, pasa
`cooldown_policy=HttpCooldownPolicy(json_field="retryAfter")` desde `lambdaforge.work`; puedes
agrupar servicios explícitamente con `cooldown_scope`. Se conserva el cooldown aunque se agoten
reintentos. La espera comprueba cancelación en pausas cortas; lecturas en curso siguen acotadas por
timeout. Errores permanentes fallan, sin fabricar metadata vacía.

Preview nativo sin ejecutar ciencia:

```sh
lf jobs recovery-dependencies JOB --json --limit 100
# Solo tras revisar, reparación de bytes desde artifacts propietarios:
lf jobs recovery-dependencies JOB --apply --json
```

Se separan presentes/no verificados, candidatos de copia, ausentes y registros sin fuente ligada.
El preview solo lee metadata. Coincidir en tamaño no verifica contenido: apply exige fingerprint
histórico exacto con basename lógico original y copia atómicamente bajo locks nativos. No mueve
checkpoints, usa symlinks, consulta la red ni ejecuta ciencia. Apply exige runtime gestionado actual.
Si los bytes se perdieron, permanecen no disponibles: ni la URL ni los recuentos los recuperan.
Las APIs equivalentes son `recovery_dependency_preview` y `repair_recovery_dependencies` de
`lambdaforge.work`.

`DatasetCache` sigue siendo el cache especializado de muestras serializadas. Work cache reutiliza
el `CrossProcessFileLock`, convenciones de huella/atomicidad y ownership, pero no usa su sobre opaco
porque un fichero científico debe seguir siendo path-like.

<a id="guide-dataset-validation-cache"></a>

#### Validación científica compartida de datasets

`self.cache` incluye identidad/semilla de la Run. Para comprobaciones costosas independientes
de la semilla, usa un recibo compartido dentro de `Work.run`:

```python
# El proyecto aporta validar_estructuras(root: Path).
recibo = self.validate_dataset(
    "dataset",  # entrada tipada exacta de self.inputs
    validator=validar_estructuras,
    contract="miproyecto.estructuras",
    version="1",
    parameters={"families": familias_seleccionadas},
)
# Construir loaders ligeros, sin repetir el validador profundo.
```

El callback lanza excepción o devuelve `False` si rechaza, `True`/`None` si acepta. Solo valida
contenido inmutable declarado: no ajusta normalización ni estado dependiente de semillas,
modifica datos o consulta entradas externas sin fijar. Declara los ajustes que afectan al
contrato en `parameters` y cambia su versión si cambian dependencias externas relevantes.
La clave incluye código consumidor nativo, ID exacto de contenido, ubicación, contrato/versión/
ajustes y runtime; no las semillas ni etiquetas de trials. Código desconocido/no versionado se
rechaza. Cambios de código/runtime invalidan conservadoramente la reutilización.

La primera llamada verifica bytes exactos mediante DatasetOperations y ejecuta el callback.
Las llamadas concurrentes comparten lock. La reutilización (`recibo["reused"]`) solo inventaría
metadata: dispositivo/inodo/tipo/tamaño/mtime/ctime. Cambiar o reemplazar archivos, índice o
ubicación exige verificar bytes y ciencia de nuevo; la corrupción falla, no se acepta como otra
identidad. `force=True` revalida todo explícitamente. Un fallo o modificación durante el proceso
no publica éxito. Recibos corruptos bloquean; force reconstruye esa caché desechable, nunca
sobrescribe el dataset. `timeout_seconds=3600` acota cada espera de lock; los logs indican espera
y reutilización/validación. Leases nativas protegen de GC la validación activa. Limpiar recibos
solo obliga a validar otra vez; no son checkpoints ni resultados científicos.

Soporte: POSIX, directorios DatasetArtifact v2 locales con checksums, máximo 200.000 entradas;
sin assets URI externos. Es una caché local de confianza bajo semántica normal del filesystem,
no una prueba criptográfica ante manipulación hostil de metadata o un filesystem que oculte
cambios. Sin `ignore_hash` global: `datasets verify`, transporte y publicación siguen exactos.
El servicio público `lambdaforge.data.DatasetValidationCache` admite `implementation` explícito
con la identidad nativa del código para llamadas fuera de Work.

**Adopción consumidora:** LF no intercepta llamadas Python arbitrarias. WISDOM debe mover
`StructuralSidecars.validate_assets` y la reconstrucción de grafos al callback, para después
vincular assets ya validados con checks ligeros. Conservar las llamadas profundas anteriores
junto a esta API no ahorra nada. No actualiza runtimes/Studies activos en caliente. La primera
validación aún puede ser larga; prepárala antes de usar GPU cuando sea posible, nunca simulando
actividad GPU para mantener reservas.

### 4.3 Map simple y reanudable

La operación habitual es directa:

```python
resultados = self.map(filas, procesar, workers=8, executor="thread", retries=2)
```

`map(items, function, *, workers=1, executor="thread", name=None, retries=0,
retry_backoff=0.5)` conserva el orden de entrada, limita concurrencia, informa progreso y no crea
cache ni checkpoints. `thread` es apropiado para I/O; `process` exige callback y argumentos
serializables mediante spawn.

Solo un cálculo largo que deba reanudar elementos elige la variante explícita:

```python
resultados = self.resume_map(
    filas,
    procesar,
    key="record_id",
    workers=16,
    executor="thread",
    resume=True,
    name="features",
    validate=resultado_restaurado_valido,
    retries=2,
    retry_backoff=0.5,
)
```

La firma completa es `resume_map(items, function, *, key, workers=1, executor="thread",
resume=True, name=None, validate=None, retries=0, retry_backoff=0.5)`. La clave es un campo de mapping, atributo
de dataclass/objeto o callable explícito; debe ser única y estable. El resultado conserva el orden
de entrada. `map(..., key=...)` conserva esta semántica por compatibilidad, pero el nombre
`resume_map` hace visible el efecto persistente y es el recomendado para código nuevo.

Cada elemento se guarda como JSON estricto ampliado solo con referencias `ManagedFile` de cache o
checkpoint. Para cada fichero devuelto, anidado en el resultado o tocado mediante `self.cache` dentro de callback
secuencial/thread se conservan scope, key, SHA y tamaño. Al restaurar se verifican primero las
dependencias. Si falta una, está corrupta o `validate` devuelve false, solo ese elemento vuelve a
pending. Una excepción del validador se informa porque puede ser un bug científico. Los reintentos
son por elemento y su agotamiento sigue siendo fail-fast. En executor `process`, el callback debe
ser serializable por spawn y las dependencias deben estar en item/resultado: el contexto de runtime
no cruza procesos.

### 4.4 Outputs y checkpoints

```python
informe = self.outputs.file(
    "informe",
    filename="informe.json",
    role="report",
    media_type="application/json",
    publish_to="resultados/informe.json",
)
informe.write_json(resumen)

figuras = self.outputs.directory("figuras", role="visualization")
renderizar(Path(figuras))
```

Un fichero gestionado tiene `write_text`, `write_bytes`, `write_json` y `build`, todos con promoción
atómica. Un directorio existe desde su declaración, permite hijos seguros con `/` y solo se hashea
al finalizar. Después de un `run()` correcto se validan tipo, containment y symlinks de todo el
conjunto y se registra automáticamente. Si una declaración falta o es insegura, la finalización
falla y ninguna declaración gestionada del conjunto entra en el resultado. No se llama además a
`outputs.artifact`.

`publish_to` es opcional para ficheros y directorios. Una ruta relativa parte del directorio que
contiene el YAML original; una absoluta se usa explícitamente. En remoto la publicación relativa
requiere el `project_root` del clúster y usa el directorio YAML equivalente bajo ese mirror. Nunca
resuelve bajo el hash del bundle/Job. Sin mirror se debe usar una ruta remota absoluta persistente;
LambdaForge rechaza una relativa en vez de publicar silenciosamente en almacenamiento interno
desechable.

LambdaForge valida y hashea primero el artefacto gestionado y después publica una copia atómica por
destino. Reutiliza contenido idéntico y rechaza contenido distinto existente salvo
`overwrite=True`. El resultado registra `published_to`, SHA-256 y tamaño. Tras persistir toda la
Execution, LambdaForge vuelve a hashear el destino y elimina por defecto los bytes redundantes del
Attempt. `retain_internal=True` conserva deliberadamente una segunda copia. Los outputs correctos
sin `publish_to` permanecen en el Attempt porque no existe otra copia. Los Attempts fallidos o
interrumpidos eliminan `artifacts/` parciales, pero conservan logs, métricas, resultado,
procedencia y checkpoints. Las escrituras directas en `run_dir` y rutas importadas con
`outputs.artifact` son escapes avanzados: la limpieza no adivina que bytes arbitrarios son
desechables. No se copian árboles grandes al controlador implícitamente. También se rechazan symlinks, cambios de
tipo y destinos de directorio que contienen su fuente o están contenidos en ella; `overwrite=True`
nunca autoriza sustituir la raíz del proyecto o del Attempt.

```python
indice = self.checkpoints.file(
    "leakage/mmseqs.tsv",
    build=lambda destino: construir_indice(destino),
    validate=indice_valido,
)
```

El checkpoint usa construcción temporal, registro de integridad y validador como el cache, pero
pertenece al Run y nunca es cache reconstruible. Retry reutiliza uno válido y reconstruye uno
ausente/corrupto/inválido. `save_json/load_json/exists` es la vía para estado pequeño. `--restart`
elimina el árbol compatible antes del nuevo Attempt.

### 4.5 Herramientas externas

```python
mmseqs = self.tools.require("mmseqs", version_args=["version"])
completado = self.tools.run(
    [mmseqs, "easy-search", str(consulta), str(base), str(salida)],
    name="MMseqs2",
    threads=self.resources.cpu,
    cwd=self.temp_dir,
    env={"PROJECT_MODE": "strict"},
    timeout=3600,
)
```

`require(executable, version_args=None, version_timeout=10)` resuelve `PATH` y solo ejecuta el probe
pedido. Devuelve un `Tool` path-like. `run(command, *, name=None, threads=None, cwd=None, env=None,
timeout=None, check=True)` exige argv y nunca usa shell. Emite stdout/stderr línea a línea, conserva
una cola acotada en `ToolResult`, registra duración/status y lanza `ToolExecutionError` si procede.
`threads` configura OMP/MKL/OpenBLAS/NumExpr solo en el hijo. La herramienta y su versión explícita
se registran una vez en `environment.json`.

#### Herramientas científicas Python aisladas

Se declaran en el **`pyproject.toml` del consumidor**, no en el YAML del Work:

```toml
[tool.lambdaforge.tool_runtimes.analysis_tool]
python = "3.11"
requirements = ["numpy==1.26.4"]
imports = ["numpy"]
entrypoints = {extract = "tools/extract.py"}
```

```bash
lf clusters bootstrap CLUSTER --project . --dry-run
lf clusters bootstrap CLUSTER --project .
```

Dry-run describe declaraciones y resolución pendiente sin instalar. Aplicar selecciona un Python
compatible independiente, resuelve conjuntamente todas las dependencias, comprueba imports y
conserva inventario/resolución exactos. Los envíos normales preparados hacen lo mismo. No instala
LF ni el paquete consumidor en ese prefijo: sus requisitos incompatibles quedan separados.
Se preparan todas las declaraciones; no hay selector adicional, hooks shell ni runner anidado.
Bootstrap no carga pesos de modelos y `Work.run` nunca instala paquetes.

El preflight científico no importa `pip` ni sus librerías privadas: interfieren con
`setuptools`/`distutils` cuando Triton los carga. El probe recoge metadata con la biblioteca
estándar y LF comprueba los pins PEP 440 fuera de ese intérprete. Bootstrap y Work comparten la
misma comprobación, sin degradar dependencias ni forzar variables de entorno de distutils.

```python
tool = self.tools.runtime("analysis_tool")
tool.preflight()  # Antes de productores costosos; run() también comprueba la primera llamada.
completed = tool.run(
    ["python", tool.entrypoint("extract"), "--input", str(entrada), "--output", str(salida)],
    threads=self.resources.cpu,
    cwd=self.temp_dir,
    timeout=3600,
)
```

`runtime(nombre)` exige preparación y, si falta, explica qué bootstrap ejecutar.
`entrypoint(nombre)` devuelve un `.py` independiente copiado/verificado por hash: máximo 32,
de 1 MiB cada uno. `python -I` evita imports accidentales del proyecto, módulos hermanos y entorno
consumidor. El intercambio usa rutas/JSON/NPZ explícitos. LF no interpreta proteínas ni contenido
científico. Rodear el productor con `checkpoints.file(..., build=..., validate=...)` permite
recuperar archivos completos sin repetirlos; pesos/checkpoints originales se leen, no se mutan.
Se conserva la historia normal de Attempts y no se promete continuación por epoch sin soporte.

`run(argv, *, threads=None, cwd=None, env=None, timeout=None, check=True)` admite `python` o un
ejecutable del `bin` aislado. Stdout/stderr llegan al log; cancelar/timeout termina únicamente el
árbol de procesos propio. Un código no cero lanza `ToolExecutionError` salvo `check=False`.
Los hilos toman por defecto el límite CPU del Work; no se permite aumentarlo ni reemplazar por
`env` variables del framework, intérprete, PATH, hilos o grants GPU. CUDA_VISIBLE_DEVICES y
NVIDIA_VISIBLE_DEVICES se heredan sin ampliar ni sustituir tokens. Declarar CUDA exige grant no
vacío y un tensor real en preflight; no existe fallback CPU silencioso ni selección física de GPU.
Es aislamiento de procesos/dependencias, **no un sandbox** para Python no confiable.

Cada paquete raíz exige `==` exacto o Git HTTPS con commit completo de 40 caracteres, sin secretos
ni condiciones. `index_url` usa PyPI por defecto; `extra_index_urls` y `wheel_sources` son HTTPS
explícitos. Todo es binary-only salvo excepciones raíz fijadas/revisadas en `source_packages`
y el propio Git inmutable declarado. No hay compilación de fallback ni instalación `--no-deps`.
Los backends de build revisados pueden ejecutar código: son declaraciones del proyecto confiable.
Torch exige `torch_index` y `torch_cuda="cpu"` o versión CUDA `major.minor`. Cada tabla
`wheel_sources` exige `url`, `packages`, `build_tag`, y versiones raíz con el mismo `+build_tag`.
Se comprueban origen/versión real e imports. Una wheel incompatible/ausente bloquea preparación.

El ID del entorno depende de paquetes/fuentes/Python/plataforma, no de cambios de código consumidor.
Los scripts pequeños tienen bundle independiente por contenido. Recibos de versiones fuera del
cache evitan deriva; inventario modificado falla sin reemplazar el prefijo inmutable. Procedencia
del entorno/resolución/scripts se añade a `environment.json`. Bindings por Job, recuperación y
leases de preparación/uso protegen entorno y scripts de GC. El pruning de reemplazos del entorno
principal no borra herramientas; almacenamiento/limpieza explícita nativos gestionan históricos
no usados. No se crea un entorno por proteína ni se importan directorios privados de checkpoints.

**Aceptación SLAE e integración pendiente en WISDOM:** `examples/isolated_tools.toml` contiene una
receta revisada **solo CPU**. En un entorno Linux/Python 3.11 temporal autorizado se importó SLAE
`078b94e436b8eceb02251a254c3c224358ec6ee2` y cargó el checkpoint base con `device="cpu"`, tras
verificar SHA-256 `ac5bf9613209fc310145e58f1aa7ddc9111d833a131488fe30ea738d51f7bb99`
(882.525.854 bytes). Versiones observadas: Torch 2.6.0+cpu, torchvision 0.21.0+cpu, NumPy 1.26.4,
ESM 3.2.1, Transformers 4.48.1, PyG 2.7.0, scatter 2.1.2+pt26cpu, cluster 1.6.3+pt26cpu,
sparse 0.6.18+pt26cpu, runstats 0.2.0, e3nn 0.6.0, atomworks 3.0.4 y Lightning 2.6.6.
Provisioning conserva inventario/resolución completos. Esto **no valida CUDA ni extracción**.
WISDOM debe declarar su runtime objetivo revisado, aportar script independiente, sustituir el
import en proceso por la llamada pública y validar archivos restaurados/producidos. Coordinar
una extracción pequeña antes de dar por aceptado el adaptador científico. No se modifica WISDOM
ni se ejecutan clusters en esta implementación. Después se puede recuperar el Job fallido por el
ciclo nativo con reconocimiento de cambio compatible, no reiniciarlo todo silenciosamente.

## 5. Referencia YAML

Los campos superiores son `name`, `run`, `with`, `resources`, `seeds`, `search`, `objective` y
`steps`. Un documento define exactamente `run` o `steps`; los campos desconocidos fallan.

```yaml
name: evaluar
run: proyecto.work.Evaluar
with:
  threshold: 0.5
resources:
  cpu: 4
  memory: 8GiB
  gpu: 1
  gpu_memory: 12GiB
  time: 4h
  storage: 20GiB
  processes: 2
```

Los recursos son la reserva absoluta. `processes` no supera CPU. Se admiten unidades decimales y
binarias de bytes y `s/m/h/d` para tiempo. `lf validate` importa cada Work y comprueba herencia,
constructor, firma, tipos evidentes, recursos, entradas, expansión y referencias sin enviar nada.
`lf explain` muestra docstring, tipos, valores y defaults.

## 6. Ficheros y datasets

```yaml
with:
  manifiesto: {file: ../data/manifest.json}
  corpus: {dataset: research-corpus@3}
```

Un `file` o directorio se resuelve respecto al YAML, se comprueba/hashea y se pasa como `Path`. En
remoto, hasta el límite inline por defecto de 10 MiB se copia al bundle inmutable. Una ruta mayor
que pertenezca al proyecto sigue el contrato de mirror de la sección 10: se conserva su nombre
relativo al proyecto, el worker recibe la ruta absoluta remota equivalente y se comprueban tipo,
bytes y SHA-256 antes del scheduler y otra vez en el worker. La identidad de directorio es un flujo
canónico y versionado: componentes lógicos NFC ordenados por bytes UTF-8, separadores, tipos,
longitudes y directorios vacíos explícitos, sin depender de enumeración, locale, fechas ni permisos.
Los bundles históricos conservan el lector de su hash original. No se transfiere nada grande
implícitamente. Una ruta grande externa al proyecto se rechaza: debe ser un dataset gestionado o
formar parte de un layout de proyecto/datos explícito y revisable.

El marcador `file` solo describe una entrada que debe existir antes del envío. Un resultado nuevo
se crea en Python con `outputs.file/directory/value/dataset`; no hay otro esquema YAML de outputs.
El artefacto gestionado es el default seguro y `publish_to` solo se añade cuando una herramienta o
persona necesita además una ruta convencional, teniendo presente que local y remoto son sistemas
de ficheros físicos distintos.

`dataset` resuelve una versión exacta e inmutable, content ID y placement válida. Se publica desde
Python con `self.outputs.dataset(name=..., version=..., members=...)`. Los miembros se escriben en
JSONL, se copian y hashean assets, se valida el índice, se calcula identidad independiente de ruta,
se promociona staging atómicamente y se registra placement. Reutilizar nombre/versión con otro
contenido se rechaza; DatasetArtifact v1 sigue siendo legible.

### 6.1 Reutilizar, reconstruir y recuperar publicación

Antes del cálculo costoso usa `lf datasets preflight NOMBRE@VERSION --intent publish|reuse|rebuild
--on CLUSTER`. El Work productor llama primero a `outputs.dataset_preflight(...)`: no se pueden
inferir nombres de outputs desde Python arbitrario antes de `run()`.
`outputs.dataset(..., intent="rebuild", scientific_identity=...)` sella un candidato fijado y no
registrado; checksums exactos intactos. `lf datasets compare RAIZ_A RAIZ_B --on CLUSTER
--verifier proyecto.modulo:funcion --policy policy.json --output report.json` verifica todos los
bytes antes del verificador científico. `lf datasets publish-candidate RAIZ --version NUEVA
--on CLUSTER` previsualiza recuperación solo de publicación; `--apply` publica sin recalcular.
La consola descubre todas las ubicaciones configuradas y señala conflictos de identidad.
[Reconstrucción de datasets](#guide-dataset-reconstruction): contratos, seguridad, retención y WISDOM.
En 0.17.1, `publish-candidate` también acepta solicitudes guardadas antes de copiar: verifica
inventario/declaración y `--apply` solo copia/registra, sin ejecutar Work. Para índices antiguos
retenidos, `DatasetPublisher.prepare_publication` requiere la declaración original explícita;
nunca infiere contratos científicos ausentes ni cambia identidades de bytes.

### 6.2 Réplica nativa comprimida

`lf datasets replicate NOMBRE@VERSION --source ORIGEN --destination DESTINO` previsualiza; `--apply`
copia bytes exactos y registra la colocación verificada en destino y controlador.
`--route auto|direct|relay` comprueba envío y descarga SSH directos; relay exige consentimiento explícito.
SSH directo no copia credenciales del controlador; relay no usa archivos intermedios locales.
Consola **Datasets → DatasetVersion → Replicate…** usa el mismo servicio, confirmación y progreso
en segundo plano. `--apply` confirma un Job operativo persistente, no que la copia ya esté
verificada. Tras recibir su ID puedes cerrar LF. SSH directo se ejecuta en el cluster origen sin
el PC; relay usa un supervisor local independiente y necesita ese PC encendido/conectado.
Consulta `lf datasets transfers [NOMBRE@VERSION]` o `lf datasets transfer-status JOB_ID`.
El destino requiere `storage.dataset_root`
y runtime LambdaForge disponible, no Work científico/claim GPU. Checksums, staging seguro,
admisión en destino e identidad inmutable siguen siendo obligatorios. Rechaza conflictos y reutiliza
destinos idénticos. [Guía](#guide-dataset-reconstruction-publicar-una-vez-replicar-el-contenido-exacto).

### 6.3 Gestión explícita de conflictos

Gestión nativa de conflictos con preview/apply: `datasets adopt NOMBRE@VERSION --on ORIGEN
--content-id ID` elige referencia futura del controlador tras verificar todos los bytes;
`datasets delete/remove NOMBRE@VERSION --on DESTINO --content-id ID` selecciona una identidad
exacta incluso en conflicto. Delete borra bytes gestionados, remove los conserva; ambos bloquean
consumidores activos y archivan índices previos. Consola **Datasets → identidad → Manage copies…**
expone las mismas acciones confirmadas. No cambia otras copias/Runs antiguas; operaciones sin hash
siguen rechazando ambigüedad. [Detalles](#guide-dataset-reconstruction-conflictos-entre-registros).

## 7. Secuencia, paralelismo, seeds y búsqueda

Las ramas condicionales pertenecen a un Study cuando responden una pregunta experimental conjunta.
Usa `when: {parent: {in: [a, b]}}` (AND entre padres) o la igualdad escalar antigua. Se omiten las
claves inactivas, las dependencias no requieren orden en YAML y generación/análisis comparten
geometría. Solo `resources` raíz se hereda por pasos; se rechaza `with`, seeds/replicates,
search/sweep, execution, objective y analysis raíz en vez de ignorarlos. La
[guía condicional numerada](#guide-conditional-studies) explica conteos exactos, referencias,
identidad, capacidad y la diferencia entre tiempo del scheduler y presupuesto lógico.

### 7.1 Defaults por intención y autoridad resuelta

El camino común no necesita enteros de seed ni ajuste manual del controlador. Con `search.space` y
objetivo, omitir política significa: stream `replicate` del proyecto, replicación mínima uno,
objetivo `balanced`, startup geométrico, sampler/pruning/carrera/confirmación nuevos automáticos,
candidatos deterministas incrementales, convergencia científica y paralelismo interno automático.
El stem del YAML es el nombre por defecto. La propiedad externa sigue explícita:
`resources.gpu` nunca se convierte en «todas las GPU disponibles».

`lf config resolve FICHERO` muestra el `ResolvedStudyConfiguration` exacto; `--format json|yaml`
sirve para herramientas. Se persiste antes del cálculo en `execution.json` y
`resolved-configuration.json`: versiones de políticas, ejecución resuelta, primera identidad de
seed, diseño, goal, parada/confirmación/generación, objetivo y recursos duros. `lf seeds [--role
replicate|confirmation] [--count N]` inspecciona los streams. Cada resultado/Run guarda valor,
ordinal, rol, namespace y versión; el export incluye YAML, configuración resuelta y seeds reales.

El stream es una permutación con clave derivada mediante SHA-256 sobre ordinales de 31 bits. El rol
confirmation ocupa el otro dominio del bit alto: cada rol posee (2^{31}) valores sin colisión,
compatibles con NumPy, y ambos dominios automáticos no se solapan. `seeds` explícitas siguen siendo
exactas y finitas. `replicates: N` raíz o `sweep.replicates: N` elige ordinales `[0, N)`.

Sin presupuesto de candidatos, la búsqueda termina por convergencia. El generador Sobol/mixto
mantiene un prefijo estable: materializa una ventana computacional acotada y solo la amplía cuando
la política científica pide otra región. Los candidatos previos nunca cambian. `goal: optimize`,
`balanced` o `understand` cambia el valor de las preguntas pendientes en la misma política. Los
presupuestos explícitos siguen siendo techos y agotarlos no se etiqueta como convergencia. Se
persisten motivos como `CONVERGED_BALANCED`, `CANDIDATE_BUDGET`, `RUN_BUDGET`, `TIME_BUDGET`,
`SEARCH_SPACE_EXHAUSTED` y `NO_POSITIVE_VALUE_ACTION`, separados del estado científico.

Un sweep auto empieza todas sus celdas en `replicate[0]`. Sus diferencias pareadas usan la secuencia
de confianza empirical-Bernstein plug-in predecible, simultánea, uniforme en el tiempo y adaptativa
a varianza `paired-pm-eb-cs-v2`, manteniendo validez con parada opcional. La multiplicidad cubre solo
la familia primaria authored: tratamientos frente a `reference`, o relaciones pairwise necesarias
para identificar un top set práctico. El controlador puede comprometer exactamente un bloque
completo siguiente como lookahead y ordena sus celdas por duración previa; no usa ese bloque para
decidir hasta completarlo. Así una celda lenta no deja GPUs seguras ociosas y se gasta como máximo
un bloque extra. Se persisten por separado líder puntual, conclusión/estabilidad descriptiva exacta
y estado/cobertura secuencial formal. Se exige objetivo acotado y la equivalencia requiere
`practical_margin`. El alpha de familia predeterminado es `0.05`; el ajuste avanzado
`sweep.sequential_alpha` debe estar en `(0,1)` y es inválido con `replicates` fijo. En sweep no
existe pruning HPO ni carrera por celda.

Runtime v2 consume el prefijo persistido de adquisición (`seed_metadata.ordinal` en streams,
orden declarado en seeds explícitas), nunca el orden numérico. Un bloque posterior completo no
salta uno anterior incompleto. Recovery restaura el compromiso anticipado existente y el preview
no lo modifica. Manifests/decisiones v1 históricos siguen consultables e inmutables. Hipótesis
predictiva, soporte/estabilidad descriptivos y resolución formal son distintos: falta de soporte
no hereda estabilidad del modelo y parar por bloque incompleto no es conclusión formal.
Consulta la [auditoría de corrección y aceptación pendiente](ai/IMPLEMENTATION.es.md#history-scientific-audit).

Los Studies largos conservan su inicialización científica inmutable en
`study/controller.json` → `initialization`: objetivo completo, margen práctico, espacio declarado
(tipos, escalas log y condiciones `when`), política resuelta y metadatos de streams de seeds.
`recent` es solo una cola visual de 25 acciones. Los estudios antiguos recuperan la inicialización
una vez desde su historial append-only cuando existe; si falta metadata obligatoria se muestra un
error explícito. El resumen vivo reutiliza el snapshot científico del controlador, por lo que abrir
la consola no reinterpreta el margen de equivalencia ni convierte una dimensión log en lineal.

La variabilidad de seeds se aprende de candidatos repetidos, nunca de diferencias entre
arquitecturas. `unresolved` significa que no hay escala interna identificada; `provisional`, que hay
una estimación pero sigue incierta su varianza o su soporte contextual. `calibrated` exige que la
incertidumbre muestral de la varianza sea menor que la cobertura contextual respaldada. Un candidato
con dos seeds permanece provisional. El modelo conserva realizaciones deterministas de varianza
mediante chi-cuadrado inversa escalada, condicionadas a la aproximación normal interna, y muestra
rango de sigma, grados de libertad residuales, soporte pareado, cobertura contextual y posible
heterogeneidad. El rango expresa incertidumbre del modelo, no una garantía libre de supuestos.
Los efectos comunes de seed siguen formando parte de la incertidumbre poblacional aunque se
cancelen en diferencias pareadas. El fallback predictivo conservador también conserva la varianza
de contextos ruidosos, sin tratar la escala pooled como homogénea. La incertidumbre normal de la
varianza sigue la relación con grados de libertad explicada en el
[manual estadístico de NIST](https://www.itl.nist.gov/div898/handbook/prc/section2/prc231.htm).

Una seed adicional compite con probes de parámetros e interacciones mediante el mismo planner y
valor por coste. Su valor incluye cuánto puede reducir la incertidumbre de muchas preguntas
científicas a la vez. Contenders y contextos poco repetidos pueden justificar nuevas seeds
compartidas después de la cobertura inicial; no hay cuota universal por candidato. Ese valor global
disminuye con la evidencia adicional. Las seeds frescas de confirmación siguen separadas. Una
continuación de fidelidad desde checkpoint puede llevar `PRUNER_CALIBRATION_EVIDENCE`: completar
esa curva permite mejorar la calibración del endpoint. Estas Runs están protegidas del pruning
competitivo; la mala calibración sigue desactivando el pruning fuerte. Las métricas diagnósticas no
se convierten en objetivos de selección ocultos.

La evidencia de recursos distingue un pico exacto, una trayectoria física completa muestreada y un
límite inferior interrumpido. Las lecturas NVML conservan su máximo físico aunque el worker salga,
junto con máximos del allocator, cadencia e intervalo de incertidumbre. Una trayectoria completa
muestreada puede enseñar al predictor sin afirmar un pico vital exacto; las Runs incompletas siguen
censuradas. `exact_history_count` y `sampled_history_count` muestran la diferencia. El mínimo de
seguridad y la VRAM libre actual siguen limitando cada nueva colocación. Los bloqueos repetidos se
agregan en `hpo-control/resources/blocked-summary.json`; el replay conserva evidencia factual.

Las métricas con `step` publican `step/sec`; `progress.update(completed=...)`, unidades de
progreso/segundo. Estas señales acotadas alimentan el aprendizaje genérico de throughput sin nombres
específicos de una aplicación. ARI separa seguridad de utilidad: compara throughput agregado entre
concurrencias observadas y puede dejar capacidad segura libre cuando otra Run reduciría evidencia
por tiempo. La concurrencia configurada es un techo, no un objetivo de número de procesos.

La inferencia científica viva usa 32 realizaciones. Study Analysis y los informes humanos usan un
presupuesto cacheado de mayor precisión (hasta 256) fuera del heartbeat de planificación. Ambos
conservan objetivo, margen y geometría declarada. Los paneles e HTML muestran realizaciones,
resolución Monte Carlo, soporte directo/censurado/predictivo y diagnóstico de seeds. Estabilidad
significa acuerdo de una conclusión cualitativa entre realizaciones, **no** la probabilidad de que
sea verdadera. La precisión puede cambiar una estimación cercana sin cambiar la configuración.

```yaml
name: comparacion
steps:
  - name: preparar
    run: proyecto.Preparar
  - parallel:
      - name: a
        run: proyecto.EntrenarA
        with: {data: {from: preparar.dataset}}
      - name: b
        run: proyecto.EntrenarB
        with: {data: {from: preparar.dataset}}
  - name: comparar
    run: proyecto.Comparar
```

Cada nivel espera al anterior. Cada miembro paralelo usa un proceso spawn aislado. Sweeps fijos y
búsquedas adaptativas poseen una reserva exterior y planifican Runs hijos mediante el mismo
dispatcher ARI. Sus controladores científicos difieren —evidencia obligatoria frente a evidencia
seleccionada—, pero la colocación física no se duplica. Una referencia requiere un único Run
productor. Ramas, condiciones y bucles complejos pertenecen a Python.

`seeds` crea Runs independientes. Con `objective`, omitir `strategy` activa por defecto todas las
optimizaciones seguras: inicio Sobol scrambled, propuestas dependientes de resultados, carrera
probabilística de seeds, pruning de curvas, convergencia, recuperación acotada y confirmación con
seeds nuevas. `proposal_pool_size` acota el pool Sobol interno (por defecto hasta 16 veces
`trials`, máximo 4096), mientras `trials` solo limita candidatos realmente ejecutables. Solo los Trials
propuestos aparecen en la Consola de investigación. El pool se conserva una sola vez y cada
especificación ligera referencia únicamente su candidato y seed, por lo que la memoria de
planificación crece linealmente y no obliga a debilitar un YAML válido. El tamaño del pool tampoco
multiplica el análisis científico en la ruta crítica: el análisis vivo construye una base inmutable
reutilizable, cachea distancias, usa 32 realizaciones deterministas y soporte
representativo acotado para contrafactuales/interacciones. La shortlist científica rota en cada
decisión e incluye siempre la propuesta del sampler de optimización; todo el pool continúa
elegible sin escanearlo por pregunta/remuestreo ni bloquear la recogida de Runs. La evidencia
persiste los tamaños de pool, referencia y soporte, mientras el análisis final puede usar un
presupuesto más denso fuera del scheduler. Si se omite `startup_trials`, LambdaForge construye un
`InitialDesignPlan` determinista. Su matriz incluye intercept, términos numéricos lineales y
cuadráticos soportados, contrastes categóricos independientes y actividad/valor condicional. La
selección greedy conserva primero el rank alcanzable del pool, después soporte discreto/numérico,
geometría D-optimal y separación maximin. Los anchors protegidos dependen del `ParameterSpace` y
`trials`, nunca de GPU o paralelismo. Un `startup_trials` positivo fija el presupuesto exacto de
anchors pero conserva esa selección geométrica.

Hay tres capas explícitas: planificación científica pregunta qué evidencia debe existir;
planificación de recursos pregunta qué puede ejecutarse ahora; dispatch inicia el proceso aislado.
Los anchors pueden esperar o reordenarse sin quedar cancelados. Slots sobrantes ejecutan candidatos
replanificables `OPPORTUNISTIC_COVERAGE` sin agrandar el diseño protegido. Una imposibilidad física
dura reemplaza el anchor conservando obligaciones y persiste `REPLACE_STARTUP_ANCHOR`. El estado
versionado conserva además rutas de Attempts completados e identidades de acciones pendientes: un
reinicio reconstruye la misma deuda de anchors sin repetir Runs ya terminadas. Después,
`sampler: auto` prefiere GP mixto qLogNEI de
BoTorch si está instalado `lambdaforge[adaptive-hpo]` y hay evidencia suficiente; ante dependencia
ausente o inestabilidad numérica usa el surrogate mixto k-NN determinista. Es un modelo de decisión
realmente conjunto: números, categorías e indicadores de activación condicional comparten un mismo
vector, por lo que el posterior puede depender de interacciones. Si las seeds repetidas producen
error estándar, se pasa como ruido observado; qLogNEI considera ese ruido y los miembros pendientes
en su totalidad. La evidencia se agrupa por rung acumulativo exacto: cada
`(candidato, target, maximum)` tiene media/error entre seeds propios y un candidato puede producir
varias observaciones. La carrera de seeds solo compara un mismo rung; nunca reetiqueta una media
heterogénea con fidelidad máxima, mínima o media. Un Run pendiente conserva su fidelidad objetivo
real en vez de representarse como un resultado ficticio a presupuesto completo. La fidelidad es
una entrada normalizada explícita:
espacios numéricos usan `SingleTaskMultiFidelityGP`, los mixtos/categóricos
conservan esa coordenada en `MixedSingleTaskGP` y el fallback k-NN combina distancia de parámetros y
fidelidad de forma conservadora. La adquisición puntúa candidatos nuevos a fidelidad objetivo. La
familia log-EI se usa por su mayor estabilidad numérica
([guía de adquisición de BoTorch](https://botorch.org/docs/optimization)).
La arquitectura se describe con precisión como **surrogate sensible a fidelidad más scheduler de
acciones externo sensible a coste**. No afirma implementar una knowledge-gradient multi-fidelidad
única para todo espacio mixto/condicional; promociones, seeds y candidatos nuevos compiten en el
controlador independiente del provider.

El objetivo admite restricciones de resultado explícitas:

```yaml
objective:
  metric: val_auprc
  mode: max
  constraints:
    val_accuracy: {min: 0.55, seed_aggregation: worst}
    val_kappa: {min: 0.05, seed_aggregation: lcb, confidence: 0.95}
```

Cada restricción acepta `min`, `max` o ambos y toma el valor del checkpoint donde la utilidad fue
mejor. `seed_aggregation` es `mean` por defecto para compatibilidad, `worst` comprueba la seed menos
favorable y `lcb` aplica conservadoramente un intervalo normal bilateral (`confidence`, 0.95 por
defecto) a límites inferiores/superiores. LCB con menos de dos seeds y cualquier métrica ausente
fallan cerrado. Una métrica puede ser simultáneamente componente de utilidad y constraint: si es
factible sigue aportando de forma gradual. Los candidatos no factibles permanecen auditables, pero
no entran en carrera, surrogate ni selección. Nunca se infieren constraints por el nombre.

Para un compromiso gradual deliberado, `objective.metrics` define una utilidad compuesta. Cada
componente exige `mode`, `weight` no negativo y `range` finito fijo; se puede agregar mediante
`weighted_mean`, `geometric` o `chebyshev`. Los pesos se normalizan al validar y los rangos nunca
dependen de observaciones futuras. Solo se combinan componentes del mismo step entero real.

```yaml
objective:
  aggregation: geometric
  metrics:
    val_auprc: {mode: max, weight: 3, range: [0.0, 1.0]}
    val_balanced_accuracy: {mode: max, weight: 1, range: [0.0, 1.0]}
  constraints:
    val_kappa: {min: 0.05}
```

Una única utilidad gobierna adquisición, seeds, pruning, fidelidad, confirmación y ranking. Las
constraints siguen siendo guardas duras separadas. Componentes, contribuciones y frente Pareto
diagnóstico permanecen visibles sin convertirse en una segunda política oculta.

Por defecto se inicia con una seed declarada por candidato y se generan tres seeds de confirmación
deterministas y disjuntas. Puede cambiarse cada valor y `confirmation_seeds: []` desactiva
expresamente la confirmación. El ruido de seed se identifica solo con resultados repetidos dentro de
cada candidato, conservando efectos comunes de seed en la incertidumbre poblacional; la dispersión entre
candidatos nunca se rebautiza como ruido aleatorio. Antes de esa calibración una estimación de una
sola seed conserva incertidumbre explícita. El orden compartido permite diferencias pareadas

$$
d_s=Y(i,s)-Y(i^\star,s).
$$

Se añade evidencia mientras

$$
P(\mu_i\geq\mu_{i^\star}-\epsilon\mid D)\geq\delta,
$$

donde `objective.practical_margin` es la autoridad canónica para \(\epsilon\) y
`seed_racing.probability_threshold` es \(\delta\). Los
candidatos dominados dejan de recibir seeds puramente competitivas, pero una repetición representativa
todavía puede aportar calibración global. La incertidumbre competitiva y las preguntas científicas
pendientes compiten por coste temporal observado. `replication.minimum` (`min_seeds` en YAML legacy) es una obligación estricta
tras proponer un candidato: el replanning no puede eliminar esas seeds. Una seed podada satisface
evidencia intentada/censurada, no respuesta completa; un fallo de infraestructura solo cuenta tras
agotar sus reintentos. Las seeds posteriores siguen siendo adaptativas. La selección final usa una
cota conservadora o, preferiblemente, la media de
`confirmation_seeds` nuevas sobre un top-K congelado. `trials` es el contrato de parada por defecto:
se proponen todos los candidatos declarados salvo que se alcance un límite explícito de Runs o
tiempo. `max_runs` y `max_time` limitan el gasto total. `convergence_patience` vale cero por defecto
(desactivado). Un valor positivo activa expresamente la parada por racha cuando esa cantidad de
observaciones completas y a fidelidad máxima no mejora el incumbent más que `min_improvement`.
Esta política simple no demuestra cobertura del espacio mixto y no debe activarse solo porque la
confianza HPO sea baja. Pruning y carrera de seeds siguen activos de forma independiente. Cada
evento terminal persiste si finalizó por presupuesto de candidatos/Runs/tiempo, pool agotado,
convergencia explícita o ausencia de otra acción útil.

Cada Run de seed conserva dos valores deliberadamente distintos. `current` es la observación del
último step y se usa para proyectar curvas comparables y decidir pruning. `best` es el mínimo/máximo
de toda la curva terminada y es el valor de checkpoint usado por la carrera de seeds, el surrogate
y el ranking final. Así una época tardía sobreajustada no infravalora un Run terminado, pero un
punto temprano prometedor tampoco rescata un Run parcial podado. El valor HPO de un Trial es la
media de los mejores valores por seed, nunca la seed más afortunada. Las seeds frescas de
confirmación y la cota conservadora reducen el sesgo de validar repetidamente. El suelo inicial por
defecto es una seed declarada por Trial propuesto: permite cubrir candidatos y después comprar
únicamente las repeticiones, comparaciones con el incumbent y pares que aportan evidencia, en lugar
de gastar tres seeds indiscriminadamente.
Confirmación nunca recibe pruning de rendimiento ni preemption oportunista. Si falta o falla una
seed requerida, persiste `summary.confirmation.status=incomplete` y
`confirmation_incomplete=true`; la media sesgada del subconjunto superviviente nunca selecciona el
modelo.

El diario append-only `hpo-control/decisions.jsonl` explica inicialización, surrogate/fallback,
`START_NEW`, `ADD_SEED`, `PROMOTE_FIDELITY`, convergencia, confirmación y final. `state.json` conserva el
snapshot compacto; `summary.adaptive_controller` enlaza ambos sin copiar modelos ni checkpoints.

El bucle de adquisición está dirigido por eventos, no por barreras. Cada observación terminal
reevalúa `START_NEW`, `DESIGNED_PROBE`, `ADD_SEED`, `PROMOTE_FIDELITY` y `RESUME_PREEMPTED`.
Equilibra dos cantidades derivadas de la evidencia:

- oportunidad de optimización \(O\): mejora práctica posterior aún disponible en el pool finito;
- incertidumbre científica \(K\): entropía normalizada de conclusiones no resueltas sobre
  parámetros e interacciones por pares.

Su magnitud relativa produce pesos automáticos de rendimiento/comprensión; ningún número de trials,
cambio de fase o umbral YAML oculto controla la transición. Cada acción registra valor de
rendimiento, reducción esperada de entropía, coste incremental observado y prioridad. Son scores de
diseño empírico-bayesianos auditables, no ganancia de información Shannon calibrada. Un probe puede
superar a un candidato algo mejor predicho si su contrafactual emparejado resuelve más incertidumbre
material por coste; si vuelve a aumentar la mejora práctica, optimización recupera prioridad.

`goal` modifica esas prioridades; no es un permiso que descarte el valor científico. Cuando el
screening y el conjunto práctico de contendientes se estabilizan, LambdaForge deja de proponer
candidatos ordinarios de optimización y entra en confirmación con seeds nuevas. Las obligaciones
protegidas del diseño inicial siguen completándose y los huecos libres pueden recoger response
coverage, comparaciones emparejadas o continuaciones desde checkpoint de mayor valor. La
confirmación conserva prioridad y solo su estado real persistido puede satisfacer convergencia. Un
límite duro de Runs, tiempo o capacidad mantiene un motivo de parada incompleta distinto.

El pruning de curvas tiene dos niveles de disponibilidad. El operativo indica que endpoints
terminados permiten identificar el modelo; el fuerte exige además resolución residual de muestra
finita, cobertura retrospectiva de intervalos, discriminación probabilística y una tasa de podas
perjudiciales compatible con el umbral escrito. Solo este último autoriza paradas competitivas.
Anchors iniciales, evidencia mínima obligatoria, confirmación y continuaciones científicas están
protegidos. Search coverage registra intentos, response coverage respuestas terminales y scientific
coverage el soporte de la conclusión exacta mostrada; ninguno sustituye a otro.

Startup puede entrelazarse con decisiones del modelo. Identidades
y fidelidades pendientes condicionan el surrogate. La cola no despachada es provisional: nueva
evidencia puede sustituir una acción no-anchor con coste científico cero y registrar
`CANCEL_SCIENTIFIC_ACTION`. Un anchor protegido registra `CANCEL_PLANNED_DISPATCH` o
`DEFER_STARTUP_ANCHOR` y sigue pendiente. Cada acción guarda valor, score, coste y alternativas.
Una acción puntuada ya activa solo puede preemptarse tras 30 segundos, con un checkpoint propio no
symlink, y si la siguiente alternativa útil supera en un 50 % tanto su prioridad original como el
valor actual de continuar. Startup sin score comparable y toda confirmación quedan protegidos.
LambdaForge escribe una parada cooperativa, espera que el Work/Lightning retorne en una frontera
segura y después reanuda el mismo Run y target de fidelidad desde checkpoint; nunca mata el worker
por esta optimización. El diario distingue `PREEMPT`, `PAUSE`, `CONTINUE` y `RESUME_PREEMPTED` con
prioridades antiguas/nuevas y el cambio de evidencia. `scheduler_preempted` es neutral, nunca
pruning de rendimiento.

El pruning de rendimiento y la continuación científica responden preguntas distintas.
`PERFORMANCE_PRUNE` dice que gastar más en esa curva no compensa para encontrar el óptimo; su prefijo
sigue siendo evidencia censurada de search coverage, nunca un objetivo final inventado. Si después
el análisis de contextos emparejados determina que ese mismo Run es especialmente informativo para
un valor o interacción, `SCIENTIFIC_CONTINUATION` reanuda su checkpoint durable con el mismo Trial y
seed como un nuevo Attempt. Ese Attempt evita el pruning competitivo; fronteras de
checkpoint/fidelidad, preemption, recursos y presupuestos globales de Runs/tiempo siguen activos.
No consume otro candidato de `trials`. La Consola muestra juntos prune original, Attempt reanudado y
pregunta objetivo.

### Preguntas científicas, confianza y regiones prácticas

El controlador y el `StudyAnalysis` final llaman al mismo `ScientificQuestionAnalyzer`; la TUI no
ajusta otro modelo explicativo. Cada parámetro tiene exactamente una conclusión actual:
`PREFERRED`, `PREFERRED_REGION`, `PRACTICALLY_EQUIVALENT`, `WEAK_PREFERENCE`, `FLAT`,
`CONTEXT_DEPENDENT`, `NO_CLEAR_PREFERENCE` o `UNRESOLVED`. Cada par elegible se clasifica como
interacción material/débil, aditiva o no resuelta. Los parámetros condicionales muestran soporte
activo e inactivo, y cada contrafactual generado se valida contra los predicados `when` declarados.

`confidence` es la estabilidad empírica determinista de esa conclusión exacta bajo remuestreo
acotado delete-d por candidato y resultados conscientes de seeds compartidas. Por eso puede crecer
para un resultado plano o contextual bien respaldado. Está separada de cobertura, magnitud del
efecto, fiabilidad y resolución Monte Carlo; no es un p-valor ni un intervalo frecuentista al 95 %.
El snapshot registra método, cantidad de remuestreos y evidencia ausente para poder auditarla.

Si se declara un margen de equivalencia, la región óptima práctica conserva candidatos cuyo regret
con incertidumbre es compatible y clasifica cada parámetro como restringido o flexible en ella. Sin
ese margen LambdaForge no inventa una tolerancia científica porcentual. `DESIGNED_PROBE` apunta a
una pregunta no resuelta y prefiere una comparación cercana; persiste preguntas, calidad del match,
objective predicho, reducción de incertidumbre, coste y alternativas descartadas. El alcance queda
limitado al pool determinista finito y preguntas predictivas por pares: no amplía fronteras del
dominio, no afirma causalidad ni sustituye la confirmación con seeds frescas.
Desde el checkout, `python -m benchmarks.hpo_scientific_design` ejecuta la regresión CPU
determinista de la suite: el selector antiguo solo de optimización y el nuevo diseño científico
reciben el mismo presupuesto en paisajes suave, de interacción y feature plana. Informa regret
simple y conclusiones resueltas; el límite de aceptación es la tolerancia de regret práctica, no la
afirmación de que una política gana en todo paisaje sintético.

### 7.1 Sweeps exactos sin HPO

`sweep` declara literalmente cada combinación finita por cada seed:

```yaml
seeds: [7, 17]
sweep:
  space:
    optimizer: [adamw, sgd]
    momentum: {values: [0.8, 0.9], when: {optimizer: sgd}}
  reference: {optimizer: adamw}
execution: {runs_per_gpu: auto, max_parallel: auto}
```

Son exactamente seis Runs: una variante AdamW y dos SGD por seed. No hay surrogate, pruning de HPO,
asignación adaptativa de seeds ni confirmación; la paciencia interna del entreno sí sigue activa.
Un rango numérico exige `points` para formar un grid finito determinista. `strategy: exhaustive` es
un alias compatible del mismo diseño y dispatcher. ARI conserva packing, recuperación OOM y
checkpoints. Un presupuesto de Runs menor que el diseño se rechaza; un límite de tiempo puede
producir un diseño honestamente incompleto.

El YAML se normaliza una vez a `StudyDesign`, `EvidencePlan`, `ExecutionPolicy` y la política de
objetivo. El presupuesto de candidatos solo detiene configuraciones nuevas; seeds mínimas,
compartidas, promociones, continuaciones y confirmaciones debidas siguen siendo elegibles. La
telemetría terminal expone `status`, `design_status`, `scientific_status` y `finish_reason`,
reconcilia la cola y distingue candidatos observados de completos.

El análisis compara seeds exactamente compartidas y remuestrea bloques enteros. Informa celdas
requeridas/completadas, media, mediana, dispersión, incertidumbre y soporte pareado; una referencia
añade diferencias, superioridad y, solo con margen práctico, equivalencia. Una celda ausente reduce
el soporte y nunca se imputa con otra seed. El líder puntual es descriptivo; la conclusión exacta
visible y su confianza siempre pertenecen a la misma hipótesis.

### 7.2 Entrenos independientes por GPU

`resources` es la reserva externa fija. `resources.gpu` indica cuántas GPUs reserva el estudio y
`execution.runs_per_gpu` cuántos procesos independientes pueden compartir cada una:

```yaml
seeds: [4, 7, 32, 54, 65, 94, 109, 124]
search:
  budget: {candidates: 40}
  replication: {minimum: 1}
  pruning: {enabled: true, min_step: 5, confirmations: 2,
            probability_threshold: 0.05}
  space:
    learning_rate: {range: [0.00001, 0.003], scale: log}
    hidden_dim: [64, 128, 256]
execution:
  runs_per_gpu: auto
  max_parallel: auto
  failure_retries: 1
objective: {metric: val_auprc, mode: max, practical_margin: 0.0}
resources:
  gpu: 2
  gpu_memory: 16GiB
  cpu: 16
  memory: 32GiB
  time: 24h
```

CPU, RAM y scratch usan un `HostResourceLease` calculado al despachar. Los residentes actuales se
reparten la asignación agregada del Job; cada admisión recalcula compromisos sin superar el techo
global/host duro. CPU es elástico: límites Torch/BLAS/OpenMP y affinity del árbol de procesos se
rebalancean cuando entran o salen residentes, evitando dar a tres Runs reales
`total_cpu / theoretical_max_parallel`. RAM/almacenamiento son compromisos para residentes actuales
contrastados con RSS vivo del árbol, no reservas cold-start para toda Run futura posible. Cada
dispatch persiste el lease exacto y su provenance.

El lanzamiento final reconcilia la cuota elástica nativa de RAM con una medición **nueva del árbol
completo de procesos**, incluidos auxiliares y workers existentes. Una cuota desactualizada puede
reducirse al margen restante de la asignación agregada; no se reduce silenciosamente un requisito
explícito independiente de una Run. Se mantienen los límites físicos/cgroup y del Job. La telemetría
incluye `launch_waits` y el motivo real de inactividad si falla la comprobación final: una propuesta
del planificador no demuestra que el worker haya arrancado.

La rampa inicial no es permanente: avances reales de paso/fase sin mayor consumo cierran esa
racha; sondear repetidamente el mismo estado no demuestra estabilidad. Un residente estable puede
usar la envolvente terminal aprendida compatible como compromiso, conservando explícita la cola
rara de consumo de toda la GPU en el riesgo conjunto. Sin historial válido no se inventa esa cota.
La comprobación final GPU revalida la misma decisión segura/exploratoria aprobada, con protección
de GPU hermanas y la comparación WAIT original; no reinicia un planificador aislado perdiendo
evidencia acumulada. Esto no garantiza que quepan todos los procesos máximos configurados.

Las métricas con `step` de un Work
personalizado publican `metric-progress.json`: una proyección pequeña por Attempt, actualizada una
vez por paso creciente. La admisión la lee sin recorrer el historial y conserva fases más precisas
de los callbacks. Un epoch reportado no demuestra estabilidad de memoria ni un checkpoint durable;
las métricas sin paso no inventan progreso.

Los eventos de capacidad física consultan la misma cola inicial/adaptativa que las finalizaciones.
Las obligaciones iniciales conservan prioridad y las semillas iniciales útiles pueden ocupar huecos
sin esperar otra finalización. Se mantienen identidades exactas candidato/semilla/fidelidad,
presupuestos de Runs/tiempo y controles de recursos/rendimiento. Esto no convierte una búsqueda
adaptativa en un barrido exhaustivo.

`auto` elimina topes artificiales globales/por dispositivo; ARI deriva un techo finito de acciones,
presupuesto de candidatos/Runs, GPUs asignadas, CPU host y reglas del sitio. Un entero conserva un
máximo duro: dos GPUs con `runs_per_gpu: 4` permiten como máximo ocho Runs y seis con valor 2,
doce. `gpu_memory` es un suelo de seguridad opcional. Si se declara, cada lanzamiento comprueba

$$
\texttt{gpu\_memory}
\leq \text{memoria de dispositivo libre en ese momento}.
$$

El modelo predice además para cada candidato una distribución de pico, envolvente futura, tiempo
hasta ella y duración restante a partir de historia compatible exacta o censurada. El compromiso
efectivo es `max(gpu_memory, envolvente superior aprendida, cota inferior OOM conocida)`; si se
omite `gpu_memory`, la predicción funciona automáticamente. La VRAM física es la autoridad.
El headroom admisible es el mínimo entre VRAM física libre y headroom futuro predicho; un proceso
externo que aparezca tras el lanzamiento bloquea inmediatamente una nueva admisión insegura.
Allocated/reserved de CUDA son diagnósticos y features, no sustitutos de la ocupación física. Las
trayectorias acotadas preservan el pico real y las transiciones; no hay un timeout fijo al pico.

Para la GPU \(g\), el planner razona con \(C_g-E_g-\sum_i M_{i,future}\): capacidad utilizable
menos uso externo dinámico y distribuciones futuras activas. Aquí
\(M_{i,future}(t)=m_i(t)+R_i(t)\), con residual no negativo e incierto. El residual es una
distribución empírica ponderada, no extremos equiprobables. Una prior débil de supervivencia de
Jeffreys se actualiza tras picos materiales y ciclos tranquilos; la dispersión robusta
física/allocator separa deriva y crecimiento real. Las fases críticas conservan riesgo tardío
hasta observarlas. Así, `RAMPING` explica el estado pero no veta absolutamente el placement.

El planner integra arrepentimiento de espera

$$
W_g(t_1)-W_g(t_0)=\int_{t_0}^{t_1} u_g(t)\,r_*(t)\,dt,
$$

donde \(u_g\) es la fracción físicamente utilizable y ociosa y \(r_*\) la mejor tasa de valor
científico pendiente normalizado. WAIT y EXPLORE se comparan hasta el siguiente step, fase,
checkpoint o final observado, no mediante timeout. Los GPU-segundos de rollback usan la misma tasa
y el valor informativo sigue separado de la prioridad. La duración desconocida usa historia o
ritmo vivo y, si no existen, sigue desconocida. Lightning puede atender selectivamente una
petición de checkpoint en un límite seguro de epoch cuando solo el rollback impide experimentar.

Cold start crea primero una `BASELINE_ADMISSION` en cada GPU concedida y ociosa, y después usa
trayectorias vivas de fase/step y checkpoints para experimentar 1→2→3 sin esperar picos terminales.
Una baseline es progreso protegido, nunca un experimento de packing sin validar ni un consumidor del
carril exploratorio. `SAFE_ADMISSION` usa envolventes respaldadas; `EXPLORATORY_ADMISSION` exige que
progreso e información esperados superen rollback e interferencia. GPU equivalentes mantienen un
carril a concurrencia baseline y no duplican el mismo escalón sin validar.
`RESOURCE_BLOCKED` es reversible; `RESOURCE_INFEASIBLE_ON_DEVICE_TYPE` exige una cota intrínseca
dura superior al dispositivo.

El controlador científico entrega una frontera ordenada y acotada. El planner usa best fit,
preserva GPU grandes para trabajo pesado y puede ejecutar backfill corto hasta la ventana predicha
de un candidato prioritario. La interferencia histórica puede frenar packing adicional cuando
reduce trabajo científico útil por tiempo. La escasez cambia cuándo/dónde se ejecuta, nunca el
objective científico. Si toda esa frontera acotada está bloqueada, la misma política científica
puede entregar más alternativas nuevas acotadas sin esperar una Run terminal. Las identidades
exactas comparten una frontera de espera dura —como máximo 16 y nunca por encima del paralelismo
restante tras las Runs activas—. Añadir una alternativa no concede recursivamente permiso para
proponer otra; una respuesta sin identidades nuevas cierra ese estado del dispatcher.

LambdaForge sondea las GPU concedidas durante toda Run activa, aunque no haya una acción en cola, y
ocupa solo capacidad segura; el resto espera. Los lanzamientos se escalonan para observar la
asignación anterior. `gpu_memory` no se multiplica por
Runs activos ni crea otra reserva oculta. Las GPU elegibles se ordenan por Runs activos y tiempo desde su último lanzamiento.
Así, cuando solo queda un slot global, el índice cero no gana siempre ni deja sin trabajo otro
dispositivo concedido. Mientras queden presupuestos de candidatos/Runs/tiempo, cada evento terminal
replantea inmediatamente y aporta una acción de reemplazo; la confianza baja del análisis posterior
no es una condición implícita de parada. Cada Run admitido tiene un proceso spawn nuevo con un único worker, que se
cierra al recibir resultado o error y libera el contexto CUDA completo. Un pool ocioso persistente
retendría VRAM y podría bloquear la cola. Una GPU llena o pequeña se omite mientras las demás
continúan. Un suelo declarado, o una cota inferior OOM dura persistida, superior a todo dispositivo
compatible asignado se marca explícitamente inviable en vez de esperar para siempre. Las
consultas de memoria se ejecutan en un hijo efímero que termina tras devolver JSON, por lo que el
controlador HPO no aparece como un proceso CUDA ocioso por GPU ni consume un slot de
`runs_per_gpu`. Las esperas/admisiones se escriben inmediatamente en el log del Work.

Es admisión preventiva, no un límite duro dentro del código consumidor. Un `gpu_memory` explícito
sirve si se conoce un mínimo no negociable; el modo automático es apropiado cuando el search cambia
mucho la demanda. `max_parallel` conserva el máximo global duro y limita estudios solo CPU.

LambdaForge atribuye VRAM por PID/árbol de procesos mediante NVML y reconcilia uso externo en cada
muestra. Heartbeats del allocator añaden fase, checkpoint y picos breves; el fallback agregado queda
censurado. Un worker vivo ausente de un inventario de procesos completo y reciente tiene uso CUDA
observado cero durante inicialización CPU. Una observación incompleta/desconocida, bytes por PID
desconocidos, PID reutilizado o allocator positivo contradictorio nunca se convierten en cero.
La reserva del driver y memoria ajena no se atribuyen a esa Run; la VRAM físicamente libre sigue
mandando. Uso cero mientras carga datos no valida un futuro consumo de entrenamiento bajo ni
elimina la protección de la vía de progreso. Una inicialización CPU larga del Work puede mantener
la GPU sin actividad: revisa el log antes de concluir que la admisión GPU está bloqueada.
La evidencia OOM separa memoria intrínseca de la firma del packing fallido: por ello
`heavy+heavy` no reduce un techo global ni prohíbe `heavy+small`.
La terminación científica y la completitud de recursos son distintas. Las fases relevantes se
aprenden de Runs compatibles; no existe una lista universal
forward/backward/optimizador/validación. Una Run podada cuyas fases cubren el riesgo de asignación
aprendido y cuya telemetría es exacta aporta un perfil exacto aunque su objective siga censurado.
Un pico tardío histórico de validation/checkpoint conserva el perfil actual como censurado hasta
observarlo o representarlo en la cola. `exploration-evaluations.jsonl` conserva comparaciones con
P(fit), hazard, wait regret y motivos como `RUN_CAP`, `HARD_LOWER_BOUND`,
`KNOWN_FAILED_PLACEMENT`, `PROTECTED_LANE`, `THROUGHPUT_REGRESSION`, `ROLLBACK_DOMINATES` y
`WAIT_CURRENTLY_BETTER`; se suprimen sondeos repetidos.
Un snapshot activo acotado y atómico sobrevive a una interrupción del controlador como conocimiento
provisional anterior. Puede informar la incertidumbre tras reiniciar, pero nunca se restaura como
proceso vivo o pico exacto; la evidencia terminal lo sustituye y el cierre normal lo vacía.

La telemetría de recursos materializa recursivamente copias JSON nativas de mappings/listas antes
de serializar, incluidos parámetros Work inmutables anidados. No modifica parámetros, identidades
ni contenedores originales. Evidencia activa/terminal, admisión, ledgers y trazas comparten ese
límite de serialización. Valores no JSON no admitidos siguen fallando explícitamente; no se ocultan
errores de persistencia ni se desactiva la monitorización segura de GPU.

Tras corregir serialización del framework, actualizar solo el controlador no parchea el runtime
remoto gestionado. Instala/usa LambdaForge corregido en el entorno local del consumidor y, desde
ese proyecto, ejecuta explícitamente `lf clusters bootstrap CLUSTER --project .` (puedes revisar
antes con `--dry-run`). La identidad del entorno usa los bytes exactos del wheel, no solo la versión
visible: se verifica un prefijo nuevo sin editar entornos inmutables antiguos. Preparar retry también
empaqueta el framework actual. Revisa `lf retry JOB_ID --dry-run --json` antes de `lf retry JOB_ID`;
no envíes un Work nuevo para recuperar el Study. Recovery reutiliza resultados completos autoritativos
y procesa solo Runs lógicas elegibles interrumpidas/fallidas/pendientes. Un log de evaluación final
o número de epoch no prueba éxito. Continuar desde checkpoint requiere soporte del consumidor;
si no existe, se reinicia la Run afectada sin repetir hermanas completadas. No edites registros ni
checkpoints originales ni desactives identidad científica para esquivar un fallo de telemetría.

ARI v3.1 separa la trayectoria visual de `ResourceTrajectoryStatistics`. La primera sigue acotada;
la segunda conserva incrementalmente ciclos de progreso/optimizer, visitas y máximos por fase,
picos materiales, sketches robustos de crecimiento/ruido/residuales y cadencia de checkpoints. El
hazard se actualiza con progreso y asignación, no con cantidad de sondeos, así que muestras
duplicadas no fabrican confianza. La convolución ponderada acotada conserva colas discretas raras en
P(fit), con un ensemble explícito independiente/epistémico compartido para varias Runs. Placements
posteriores calibran la probabilidad sin contaminar el objetivo científico.

Las mezclas posteriores activas y entradas de convolución se acotan a 128 puntos ponderados;
es un límite interno de representación/coste, no un umbral de concurrencia o riesgo. Los soportes
pequeños siguen siendo exactos. En los grandes se conservan las masas de los extremos y se agrupan
intervalos interiores hacia arriba: comprimir no aumenta la probabilidad de que una admisión quepa.
Las actualizaciones conservan los pesos de residuales históricos, sin convertir colas raras en
muestras equiprobables. El ensemble de incertidumbre compartida recorre eventos de la CDF una
sola vez, sin ordenar en cada cuantil. Así un Study largo no puede hacer crecer indefinidamente
el posterior de recursos y bloquear recogida de Futures, cierre de workers, telemetría o admisión.
Memoria física libre, cotas OOM y permisos GPU siguen siendo la autoridad. Los runtimes inmutables
ya en ejecución no cargan esta corrección en caliente.

WaitRegret es una integral atómica versionada. Un cambio de frontera cierra el tramo anterior con
su tasa y fracción ociosa previas; no borra la oportunidad perdida. Solo se reinicia si desaparece
la oportunidad factible o la capacidad utilizable, o una admisión la satisface. Cotas físicas,
packings dominados, run caps, política del sitio y regresión de throughput siguen siendo absolutas.
Checkpoint calcula rollback por residente, cobra la duración aprendida y espera evidencia durable
antes de tratarlo como reducido. El horizonte usa tiempo *hasta* el próximo evento y cadencia
aprendida, nunca edad desde el checkpoint anterior. Los intervalos repetidos aportan una
incertidumbre robusta de cadencia; sin evidencia suficiente queda explícitamente desconocida.

El aislamiento sigue el límite de Run. Tanto en CPU como GPU cada Run adaptativo usa un proceso
nuevo de un worker: matar uno no rompe un pool compartido ni cancela candidatos ajenos. Un worker
perdido antes de devolver resultado y una OOM/asignación CUDA se reintentan como Attempt nuevo hasta
`failure_retries` (1 por defecto, rango 0–3). Una OOM exploratoria entra en `RESOURCE_RECOVERY`,
conservando checkpoints compatibles bajo el mismo
Run. Una OOM CUDA registra asignación intentada (si aparece), headroom físico, residencia del
candidato, uso externo y co-runners como cota inferior censurada, y actualiza las predicciones
pendientes. El reintento no se admite hasta que mejore materialmente el headroom u otra condición
compatible. Nunca se matan hijos sanos. Si el impedimento se repite con packing más
seguro queda terminal; las excepciones normales de datos o código no se repiten a ciegas. Los demás
Runs activos o pendientes continúan. El Work exterior sigue quedando
fallido si un Run agota recuperación, para no ocultar evidencia científica incompleta.

La evidencia queda bajo `hpo-control/resources/`: observaciones acotadas, evidencia OOM, decisiones
de admisión y ledger actual. Una firma Work/código/entorno/hardware permite warm-start entre Studies
compatibles; lo incompatible no se convierte en prior exacto. Resources en la Consola muestra el
mismo modelo de lectura: memoria física libre, uso externo, compromiso actual/futuro, headroom,
P(fit), bloqueo y backfill. El análisis final registra muestreo condicionado por recursos para no
describir como mala una región pesada simplemente infra-muestreada.

#### Replay del scheduler de recursos

Cada estado cambiado queda en una traza acotada y versionada con dispositivos, frontera científica,
predicciones, admisiones/bloqueos y evaluación exploratoria. Reutiliza observaciones existentes y
no copia outputs ni checkpoints:

```bash
lf results replay EXECUTION --policy recorded
lf results replay EXECUTION --policy ari-v3.1 --json
lf results replay EXECUTION --policy ari-v3-compat --json
lf results replay EXECUTION --policy ari-v2-compat --json
python -m benchmarks.resource_scheduling --execution .lambdaforge/runs/WORK/EXECUTION
```

Solo es exacto mientras la política coincida con las admisiones registradas. La primera diferencia
es `counterfactual_divergence`; lo posterior se etiqueta como simulación condicionada por la traza,
nunca como hecho. Un resultado científico terminal u OOM de la rama real pasa a `null` en una rama
alternativa en vez de reasignarse falsamente. Las políticas compatibles solo existen en replay. Las métricas cubren llegada a
concurrencia, packing medio, VRAM/cómputo ocioso medible, tasas científica/útil y throughput, Runs
completas/podadas, tipos de OOM, rollback/checkpoint, starvation, bloqueo, falsos seguros/esperas,
calibración de P(fit) y errores de predicción. Evidencia ausente da `null`. El benchmark sin
argumentos es solo regresión sintética; `--execution` usa evidencia cuantitativa de una traza real.

#### Geometría canónica de parámetros

`ParameterSpace` es la autoridad matemática del search authored: encoding/decoding transformado,
distancia, máscara de actividad, bounds, categorías, condicionales, soporte y posición en frontera.
Los parámetros log se normalizan en espacio log; los integer mantienen topología ordinal; los
categorical son discretos; un condicional inactivo usa un bit de actividad y no finge ser el valor
mínimo. Sobol, selección adaptive/Bayesian, similitud de recursos, diseño científico,
coverage/effects y Study Analysis consumen la misma geometría. Los candidatos persistidos y sus
fingerprints no cambian.

### 7.3 Continuación multi-fidelidad

`search.fidelity` es opcional porque solo el Work consumidor sabe si su presupuesto es acumulativo
y reanudable; LambdaForge nunca lo infiere de un parámetro llamado `epochs`:

```yaml
search:
  trials: 40
  fidelity: {min: 5, max: 100, reduction_factor: 3}
  learning_rate: {range: [0.00001, 0.003], scale: log}
objective: {metric: val_auprc, mode: max}
```

El primer Attempt recibe `self.fidelity.current == 0` y `target == 5`. Si se promociona conserva la
misma identidad científica y raíz de checkpoints, con targets acumulativos 15, 45 y 100. El Work
debe avanzar solo de `current` a `target`, guardar estado con `self.checkpoints` y registrar la
métrica; no debe repetir desde cero todo el target. La confirmación siempre usa `maximum`.
`LightningRunner` limita `max_epochs`, guarda un `last.ckpt` gestionado y lo entrega al Attempt
siguiente; un trainer propio debe implementar expresamente el mismo contrato. Sin `fidelity` se
mantiene el entreno normal a presupuesto completo.

### 7.4 Contrato de early stopping

El stop es cooperativo. Un Work registra `self.metrics.log("val_auprc", valor, step=epoch)` y
consulta `self.stop_requested` en un límite seguro, guardando checkpoint antes de retornar. El
controlador compara Runs a un step común tras `search.pruning.min_step`. Una proyección lineal local
acotada conserva incertidumbre residual. El pruning no se activa hasta que al menos dos candidatos
distintos hayan terminado con suficiente curva para contrastar predicciones de prefijos contra sus
endpoints reales y obtener un error de curva finito. Dos es el mínimo que identifica una
comparación, no un umbral configurable; antes todas las curvas iniciales pueden crear la referencia.
Una vez calibrado, solicita parar solo cuando la probabilidad de seguir a
`objective.practical_margin` del incumbent cae bajo
`search.pruning.probability_threshold`; no elimina una fracción
fija. Por defecto la condición debe mantenerse en dos steps comunes distintos
(`confirmations: 2`): sondear repetidamente la misma época no suma evidencia. Usa
`confirmations: 1` solo para una política deliberadamente agresiva. `min_step` solo marca el mínimo
de evidencia: jamás se reutiliza como distancia de predicción. Si hay rung de fidelidad declarado,
ese rung es el horizonte; en otro caso se proyecta una única ventana reciente ya observada, y un
candidato aún competitivo en el checkpoint común exacto no se detiene solo por una pendiente local
peor. Así se evitan extrapolaciones largas sin soporte. `LightningRunner` enlaza
`callback_metrics[objective]` y para en límites de batch. Un trainer
propio usa la API pública anterior. Si solo hay métrica final se adaptan las seeds, pero no puede
detenerse de forma segura el entreno actual.

### 7.5 Observabilidad viva del estudio

El Job externo del scheduler delimita reserva y cancelación, pero no obliga a mezclar el significado
científico. LambdaForge mantiene junto al Job un índice compacto del estudio y un registro pequeño
por Run interno. Los `work.log`, `metrics.jsonl`, `training-metrics.jsonl` y `result.json` ya
existentes siguen siendo la autoridad; el índice los referencia y resume solo estado y últimos
escalares. Nunca copia checkpoints de modelos, directorios de output ni bytes de artefactos.
Las lecturas remotas siguen la vista visible, no el tamaño de la Execution:

- Overview y la lista de Studies reciben solo contadores, objetivo/líder y metadatos básicos de
  estado/coste de `study/overview.json`. No descargan listas de candidatos ni seeds.
- Abrir un Study lee `study/interactive.json` (proyección v3): celdas de la tabla de Trials, estados
  agregados de Runs y parámetros de un único líder. No incluye seeds, curvas ni estado HPO.
- Abrir un Trial lee su `study/trials/trial-XXXXX.json`: parámetros y tabla de sus seeds. Buscar
  parámetros filtra en el host de ejecución, sin descargar todos los parámetros.
- Abrir una seed lee solo su registro nativo, curvas acotadas, metadatos de artefactos y log.
- HPO lee `study/hpo.json` precomputado; Resources lee `study/resources.json`. El historial completo
  se pagina únicamente al abrir explícitamente Action history.
- Analysis lee los agregados de `study/analysis-panel.json`, sin evidencia individual por Run ni
  catálogo de investigación HTML. El worker persiste el análisis final. El post-hoc provisional se
  calcula/cachea en el host de ejecución solo cuando se solicita, en un proceso de lectura separado,
  nunca en el heartbeat de planificación. HPO no reajusta este análisis. Exportar HTML solicita
  explícitamente el análisis completo, no sustituye sus datos por el modelo ligero de la consola.

Los resúmenes/índices antiguos ricos se proyectan en su host, nunca se descargan enteros. Todas las
vistas de lectura de Study usan páginas de 512 KiB con comprobación de generación; un refresco sin
cambios devuelve solo un marcador pequeño. Se conserva la seguridad por respuesta, pero no se
rechaza un Study grande válido por un límite agregado de 8 MiB. Las vistas padre ocultas dejan de
sondear hasta volver a ser visibles. Son cambios de lectura/presentación, no de decisiones HPO ni
de evidencia científica.
Analysis de sweeps usa el espacio del StudyDesign persistido, incluidas ramas condicionales aún
sin completar; no reconstruye condiciones a partir de filas observadas. Si faltan contextos válidos
de respuesta, indica evidencia insuficiente, no un fallo del Study en ejecución. El proceso de
lectura pasa esa geometría explícitamente a workers antiguos e inmutables: actualizar el controlador
no requiere reiniciar el Study para aplicar esta corrección de lectura.

la Consola de investigación presenta esta jerarquía:

```text
Work
  └─ Trial 17: {dropout: 0.21, hidden_dim: 128, ...}
       ├─ Seed 4: ejecutando → curvas, tiempos y log vivo aislado
       └─ Seed 7: terminada  → curvas, tiempos y log aislado
```

La vista separa candidatos propuestos del presupuesto planificado y muestra Runs
activos/terminados/fallidos/podados y estado de promoción. La tabla separa mejor objective, su
seed/época y objective actual; HPO usa aparte la media de los mejores checkpoints de las seeds
terminadas. Debajo aparece cada parámetro del Trial seleccionado. La tabla de seeds incluye última
época, mejor época, mejor objective, objective actual e índice GPU, y termina con un preview acotado
de métricas; el panel completo queda a un Enter y se reservan al menos tres filas de seed con una
altura de terminal normal. `pruned` indica que el Run aceptó una solicitud probabilística terminal;
no es un fallo, no queda pausado ni se reanuda después, y se muestra su motivo persistido. Un Trial
con solo seeds podadas queda `pruned`, no `failed`. Sus curvas parciales se conservan como evidencia
censurada, pero no entran como valores exactos en medias de seeds completas, ajuste del surrogate
ni estadísticas marginales del objective: tratar su mejor punto parcial como presupuesto completo
sesgaría la búsqueda. Si cualquier seed recibe pruning de candidato, el candidato entero queda
censurado: las seeds terminadas antes no se promedian como un objective solo de supervivientes.
LambdaForge sí conserva la información: el modelo de lectura HPO muestra la
tasa terminal de poda por región. Los resultados completado/podado por candidato alimentan un
modelo conjunto de supervivencia con intervalo suavizado de incertidumbre; varias seeds no
sobrerrepresentan un candidato y fallos de infraestructura/recursos o preemption son neutrales. Es
evidencia negativa censurada, no un objetivo escalar inventado. El
detalle JSON conserva además el token heredado exacto para grants UUID/MIG.

El controlador también reejecuta retrospectivamente la política de pruning sobre curvas completas
y escribe el informe acotado en `hpo-control/state.json` como `pruner_calibration`. Conserva los
parámetros de política, podas simuladas, Runs/epochs/GPU-seconds que se habrían ahorrado, tasa de
falsos prunes, regret, Brier score de probabilidad, RMSE de curva y cobertura aproximada del
intervalo al 90 %. Un estudio pequeño devuelve incertidumbres `null`, no certeza inventada. Es una
auditoría retrospectiva: no altera objetivos ni inventa el contrafactual de un Run realmente
podado.

la Consola de investigación consume esa misma evidencia del controlador, no ajusta otro modelo HPO. La vista de
candidato muestra media/error estándar de la utilidad compuesta, agregación, componentes raw y
normalizados del mismo checkpoint, pesos/contribuciones, constraints y pertenencia Pareto
diagnóstica. El panel HPO muestra slots activos/disponibles, Runs en cola/pausados y score,
proxy de valor del controlador y coste de la última acción. Un Run podado despliega candidato, step/fidelity,
incertidumbre de predicción, incumbent, probabilidad, umbral, margen, confirmaciones y modelo de
curva persistidos.

Abre la pestaña **HPO** del Study y selecciona un parámetro con Enter/derecha para inspeccionarlo.
Para cada hiperparámetro el resumen separa dominio declarado y observado, región respaldada,
importancia y fiabilidad. El detalle combina respuesta e incertidumbre numérica o medias
categóricas, dispersión empírica, cobertura y ganancia predictiva por pares. Etiqueta evidencia
baja/media/alta con suelos de muestra conservadores y un score de
asociación, y sugiere la
comparación que más reduciría ambigüedad. La cabecera separa la última acción real del controlador.
También separa observaciones comparables, provisionales y podas censuradas. Las asociaciones
marginales pueden estar confundidas por parámetros correlacionados, activación
condicional, distinta cantidad de seeds o fidelidad parcial, mientras el sampler opera en el
espacio mixto multivariable. Por eso dice «parece asociado», nunca causal ni garantizado. Métodos
de importancia de interacciones como functional ANOVA pueden resumir un surrogate maduro
([artículo fANOVA](https://proceedings.mlr.press/v32/hutter14.html)), pero aplicarlos a los primeros
Trials produciría falsa precisión inestable. Por eso las explicaciones tempranas siguen siendo
marginales y auditables mientras el controlador GP/k-NN real sí permanece multivariable. La
cabecera separa `MARGINAL / PAIRWISE DIAGNOSTICS` (descriptivo/no causal) de `SURROGATE BELIEF`, el
snapshot ligero del sampler real: backend, fidelidad objetivo, pendientes, candidato/región predicha
e incertidumbre. Se conserva fuera de la cola de 25 eventos y nunca serializa el modelo.
Enter/derecha sobre el parámetro seleccionado abre una curva numérica agrupada o barras de medias
categóricas y una tabla de calor de relaciones por pares. Cada valor es la ganancia predictiva
leave-one-out de k-NN conjunto frente al mejor predictor marginal, normalizada por la desviación del
objective: indica ayuda predictiva conjunta, no interacción causal. Los puntos acotados y una matriz
máxima 12×12 permiten reproducir la vista en wrappers. La respuesta descriptiva y la cobertura
conjunta empiezan con dos observaciones; la ganancia predictiva empieza con tres, manteniendo suelos
de muestra conservadores para la confianza. Un observador nuevo reconstruye localmente un snapshot
acotado antiguo a partir de la telemetría de candidatos, sin reiniciar un estudio remoto activo. La
consola también muestra restricciones y
candidatos no factibles. El mismo JSON está en `overview` →
`work.items[].study.hpo_analysis`, las acciones recientes en `.controller` (25 eventos) y el
sampler real en `.surrogate_belief`.

La finalización científica es distinta de estabilizar el screening. Un screening estable detiene
propuestas `OPTIMIZE` ordinarias; después pueden continuar confirmation y la cobertura material de
respuestas, parámetros e interacciones hasta agotar las acciones factibles con valor material o un
presupuesto duro. `remaining_information_value` combina incertidumbre predictiva y deuda explícita
de soporte: un surrogate seguro no convierte un valor discreto sin observar en evidencia real. El
`scientific_status` compartido incluye preguntas de parámetros y deuda de interacciones material y
factible.

Con un objective escalar, los Runs completados también pueden exponer `diagnostic_metrics`:
métricas terminales numéricas comunes a las seeds comparables, con media, dispersión y soporte. Se
pueden inspeccionar en Trial y en el dashboard offline, incluso frente a parámetros. Son evidencia
solo diagnóstica: nunca se convierten en componentes del objective, constraints ni entradas HPO.

F1, balanced accuracy, kappa de Cohen, accuracy, precision y recall dependen potencialmente de un
threshold. LambdaForge usa exactamente el valor registrado: no optimiza ese threshold, no elige uno
por candidato ni las equipara a AUROC/AUPRC. Esa política pertenece al protocolo del Work.

La vista de Run se divide en dos regiones. Arriba aparecen la lista completa y alineada de
parámetros, el resumen vivo de duración/tiempos y hasta cuatro curvas. Los botones compactos
anterior/siguiente y un selector numerado permiten usar ratón; `n` y `p` siguen como atajos. La
tabla por época pone cada escalar registrado en su propia columna y se desplaza horizontalmente.
Arriba/abajo selecciona, cada curva aplicable la marca en rojo, la época óptima permanece con
marcador verde y fila destacada, y Enter/derecha abre la misma evidencia en vertical cuando resulte
más legible. La duración se calcula desde
`started_at_utc` durante la ejecución. Los `epoch_time_s` y `validation_time_s` medidos sustituyen
la estimación media por época, etiquetada como aproximada, cuando llega la telemetría. El detalle se
refresca fuera del loop de terminal.

Las curvas se leen con límite de bytes y se reducen a 10–500 puntos representativos conservando
extremos y la época óptima exacta; el TUI pide 80. El log también está acotado durante el refresco y observar nunca modifica
la evidencia.

La Consola Textual consume estas curvas acotadas sin modificar su evidencia. La presentación rica
es opcional para la ejecución científica y la salida máquina sigue siendo JSON plano. Los detalles
de recursos usan los mismos snapshots y motivos de admisión persistidos.

Recogida y presentación son independientes. `LightningRunner` conserva todos los escalares finitos
acotados, pero un proyecto selecciona con patrones shell las curvas que dibuja la Consola de investigación:

```python
from lambdaforge.training import LightningTrainConfig

training = LightningTrainConfig(
    epoch_console_include=["train_loss", "val_*", "*_time_s"],
    epoch_chart_include=["val_*", "epoch_time_s", "validation_time_s"],
    epoch_chart_exclude=["*_aux"],
    epoch_metric_display_names={"val_balanced_accuracy": "Exactitud equilibrada"},
)
```

`epoch_console_include`/`epoch_console_exclude` seleccionan la tabla humana por época; los patrones
de gráfica eligen independientemente solo la presentación TUI. La selección del CSV sigue disponible
mediante `epoch_metrics_include`/`epoch_metrics_exclude`. `epoch_metric_display_names` cambia solo
etiquetas visibles de la Consola; no altera la identidad de métrica, el lookup del objective ni los
valores almacenados. Los nombres habituales de validación, entreno y recursos se humanizan solos.

El orden automático prioriza objective, `epoch_time_s`, `validation_time_s`, memoria, validación y
entreno. Los nombres CUDA son precisos: `gpu_mem_mb` es el pico de tensores vivos
(`max_memory_allocated`); `gpu_reserved_mb` es el pool actual de caché del allocator PyTorch;
`gpu_peak_reserved_mb` es su pico en la época. La memoria reservada incluye asignaciones vivas y
bloques cacheados reutilizables, por lo que puede ser mucho mayor sin ser otra reserva del scheduler.
El packing adaptativo nunca deduce concurrencia de ese dato: usa memoria física del driver, la
envolvente aprendida por candidato y el suelo opcional `resources.gpu_memory`, llenando capacidad
segura sin exigir el máximo de antemano. Tras cada Run empaquetado, termina su proceso dedicado y libera el contexto
CUDA completo antes de readmitir el slot.

No existe un tipo de ejecución específico para entreno. La validación local marca como estudio de
parámetros cualquier Work ordinario con `search` o varias `seeds`. Por ello su pantalla está
disponible durante `preparing` y explica que la telemetría seguirá pendiente hasta que el worker
actual cree el índice acotado. Los clientes máquina reciben la misma declaración en
`work.items[].study_expected`; `study` permanece `null` hasta que exista evidencia viva. Un Run
lanzado con un worker antiguo no puede generar retroactivamente telemetría por Run y necesita una
nueva ejecución con el runtime remoto actualizado.
El workspace de Study conserva la cancelación mientras `study` siga siendo `null`; usa la identidad
exacta del Work semántico y detiene todos sus Attempts activos sin depender de telemetría HPO. Los
fallos de refresco conservan el último snapshot correcto y se muestran inline sin notificaciones
emergentes repetitivas.
Varios pasos de workflow, un nivel paralelo y la concurrencia de `self.map()` no forman por sí
mismos un estudio de parámetros: esos Works conservan la navegación Attempt/log y no crean índice
de estudio. Esta inferencia semántica evita un selector de presentación en YAML que podría
contradecir la ejecución real.

`LightningRunner` registra automáticamente los escalares finitos de callbacks tras validar.
`EpochStats` aporta la duración de época y el bridge mide la validación. Otro trainer usa las
métricas genéricas, sin API especial:

```python
self.metrics.log("train_loss", train_loss, step=epoch)
self.metrics.log("val_auprc", val_auprc, step=epoch)
self.progress.update(epoch, epochs, message="entrenando")
self.log(f"época {epoch} terminada")  # narración opcional con flush inmediato
```

`print()`, logging estándar y `self.log()` quedan capturados en el `work.log` del Run seleccionado,
de modo que el drill-down es legible incluso con 24 hijos activos. `metrics.log` es evidencia
numérica estructurada y alimenta curvas/objective; `progress.update` expresa progreso grueso; el log
explica a humanos. No se deben codificar curvas imprimiéndolas.

Los consumidores máquina usan el mismo modelo de lectura sin parsear el TUI:

```bash
lf overview --json
lf show WORK --json
lf show WORK --run trial-00017-seed-4 --json
lf logs WORK --run trial-00017-seed-4 --tail 300
```

`work.items[].study` solo contiene contadores compactos, objetivo y metadatos del líder. Abre un
Study y Trial en la consola para obtener sus claves de Run sin descargar todos. `show --run` devuelve parámetros,
últimos valores, curvas reducidas, log acotado, fallo, rutas de evidencia y el inventario de
artefactos gestionados ya finalizados. La salida humana destaca la ruta utilizable preferida; el
JSON conserva checksum, tamaño, rol, tipo MIME, ubicaciones gestionada/publicada, retención y
metadatos. La Consola de investigación muestra el mismo inventario en **Artifacts** de la seed. La
inspección solo lee metadatos: no copia ni abre ficheros remotos. Los artefactos no aparecen antes
de finalizar correctamente los outputs; `--curve-points N` elige
10–500 puntos. `logs --run` emite solo ese log. Una interfaz headless debe consultar el JSON;
`--follow` queda para logs externos porque la Consola de investigación ya refresca cada Run de forma segura.

## 8. Ejecución, identidad y reutilización

La identidad científica incluye import del Work, identidad del código consumidor, argumentos,
hashes de ficheros, content IDs de datasets, seed y parámetros de variante. Clúster, rutas, tiempo,
Job y Attempt son procedencia operacional y no cambian la definición.

El plan no crea estado. Una ejecución normal reutiliza un éxito verificado. Un Run fallido puede
tener otro Attempt usando el snapshot YAML enviado aunque el original cambie; checkpoints
compatibles activan `resuming`. `--restart` elimina checkpoints del Run. `--rerun` crea otra
Execution deliberada. El control plane rechaza un duplicado activo de misma identidad/destino salvo
`--allow-duplicate`.

### Recuperar un Study interrumpido

`lf retry STUDY` y **Resume Study…** en su ventana usan el mismo servicio de recuperación.
El selector identifica un Attempt fallido/cancelado/con timeout del clúster original. Se consulta
un plan acotado en ese host, se prepara el código actual en otro Job y el runner se reconecta a la
Execution original bajo propiedad del framework. No se copian pesos ni artefactos pesados para
reintentar: solo índices pequeños de consola y referencias a decisiones. Logs, métricas,
checkpoints y evidencia siguen referenciados en su ubicación original.

La recuperación resuelve seeds en el namespace original persistido, no en la carpeta del nuevo Job.
Conserva valores/orden explícitos y coordenadas del stream automático; rechaza coordenadas ausentes
o corruptas y cambios de política. En consola, Logs y Resume Study apuntan al último Job. Hasta que
publique telemetría, la evidencia propia anterior de Trials/Runs sigue visible sin sustituir su
estado actual. No requiere modificar el YAML ni saltarse las comprobaciones de identidad.

La operación cubre Studies adaptativos, seeds repetidas y sweeps fijos. Los diseños fijos leen
el diseño inmutable y Attempts propios, consideran el último resultado y reutilizan Runs correctas;
no requieren ni inventan estado de controlador adaptativo. `lf retry STUDY --dry-run --json`
expone `recovery_plan`: `reuse_runs`, `retry_runs`, `pending_runs`, seed/Trial/acción por celda
y presupuestos físicos gastados. Una seed fallida recibe otro Attempt de la misma Run lógica,
no un experimento nuevo; las Runs sin empezar siguen siendo obligatorias. Recuperar el Study no
garantiza continuar una época: se ofrecen checkpoints compatibles, pero el Work debe restaurarlos.
Sin ese soporte, solo se reinicia la Run incompleta. Diseños/entradas incompatibles y checkpoints
con rutas inseguras se rechazan.

Para Studies adaptativos se restauran candidatos propuestos, números de Trial, expansión determinista, seeds, acciones
pendientes, obligaciones del diseño inicial, convergencia y confirmación. Resultados completos y
podados conservan su evidencia. Runs fallidas y acciones interrumpidas se reencolan una vez como
nuevos Attempts: usan un checkpoint compatible cuando existe y empiezan de nuevo si no existe.
Un resultado publicado justo antes de caer el controlador se recoge, no se repite. Los fallos
físicos quedan en el historial y los presupuestos gastados de Runs/tiempo no se reinician.
El estado agregado considera el último Attempt de cada Run lógica: recuperar un fallo permite
éxito, mientras un fallo que persiste sigue siendo un fallo real.

`lf retry STUDY --accept-code-change` permite reconocer explícitamente una corrección compatible:
el investigador debe saber que métricas anteriores y formatos de checkpoints siguen siendo válidos.
La consola ofrece la misma casilla, desmarcada por defecto. Si la previsualización detecta cambios
de código, rechaza el envío antes de crear un Job salvo reconocimiento explícito; Resume queda
deshabilitado hasta marcar la casilla. El worker recién preparado vuelve a comprobar compatibilidad.
Esto no omite comprobaciones de
configuración científica, contenido de entradas, objetivo, streams de seeds ni política de búsqueda.
Si la corrección invalida resultados previos, crea otro Study. `execution.json` permanece inmutable;
`current-code.json`, procedencia de cada Run y `recovery-history.jsonl` muestran las revisiones reales.

Un bloqueo entre procesos impide escribir simultáneamente el mismo controlador. Las dependencias
de recuperación protegen los Jobs originales de GC y borrado individual; borrar todo el historial
terminal del Work previsualiza y elimina la cadena junta. Estado ausente/corrupto y rutas ajenas o
con enlaces fallan de forma segura. Los adaptativos siguen requiriendo `hpo-control/state.json`
compatible; los diseños fijos requieren sus registros propios de diseño/Attempts. La recuperación
cubre un único Study por Execution
en el mismo clúster; un fallo de preparación sin estado de Study permite el retry normal del envío.
Los reintentos automáticos siguen limitados a infraestructura: errores consumidores solo vuelven
a intentarse por esta acción deliberada del usuario, nunca en un bucle automático infinito.

## 9. Resultados y metadata

Telemetría y resultados de Execution incluyen `lifecycle` versionado con `operational`, `evidence`
y `health` independientes. `running / pending_recovery / degraded` no significa fallo científico
terminal. `stopped / complete / degraded` conserva fallos físicos anteriores tras recovery. Retries
en espera son cola, no procesos activos. El historial visible está acotado pero conserva contadores
acumulados; historias antiguas truncadas se señalan como incompletas. Los envelopes antiguos se
pueden proyectar en memoria sin crear ficheros ni inventar Attempts ausentes. El
[registro de reforma](ai/IMPLEMENTATION.es.md#history-architectural-reform) detalla fronteras implementadas y migraciones
pendientes. No cambian ajuste de objetivos, censura de pruning, identidad exacta ni presupuestos.
La finalización resuelve los últimos Attempts de cada celda lógica antes de calcular el estado
terminal compatible: un fallo físico antiguo no invalida una Run recuperada. Fases y fidelidades
permanecen separadas. Previews de storage/clean no crean cachés ni locks; apply revalida propiedad
y objetivos bajo locks, no confía en un preview anterior.

[Productos durables](#guide-products) documenta publicación `lf products`, selección ModelSet,
StudyDecision, bundles portables verificados e inputs product tipados en `with`. Resolver metadata
es de solo lectura; Attempts reales registran consumidores y pesos se verifican al acceder
explícitamente. Catálogos propios del proyecto, no de Jobs/cachés desechables. Espera/replanificación
automática de dependencias y export Fleet completo todavía no están implementados.
`lf import PACKAGE` verifica exports de un host; `--apply` registra evidencia y productos sin ejecutar.
Procedencia original intacta; registros importados son read-only, no recovery nativo. Consola
**Studies → Import Study…** (también en Products) ofrece el mismo flujo verificado y confirmado,
con explorador de carpetas. El import aparece en Studies y Overview identificado como snapshot
importado/read-only con su estado capturado. Trials/seeds leen métricas, logs y artifacts locales
conservados; Analysis lee evidencia incluida sin recalcularla. No muestra cancel/retry remoto.
Reaplica un import antiguo para añadir sus índices visuales locales. El refresco raíz nunca vuelve
a analizar el resultado agregado. El `result.json` agregado verificado no tiene un límite fijo de
64 MiB; tamaño/SHA del inventario, JSON estricto y procedencia siguen comprobándose.
Verificación y registro se realizan fuera del bucle de eventos de la consola.

Cada Attempt escribe `result.json`, `environment.json`, `work.log`, JSONL de métricas, progreso,
outputs y artefactos. La Execution escribe `execution.json`, su configuración inmutable y un
`result.json` agregado. Se preservan identidad, paquete/código consumidor, argumentos lógicos,
entradas, recursos pedidos, tiempos, resultado, métricas, checksums, datasets, fallo, resume y Job.

`environment.json` es la única fuente de procedencia del entorno: Python/plataforma, paquetes
críticos, Torch/CUDA, Git y herramientas externas usadas. No se duplica el listado en cada resultado.
Las rutas físicas son evidencia operacional y nunca sustituyen la identidad lógica.
`lf results list/show/compare` lee manifests, no infiere semántica mediante globs. La comparación
calcula count/media/min/max; una clasificación exige `--metric` y dirección explícita.

Usa `lf export WORK --output PADRE_LOCAL` cuando la evidencia disponible deba salir del clúster.
No equivale a copiar manualmente el workspace: selecciona el Attempt exacto más reciente, registra
su estado actual, incluye artefactos finalizados externos y crea un inventario de checksums sin
estado de runtime compartido. La evidencia final puede incluir análisis/replay; la activa,
cancelada o fallida queda identificada como snapshot. Si un nombre identifica varios Works falla
como ambiguo; usa el `work_id` que muestra `lf overview --json`.

Valores finitos estructurados como `values: [[0.005, 2560.0]]` representan un único valor
categórico, no dos límites independientes a optimizar. Cobertura/efectos los agrupan mediante JSON
canónico sin reescribir los parámetros. Exportar su evidencia completada no requiere modificar
el YAML ni repetir entrenamientos.

El export remoto crea primero un ZIP64/Deflate y lo elimina siempre tras transferirlo. Descarga y
extracción local se alojan bajo `PADRE_LOCAL`, no en `/tmp`; ese filesystem debe admitir el ZIP y
un paquete extraído. Hard links seguros evitan duplicar otra vez esos bytes durante el ensamblado
atómico. En la Consola es una operación background: puedes abandonar el Study y seguir fase,
tiempo y tamaño comprimido en el panel global **BACKGROUND**.

`--profile default` es el perfil portable de auditoría: resultados científicos y decisiones del
controlador siguen completos, las trayectorias métricas grandes se reducen de forma determinista
conservando extremos y la telemetría de recursos de alta frecuencia se resume. `--profile full`
mantiene los streams métricos/de recursos crudos. `manifest.json` enumera cada fuente transformada
u omitida con motivo, bytes originales y SHA-256, evitando confundir compacidad con evidencia
desconocida.

## 10. Clústeres y Jobs

<a id="guide-execution-queues"></a>

### Colas de ejecución secuenciales

`--queue` prepara un Work/Study ahora, pero solo ejecuta su proceso científico cuando acaben los
anteriores del mismo proyecto/cluster. Admite múltiples entradas, no solo un sucesor:

**La cola ordena envíos completos, no Runs.** Una vez admitido un Study, sus Trials/seeds
conservan la ejecución concurrente habitual y los límites de recursos/ejecución declarados.
`--queue` no fija `max_parallel: 1`, reduce las GPU solicitadas ni serializa un sweep. Una Run
activa puede estar preparando entradas del consumidor en CPU antes de reservar CUDA; la
admisión espera su evidencia de consumo en vez de lanzar Runs adicionales sin medir sin límite.

```bash
lf run experiments/a.yaml --on citius-ctgpgpu16 --queue
lf run experiments/b.yaml --on citius-ctgpgpu16 --queue
lf run experiments/c.yaml --on citius-ctgpgpu16 --queue
lf queue --on citius-ctgpgpu16
lf queue --on citius-ctgpgpu16 --all --json
```

También sirve si A ya está ejecutándose por un envío normal y se ponen B/C con `--queue`.
El orden FIFO se asigna atómicamente al registrar en el destino; para llamadas simultáneas manda
ese orden de llegada. Terminar antes la preparación no permite adelantar al anterior.
Sin `--queue` se conserva el envío concurrente: un nuevo Job normal no se serializa ni cancela
automáticamente. La cola pertenece al proyecto/destino, no a otros proyectos o usuarios.
La admisión de recursos/GPU sigue considerando ocupación ajena.

En `lf`, abre **Run Work**, selecciona YAML/cluster y marca **Queue after earlier Studies / Works**.
**View execution queue…** muestra orden, nombres, fase de admisión y último estado Job registrado,
con Refresh y Back. Es metadata, no una prueba nueva de vida ni historiales Study/Run.
Las vistas normales muestran `queued` y mantienen logs y cancelación.

La preparación sigue el ciclo nativo detached: validación, bundle, entradas exactas y entorno
gestionado antes del acuse del proveedor. El YAML encolado queda fijado; editarlo durante esa
preparación rechaza el envío en vez de cambiar su revisión. Puedes cerrar la consola mientras el
preparador detached trabaja, pero **mantén el PC conectado hasta que todos los Jobs deseados estén
en `queued` o `running` del proveedor**. A partir de ese hand-off, los supervisores y metadata FIFO
residen en el cluster, sin despachador en el portátil ni terminal abierta. `preparing` no significa
que código/entradas hayan llegado ya al servidor.

El ProcessSupervisor existente aplica la espera antes de reservar CPU/RAM/GPU/almacenamiento o
lanzar ciencia. Los supervisores en espera no ocupan slots de entreno ni contextos GPU por la cola.
Solo avanzan tras un estado terminal y liberación de procesos/recursos propios; propietario
desconocido, evidencia ausente o no inspeccionable bloquea de forma segura. La limpieza confirmada
deja un recibo de liberación para poder borrar después el workspace antiguo sin perder el orden.
Predecesores, código y entornos preparados quedan protegidos mientras son necesarios. La espera
no consume tiempo de entrenamiento científico; tras admisión se aplican los límites normales.

El siguiente entra tras **cualquier resultado terminal**: éxito, fallo, cancelación o timeout.
Es planificación, no un DAG de dependencias científicas ni retry automático. Cancela con los
controles confirmados del Work/Study o el ciclo nativo, por ejemplo:

```bash
lf cancel JOB_ID --dry-run
lf cancel JOB_ID --apply
```

Un ticket **no despachado** abandonado porque su preparador no puede terminar se puede retirar
explícitamente con `lf queue --on CLUSTER --remove-pending JOB_ID --apply`; no envía señales a
Jobs ejecutándose. Los ya despachados exigen cancelación nativa. Una recuperación crea otro Job,
no reutiliza su ticket FIFO antiguo; `--queue` se ofrece en envíos nuevos `lf run`, no en acciones
de recuperación. No relances un Study completo como sustituto de recuperar sus Runs.
No borres/edites metadata para desbloquear: manifiesto ausente/corrupto bloquea, no reinicia la cola.

Soporte actual: hosts POSIX de procesos directos, locales o SSH, incluido `gpu exec`. Se rechaza
SLURM/Fleet explícitamente, sin otro runner. Hay que actualizar un runtime existing antiguo;
se comprueba su capacidad antes de despachar para que no ignore la espera. Registro/historial
acotados a 10.000 tickets y 8 MiB de metadata: protección operativa, no límite de una entrada.
No se ofrece reinicio automático de preparadores/supervisores perdidos, prioridades ni cola
entre proyectos. La cola **no** reclama/renueva reservas GPU ni evita expiraciones/revocaciones
del proveedor: siguen mandando los grants y la admisión nativos.

### Configuración de clústeres y destinos Fleet

Inspección Fleet: `lf fleets list/show/offers`; controles de catálogo:
`lf fleets drain/disable/enable FLEET CLUSTER [--apply]`. Observar no concede GPUs y estas acciones
nunca detienen Jobs activos. `lf run CONFIG --on-fleet NAME [--dry-run]` usa un coordinador local
durable y allocations managed para Studies adaptativos nuevos/fixed/repetidos/pareados automáticos, con un planner
nativo y análisis único. Run Work ofrece el mismo selector. Show/logs de Runs vivos/terminados
resuelven su miembro; resúmenes escalares actuales/mejores nunca leen rutas remotas localmente.
Curvas/logs siguen siendo lazy; cancel/delete abarcan la familia propia. Streams escalares ordenados
y poda central exacta soportan HPO adaptativo nuevo. Offers GPU baseline; continuación de checkpoints/
fidelity, co-location, recovery/adopción, expansión en vivo y export distribuido siguen bloqueados.
Consulta
[estado y contratos de implementación](#guide-coordinated-studies) antes de usar fleets.
El ejecutor preparado común delega en ControlPlane/JobService y verifica identidades inmutables y
reubicación canónica de inputs. La observación en el host va por lotes; nunca autoriza GPUs
no concedidas.

Un usuario configura el perfil en `lf` → Clusters. Primero aparecen campos comunes y cada opción
enfocada explica significado/riesgo; SSH, entorno, rutas y política GPU avanzados quedan ocultos
hasta pedirlos. Guardar, test, bootstrap y doctor son acciones explícitas separadas. La Consola usa
directamente catálogo/credenciales y nunca guarda passwords en el perfil. Para automatización:

```bash
lf clusters add gpu --host HOST --user USER --workspace /remote/work \
  --project-root /scratch/USUARIO/mi-proyecto
lf clusters bootstrap gpu --dry-run
lf clusters bootstrap gpu
lf doctor --on gpu
lf run experiments/train.yaml --on gpu
```

Para un perfil existente:

```bash
lf clusters set gpu project_root /scratch/USUARIO/mi-proyecto
lf clusters show gpu
```

El acceso a GPU se configura una vez por clúster:

| `gpu_access.mode` | Significado |
|---|---|
| `auto` | `scheduler` con SLURM; `exclusive` en un host de procesos directo |
| `scheduler` | el batch scheduler posee reserva y aislamiento |
| `exclusive` | espera lease LambdaForge y evita uso externo observado |
| `shared` | los Jobs pueden compartir GPUs; cada Study coordina comprobación y arranque entre proyectos |
| `command` | no crea lease directo; antepone un argv de claim/launcher del centro |

```bash
lf clusters set host-libre gpu_access.mode shared
lf clusters set citius-gpu gpu_access '{mode: command, command_prefix: [gpu, exec]}'
```

En un host directo compartido, los Studies no reservan exclusivamente todas sus GPUs durante
la vida del Job. Las nuevas Runs prefieren una GPU permitida sin procesos de cómputo observados
(también se consideran otros Studies/usuarios); si están ocupadas, se conserva el empaquetado
best-fit del planificador de recursos. Locks breves por dispositivo serializan la comprobación
actualizada de VRAM y el arranque, con separación temporal entre Studies. No son otra reserva de
memoria. Las aplicaciones ajenas no participan en estos locks: no se garantiza aislamiento ni
ausencia de OOM. Las concesiones del centro y leases exclusivos siguen siendo autoritativos;
nunca se añade una GPU fuera del subconjunto permitido.

Los Studies de seeds repetidas usan el mismo dispatcher: un candidato con diez seeds tiene diez
Runs independientes obligatorias, no un límite de paralelismo de uno. `execution.max_parallel`,
`runs_per_gpu`, la seguridad de memoria/ARI y la asignación agregada CPU/RAM del Job siguen
limitando la concurrencia. Varios Jobs necesitan **recursos del host**, no solo VRAM libre: reservar
46 núcleos en un host de 48 deja dos para otros Jobs, así que otra solicitud de 32 queda en cola
correctamente. Ajusta deliberadamente los recursos de futuras ejecuciones, sin sobreasignar el
host. Builds del mismo entorno gestionado esperan durante el aprovisionamiento y reutilizan su
finalización verificada; la limpieza opcional de entornos obsoletos se pospone si Jobs activos
mantienen un lease de caché.

El comando es argv y nunca un string de shell. En CITIUS, `gpu exec` es el modo gpuctl preferido
para una GPU: su reserva coincide con la vida del comando. Para un Work multi-GPU, o si se necesita
claim persistente, claim/release se declaran juntos:

```bash
lf clusters set citius-gpu gpu_access \
  '{mode: command, command_prefix: [gpu, exec], claim_command: [gpu, claim, --numgpus, "{gpu_count}"], release_command: [gpu, release]}'
```

`{gpu_count}` es la única interpolación y procede de la petición GPU absoluta del YAML. El claim
ocurre en el envío durable en segundo plano; el supervisor directo libera tras éxito, fallo o
cancelación y también se intenta liberar si falla el submit. En SLURM se rechazan claims
persistentes. En acceso command/scheduler, el proceso hereda `CUDA_VISIBLE_DEVICES` del centro:
LambdaForge trata índices/UUID/MIG UUID como grants opacos, solo los estrecha por Run y nunca los
amplía o sustituye. Una asignación ausente o repetida falla cerrada y scheduler exige el grant
exacto. Un Study adaptativo tras un launcher command puede reducirse a menos tokens heredados; su
paralelismo y probes usan exactamente ese conjunto reducido. El Python gestionado es una ruta
absoluta, sin depender de activación shell/Conda. `shared` sigue siendo una
elección explícita de riesgo en hosts permisivos, no el default.

Si el grant puede reducirse o ampliarse durante la ejecución, configura
`gpu_access.visibility_command` como argv cuya primera línea no vacía contenga los tokens opacos
actuales separados por comas. `[gpu, exec]` deriva `[gpu, env]` automáticamente. LambdaForge cruza
cada lectura con el grant heredado inicial, por lo que nunca descubre ni usa una GPU física nueva.
Un token revocado detiene solo su worker verificado y reencola la misma Run lógica como Attempt
compatible con checkpoint; los demás continúan y un token original restaurado vuelve a ser
elegible. Si el probe falla temporalmente, se conservan Runs sanas pero no se admiten nuevas.
Este comando debe informar reservas/propiedad, no solo GPU con procesos activos. La telemetría de
recursos expone GPU solicitadas, tokens iniciales, grants actuales, slots admisibles y el motivo de
ocio de cada dispositivo; una reducción real se persiste y registra como
`GPU_ALLOCATION_SHRUNK N → M`.

`workspace` y `project_root` tienen responsabilidades distintas. `workspace` es estado propiedad de
LambdaForge: bundles, entornos, directorios de Job y logs se pueden limpiar según sus reglas.
`project_root` es un mirror parcial persistente propiedad del investigador del directorio local que
contiene `pyproject.toml`; LambdaForge nunca lo sincroniza ni elimina. Debe ser una ruta SSH absoluta
distinta de `/`. `lf doctor` comprueba que exista el directorio configurado.

Se conserva el layout original. Si el YAML es `PROYECTO/experiments/design.yaml`,
`{file: ../data/dna/design}` corresponde a `PROYECTO_REMOTO/data/dna/design`; también
`publish_to="../data/resultados.json"` se mapea así. Un `publish_to` absoluto sigue siendo una ruta
remota explícita y puede quedar fuera del mirror.

Hay tres vías deliberadas:

1. Entrada pequeña (hasta 10 MiB): snapshot y transferencia automática en el bundle.
2. Entrada mayor del mirror: el investigador o servicio de transferencia del centro la coloca en la
   ruta relativa equivalente; LambdaForge lee ambos lados y exige tipo/tamaño/SHA-256 exactos, por
   lo que datos ausentes, viejos, parciales o con symlinks fallan de forma segura.
3. Corpus grande reutilizable: dataset gestionado/materializado, que evita rehashear un mirror de TB
   en cada envío y aporta identidad inmutable y placements por clúster.

El mirror no es un `rsync` oculto: copiar cientos de GB durante `lf run` haría impredecibles la
latencia, cuota y red, y muchos centros exigen un DTN. Se sincroniza mediante el mecanismo aprobado
por el centro y después se ejecutan `lf doctor` y `lf run`.

El bootstrap humano informa cada fase por stderr y emite un latido periódico durante solves o
instalaciones largas. Esto no ensucia `--json`, cuyo stdout sigue siendo un único documento máquina.

`lf run` local/remoto crea un Job durable de preparación y devuelve control. `--wait-for-submit`
espera preparación y aceptación del scheduler, no el cálculo; `--dry-run` es directo y sin efectos.
SSH usa OpenSSH, claves/agent/known_hosts/ProxyJump y un ControlMaster privado durante su persistencia.
Los passwords solo proceden de prompt/keyring/env y no entran en argv/YAML/bundles/estado/logs.

Los entornos gestionados son instalaciones inmutables de usuario identificadas por wheels, Python
y plan Torch. Pueden reutilizar Conda o micromamba verificado, pero nunca modifican Python/CUDA/
drivers del sistema ni hacen fallback silencioso a CPU. Cada Job tiene workspace mutable propio;
no ejecuta sobre bundle cache. En remoto `source_dir` es el código consumidor staged.

### Paquetes nativos declarados por el proyecto

Los ejecutables nativos pertenecen al proyecto instalable, no a cada YAML. La declaración opcional
es:

```toml
[tool.lambdaforge.environment]
manager = "conda"
file = "environment.yml"
required_executables = ["mmseqs", "foldseek"]
```

`manager` solo admite `conda`, pero LambdaForge usa su micromamba fijado y verificado; no exige
Conda global. `file` y `lockfile` son excluyentes y quedan bajo la raíz del proyecto. Los ejecutables
son nombres simples. `package_cache` solo acompaña un lock offline. En `environment.yml` solo se
admiten `name`, `channels` y `dependencies` como strings. Se rechazan prefix, variables, secciones
`pip:` anidadas, hooks y paquetes Torch/CUDA, cuyo plan pertenece al clúster.

```yaml
name: wisdom
channels: [conda-forge, bioconda]
dependencies: [python=3.11, pip, biopython>=1.84, foldseek, mmseqs2]
```

```bash
lf clusters bootstrap gpu --project . --dry-run
lf clusters bootstrap gpu --project .
lf doctor --on gpu
lf run experiments/design.yaml --on gpu
```

El dry-run no descarga, resuelve canales, construye wheels ni modifica el clúster: explica hash,
paquetes, plataforma/subdir, ejecutables, conectividad y un solve exacto ya cacheado si existe. La
preparación real resuelve inventario name/version/build/channel/subdir y lo incorpora a la identidad
junto con bytes de especificación y wheels, plataforma, Python, Torch y política offline.

Micromamba crea el prefijo de dependencias exacto en su **ruta definitiva**, donde su propio Python
instala Torch y wheels. No se supone que Conda, console scripts Python ni lanzadores shell sean
reubicables. `pip check`, inventario Conda, TLS, framework, Torch/CUDA y ejecutables se verifican
en esa misma ruta antes de publicar atómicamente el recibo de «listo», sin renombrar el entorno
instalado. El recibo registra path/versión y paquete propietario con versión, build,
canal y subdir. Builds concurrentes comparten lock por identidad. `tools.require()` busca primero el
`bin` de ese Python y sigue siendo comprobación, nunca instalador.

**Publicación y recuperación gestionada.** El contrato `stable-prefix-v1` forma parte de las
identidades de entorno/dependencias. Los prefijos del contrato anterior se conservan: no se
parchean, sobrescriben ni reutilizan silenciosamente. Bootstrap/envío actual construye otro
prefijo; si existe un recibo congelado válido del contrato anterior, conserva sus versiones Python
exactas y mantiene el inventario nativo exacto. Un recibo corrupto/incompatible bloquea la operación.
El nuevo binding operativo no cambia identidades de inputs, datasets ni parámetros científicos.

El lease de construcción/GC por identidad cubre reserva, instalación, verificación en destino y
publicación. Micromamba recibe un directorio vacío: la reserva se guarda en el lock, fuera del
prefijo. Un prefijo sin sellar nunca se considera listo/activo. Un fallo deja
`.lambdaforge-publication.json` con diagnóstico redactado, sin recibo de listo. No se repara en
sitio automáticamente. `lf clean --on CLUSTER --environments-only --json` muestra el plan y
`--apply` elimina solo prefijos fallidos/obsoletos sin referencias ni leases activos. Jobs activos,
observaciones de procesos y dependencias de recuperación siguen protegidos; si GC lo conserva,
no fuerces su eliminación. Un build interrumpido sin evidencia fiable permanece protegido por
propiedad no resuelta, no se supone basura.

Tras instalar LF corregido en el controlador, un fallo de preparación se recupera sin cambiar
el YAML de WISDOM. Ejemplo, **revisar antes de aplicar**:

```bash
cd /home/david/Escritorio/WISDOM
lf clusters bootstrap citius-ctgpgpu16 --project . --dry-run
lf clusters bootstrap citius-ctgpgpu16 --project .
lf retry job-20261113230941-0a6d159f --config experiments/preprocess/enrich_physics.yaml --accept-code-change --dry-run --json
lf retry job-20261113230941-0a6d159f --config experiments/preprocess/enrich_physics.yaml --accept-code-change --json
```

Bootstrap prepara/verifica runtimes, no ejecuta ciencia. `retry` conserva Execution/Run/checkpoints
propios cuando la ciencia ya empezó; si aún no hubo Execution, reintenta preparación, sin fingir
continuación de checkpoints. Aceptar cambio de código no omite identidad científica. No hagas un
`lf run` nuevo para recuperar una Execution existente, ni edites lanzadores, borres prefijos
protegidos o uses `--rerun` para ocultar este fallo de infraestructura.

Bootstrap, preparación automática de envíos y reutilización de prefijos comparten el mismo sondeo
de versiones nativas. Normalmente se exige salida no vacía y código cero. APBS requiere además un
banner `APBS <versión del paquete>` coincidente con el paquete Conda propietario `apbs`; únicamente
APBS **3.4.1** con `--version` puede devolver **13** en lugar de cero. La excepción no afecta a la
ejecución científica ni a otros argumentos, versiones o ejecutables. El recibo conserva argv,
código de salida, stdout/stderr acotados y política junto con versión/build/paquete. Los fallos
explican ambos sondeos, códigos y diagnósticos capturados, incluidos errores de arranque y timeouts.
Siguen siendo obligatorias la pertenencia al paquete y la verificación del inventario exacto.
Actualiza LambdaForge en el controlador antes de preparar/hacer bootstrap del entorno gestionado;
no parches prefijos históricos inmutables ni cambies el YAML científico.

**Selección de bibliotecas nativas.** En Conda gestionado, LF vincula la política por proceso
`native-prefix-libraries-v1` al directorio `lib` del prefijo de dependencias seleccionado. Los venvs
de código utilizan su base vinculada, no un `lib` inventado bajo el venv. Un lanzador aislado de
solo biblioteca estándar (`-I -S`) establece la ruta del cargador y hace `exec` del comando original:
se aplica antes de Torch, SciPy o el `sitecustomize` del destino. Sin exportaciones globales,
reordenar imports, rebajar paquetes, parchear binarios ni rutas propias de un clúster.

Preparación, reutilización, sondeo de ejecutables, ejecución supervisada/SLURM y descendientes
normales conservan esa política. Se aplica **dentro** del wrapper GPU del proveedor: respeta su
concesión y conserva rutas absolutas de drivers después del prefijo elegido. Omite entradas
vacías/relativas del cargador y elimina duplicados. No cambia visibilidad GPU ni fuerza
`LD_PRELOAD`; preloads incompatibles u otro runtime C++ realmente cargado hacen fallar la
verificación. `tools.run(env=...)` no permite sobrescribir variables del cargador con la política
activa. Al entrar en una herramienta aislada declarada se retiran rutas/marca del consumidor,
conservando drivers del proveedor y concesiones GPU.

Antes de sellar/reutilizar, procesos nuevos prueban Torch → SciPy y SciPy → Torch (si SciPy está
instalado), operaciones CPU pequeñas de tensor/KDTree y el origen real en `/proc/self/maps` del
runtime C++/GCC seleccionado, cuando existe. Importar sin error no basta para acreditar qué
biblioteca se usó. JSON de bootstrap, metadata del Job y recibos nuevos incluyen
`native_library_policy`/`native_libraries`; `environment.json` de cada Run registra las bibliotecas
realmente cargadas y rechaza orígenes incompatibles con la política activa. No se reescriben
recibos inmutables ni Runs antiguos: reutilizar genera evidencia operativa actual, sin afirmar que
el pasado superase estas comprobaciones. No hace falta reconstruir dependencias compatibles solo
para activar la política. No se adivinan rutas para entornos no nativos o `existing`. Invocar
manualmente el Python cacheado fuera de LF no equivale a este arranque gestionado.

Para desplegar la corrección, activa el entorno Python habitual de LF y revisa cada preview antes
de su comando de aplicación:

```bash
python -m pip install --force-reinstall --no-deps /home/david/Escritorio/SimpleLambda/LambdaForge
cd /home/david/Escritorio/WISDOM
lf clusters bootstrap citius-ctgpgpu16 --project . --dry-run
lf clusters bootstrap citius-ctgpgpu16 --project .
lf retry job-20261113234529-a34c00f4 --accept-code-change --dry-run --json
lf retry job-20261113234529-a34c00f4 --accept-code-change --json
```

Se preparan wheels exactos por la vía gestionada, sin editar con pip prefijos remotos activos.
No borres entornos de dependencias ni abras otro experimento con `--rerun` para recuperarlo.
Retry mantiene identidad científica, checkpoints y compatibilidad: no omite controles ni promete
continuación de entrenamiento sin soporte de checkpoints del Work.

La verificación enriquece `micromamba list --json` con los registros regulares `conda-meta` del
prefijo, conservando el subdir real `linux-*` o `noarch` aunque esa versión del gestor lo omita. La
metadata ausente es un error de entorno y una diferencia real se resume por paquete/campo sin volcar
el prefijo completo.

Offline requiere lock Conda `@EXPLICIT` específico de plataforma, SHA-256 en cada URL y sus bytes:

```toml
[tool.lambdaforge.environment]
manager = "conda"
lockfile = "locks/linux-64.explicit"
package_cache = "vendor/conda-linux-64"
required_executables = ["mmseqs", "foldseek"]
```

Cada basename/hash debe corresponder a un fichero regular; links, duplicados, credenciales, queries
y plataforma distinta fallan antes del envío. El cache es content-addressed y reconstruible tras
publicar el prefijo. Para que también pip/Torch sea offline se mantiene además el `wheelhouse` del
perfil. Cada plataforma (`linux-64`, `linux-aarch64`, `linux-ppc64le`) necesita su lock/cache.
`environment: existing` conserva intencionadamente su contrato manual y no instala esta declaración.

la Consola de investigación muestra Works y clústeres semánticos: arriba/abajo recorre. En un estudio Enter/derecha
avanza por Trials, Runs de seed y su panel vivo; `a` abre los Attempts externos numerados. Los demás
Works avanzan directamente por Attempts y logs. Un estudio terminal sin índice publicado cae a sus
Attempts externos, conservando la salida ordinaria y el error aunque el fallo precediera a la
telemetría. Dentro de una seed fallida, `e` amplía tipo/mensaje/fase/ruta/traceback; en el log de un
Attempt la misma tecla alterna el traceback persistido. Izquierda vuelve. El log abierto se actualiza solo y conserva
el scroll manual; situado al final sigue líneas nuevas. Los IDs operativos largos se reservan para
`lf jobs` y la vista máquina. `lf overview --json` expone Works, `attempt_history` y Jobs para
herramientas. `lf jobs ...` conserva el
control avanzado de IDs, scheduler, logs, cancelación y reconciliación. Cada Job local conserva sus
raíces absolutas, por lo que envío detached, observación, logs y borrado no dependen del directorio
actual; los registros relativos antiguos se resuelven desde la configuración fuente guardada. Un
proveedor inaccesible es estado unknown/last-known, no un falso fallo científico; al recuperarse se
restaura el estado real y desaparece el error de conexión obsoleto.

Un registro `unknown` no se declara terminal. Aun así, `lf delete WORK` puede previsualizar y
aplicar un olvido exclusivo del historial local si el estado sigue sin poder verificarse: no borra
ningún proceso ni workspace remoto. Así se limpia historial obsoleto sin fingir una cancelación;
si el cómputo puede seguir vivo hay que reconectar y cancelarlo primero.

La cancelación respeta la jerarquía. `lf cancel WORK` cancela todos los Jobs no terminales agrupados
en ese Work semántico; aunque falle un proveedor intenta los demás y después informa de operación
incompleta. `lf jobs cancel JOB` y cancelar un Attempt numerado afectan solo a ese Job. Un supervisor
directo señala primero el grupo científico cuya identidad verificó y después localiza descendientes
del mismo usuario con el `LAMBDAFORGE_JOB_ID` exacto heredado, incluidos workers reparentados o con
sesión propia. Termina, escala tras una gracia acotada y comprueba cero supervivientes antes de
confirmar. La misma limpieza se aplica al salir el proceso principal, evitando que un éxito nominal
abandone trainers o dataloaders. Este marcador es procedencia interna; el consumidor no debe
asignarlo ni sustituirlo. En SLURM se usa el comando configurado, cuya política debe cancelar la
asignación completa y sus steps. La cancelación es idempotente para Jobs directos ya registrados
como cancelados: repetir `lf cancel WORK` vuelve a verificar ownership y corrige fugas huérfanas de
versiones anteriores.

El workspace de Study ofrece **Delete Study History…**: obtiene primero un preview del `work_id`
exacto, confirma una vez y elimina solo su estado terminal. Un Study activo debe cancelarse antes;
ni datasets publicados, ni entornos compartidos, ni ejecuciones homónimas quedan incluidas.

Cada Job Work publica además un resultado estructurado acotado en la raíz exacta del Job.
`lf logs WORK` y `lf jobs logs JOB` añaden `Scientific failure` después del stream recortado si los
logs no contenían el diagnóstico. Siempre conserva tipo, mensaje, fase/ubicación y ruta del
`result.json`; `--verbose` o `--debug` añade el traceback salvo que ya esté visible. `--tail N` solo
recorta stdout/stderr, nunca la causa terminal. `--json` devuelve `failure`, `failures` y
`result_path`. `lf show` y `lf jobs show` exponen el mismo fallo. Workspaces 0.12 anteriores se leen
de su ubicación legacy acotada; los nuevos usan la copia exacta de la raíz del Job.

## 11. Limpieza y seguridad

- Attempt/Execution: resultados, logs, artefactos, progreso y workspace exacto;
- Run: checkpoints e intentos;
- durable independiente: datasets publicados y placements;
- compartido inmutable: entornos y bundles referenciados;
- reconstruible: caches y entradas no referenciadas;
- externo: fuentes y ubicaciones mencionadas, nunca poseídas por referencia.

`lf delete WORK` es preview-first y elimina una Execution local exacta o Job terminal exacto. Los
datasets y materializaciones tienen operaciones separadas. `lf clean` presenta cache reconstruible
y la misma compactación de Attempts terminales verificada por hash que se ejecuta automáticamente.
Conserva referencias/leases activas, outputs correctos no publicados y evidencia ligera; rechaza
raíces/symlinks inseguros y es idempotente. Esto permite recuperar con seguridad el bulk de Jobs
anteriores tras actualizar. YAML es código confiable y puede importar Python arbitrario;
LambdaForge no es sandbox.

**Clear storage…** en el clúster usa la misma autoridad preview/apply, confirma categorías y muestra
feedback en segundo plano. **Clear output** no limpia disco. La [guía](#guide-storage) describe reservas
por volumen, margen, presión, quotas/LRU, gracia/pins/publicación, Fleet/SLURM y límites actuales.
Los sondeos normales solo leen filesystem/leases, nunca recorren datasets ni Runs.
`lf storage status --on CLUSTER` inventaría uso explícitamente. `lf storage reconcile --on CLUSTER`
informa cambios; `--apply` actualiza atómicamente el inventario diagnóstico y nunca elimina ciencia.
Los snapshots de publicación gestionados reservan en el volumen destino, no en el de la Run.

En la Consola de investigación, `d` confirma el borrado del Work terminal o Attempt seleccionado y `D` confirma la
limpieza de todo el historial terminal. Los Jobs activos nunca se borran. La operación se ejecuta
fuera del loop de teclado, elimina workspace, eventos y registro de envío exactos y conserva
datasets publicados, caches, entornos y Jobs ajenos. `lf jobs clear` presenta la operación y
`lf jobs clear --apply` la aplica; un fallo conserva el registro local afectado.

Al terminar existe una retención automática más estrecha: elimina artefactos
gestionados parciales de Attempts fallidos/interrumpidos y duplicados internos verificados de
outputs publicados. Conserva toda evidencia ligera requerida por la Consola de investigación, `lf logs`, resultados y
reproducción; `retention.json` registra bytes recuperados. Los entornos inmutables sustituidos
también se podan tras activar un reemplazo verificado, excepto el activo y los referenciados por
Jobs vivos/recuperados. Checkpoints correctos sin pin solo son elegibles según gracia/publicación;
la recuperación fallida/interrumpida se conserva. Bootstrap y preparación automática comparten
la regla; la procedencia del Attempt
permanece aunque se recojan bytes reconstruibles del entorno.

## 12. Referencia CLI

| Comando | Propósito único |
|---|---|
| `init` | crear proyecto Work instalable |
| `project` | mostrar raíz e identidad de almacenamiento del proyecto actual |
| `seeds` | inspeccionar streams deterministas replicate/confirmation |
| `config resolve` | mostrar la configuración versionada exacta sin ejecutar |
| `config schema/explain/diff` | ayuda estática instalada y diferencias nativas descriptivas |
| `help --search`, `completion` | descubrimiento por tareas y completion opt-in del parser nativo |
| `inspect`, `diagnose`, `watch` | lectura compacta por identidad exacta y observación solo de cambios |
| `validate` | validar configuración/clase/entradas localmente |
| `explain` | explicar firma, defaults y recursos |
| `run` | única ejecución científica |
| `lf` sin argumentos, `overview` | Consola Textual humana y vista máquina |
| `show`, `logs`, `cancel`, `retry`, `delete` | operaciones semánticas de Work |
| `jobs ...` | control de bajo nivel; `clear [--apply]` limpia historial terminal |
| `clusters ...`, `doctor`, `resources` | configuración scriptable; bootstrap admite `--project` y `--dry-run` sin efectos |
| `datasets ...` | inspección/verificación/placement/borrado de versiones |
| `results list/show/compare/analyze/report/table` | consultar/analizar evidencia; tablas derivadas JSON/CSV acotadas |
| `clean` | preview/aplicación de GC reconstruible |
| `storage status/reconcile` | uso/cambios explícitos; reconcile `--apply` solo actualiza el inventario diagnóstico |

`lf help`, `lf --help`, `lf help clusters add` y los `--help` anidados terminan con código cero.
Los fallos tienen categorías/códigos estables. `--json` es para tooling y `--debug` añade traceback.
No existen comandos antiguos de authoring o dataset build: todo cálculo usa `lf run`.

Véase [Experiencia de desarrollo](#guide-developer-experience) para ejemplos comprobados de
authoring, shell, agentes, filtros de consola y tablas paginadas, con sus límites pendientes.

## 13. Clustering

Cada estrategia tiene su propio módulo de clase en `clustering/algorithms/`; los contratos
compartidos también están separados por clase. Las importaciones de `lambdaforge.clustering`
se conservan. La [distancia FGW](#20-distancia-fused-gromovwasserstein-fgw) entre objetos
completos es una API distinta, no otra estrategia de clustering.

Se instala con `python -m pip install "lambdaforge[clustering]"`. Importar `lambdaforge` funciona
sin el extra; usar un clusterer sin backend produce el comando de instalación. El backend es
scikit-learn >=1.3,<2 por incluir una implementación madura de HDBSCAN.

```python
import lambdaforge as lf

resultado = lf.clustering.HDBSCAN(
    min_cluster_size=20,
    min_samples=5,
    distance="euclidean",
    threads=8,
).cluster(features)
```

El paquete está fuera de `nn` porque no es una capa neuronal. No hay factory ni registro adicional:
las clases directas son explícitas y una extensión hereda el ABC pequeño `Clusterer`.

Todo clusterer acepta matriz finita `[N, F]` de NumPy o PyTorch; el tensor se separa y normaliza por
CPU. Un `Distance` custom se evalúa en el device/dtype de sus parámetros o buffers y la matriz se
separa después hacia CPU para el backend.
Los algoritmos compatibles admiten además `[N,N]` con `distance="precomputed"`:
[validación y límites](#disimilitudes-precalculadas).
Devuelve `ClusteringResult` inmutable con `labels`, `n_clusters`, `noise_count`, `noise_fraction` y
`diagnostics`. `centers`, `inertia` y `probabilities` solo existen cuando son reales. Ruido es `-1`.

| Clusterer | Distancias | Evidencia específica |
|---|---|---|
| `KMeans`, `MiniBatchKMeans` | solo Euclídea/cuadrada | centros e inercia; `seed` determinista |
| `DBSCAN` | euclidean, manhattan, minkowski, chebyshev, cosine; precomputed explícita o `Distance` | ruido |
| `HDBSCAN` | mismas métricas density; precomputed explícita o `Distance` | ruido y probabilidades |
| `Agglomerative` | Ward solo euclidean; otros linkages native/precomputed | etiquetas |

Solo existe el contrato `lambdaforge.nn.distances.Distance`: `EuclideanDistance`,
`SquaredEuclideanDistance`, `ManhattanDistance`, `MinkowskiDistance`, `ChebyshevDistance`,
`CosineDistance`, `AngularDistance` y `MahalanobisDistance`. Un objeto no nativo crea de forma
explícita `[N,N]` bajo `no_grad`, valida resultado y usa `metric="precomputed"`. El límite por defecto
es 512 MiB; `max_pairwise_bytes=None` es un escape deliberado con coste O(N²).

`adjusted_rand_index`, `silhouette_score` y `stability` producen evidencia. Escalado, imputación,
PCA, parámetros y umbral de estabilidad siguen siendo política científica del proyecto.

## 14. Componentes neuronales reutilizables

Los componentes PyTorch no dependen de Work/YAML: se instancian en Python del proyecto.
`lambdaforge.nn.models` reexporta el catálogo público:

| Familia | Opciones públicas |
|---|---|
| general/densa | `Model`, `MLP`, `CNN2D`, `ECMP`, `BatchedKNN` |
| grafos | `GCN`, `GAT`, `GATv2`, `GIN`, `GraphSAGE`, `PNA`, `RelationalGCN`, `GraphTransformer`, `EGNN`, `TensorFieldNetwork`, `GINE`, `NNConv`, `SchNet`, `GVPGNN`, `PaiNN`, `DimeNet`, `StructuralGraphTransformer`, capas públicas y `GraphReadout` |
| secuencia | `RNNModel`, `GRUModel`, `LSTMModel`, `TemporalConvNet`, Transformer encoder/decoder/seq2seq, `ConformerModel`, `StateSpaceAdapter` |
| conjuntos | `DeepSets`, `SetTransformer` |
| tabular | `ResidualMLP`, `FTTransformer`, `TabNet`, `SAINT`, `AutoInt`, `DeepFM` |
| visión | `ResNet2D`, `UNet2D`, `MobileNetV2`, `ConvNeXt2D`, `VisionTransformer2D`, `FeaturePyramidNetwork2D` y bloques/backbone públicos |
| composición | `AutoEncoder`, `VariationalAutoEncoder`, `SiameseModel`, `MultiTaskModel`, `MixtureOfExperts`, `EnsembleModel` |
| generativos | `GaussianDiffusion`, `DiffusionSchedule`, `VectorQuantizedAutoEncoder` |
| científicos/implícitos | `NeuralODE`, `NeuralCDE`, `DeepONet`, `FourierNeuralOperator1D`, `SIREN` |
| árboles diferenciables | `ObliviousDecisionTree`, `NODE`, `GradTree`, `GRANDE` |

Los namespaces contiguos son `lambdaforge.nn.losses`, `activations`, `normalizations`, `pooling`,
`distances`, `similarities`, `kernels`, `encodings`, `regularization` y `uncertainty`; las métricas
clásicas están en `lambdaforge.metrics`. Las firmas y docstrings concretas son la autoridad de
parámetros.

Esta revisión no añade un `GNN` genérico, otro MLP ni una factory. El catálogo ya cubre baselines
densos, grafos, secuencias, imágenes, conjuntos y tabular. Un alias vago ocultaría decisiones
científicas de topología, agregación, readout, equivariancia y forma de salida, además de duplicar
clases probadas. Una arquitectura de dominio pertenece al proyecto; un nuevo primitivo reutilizable
solo se añade con contrato tensorial y pruebas de conformidad precisas.

### Activación simplex dispersa: Entmax15

`lambdaforge.nn.activations.Entmax15(dim=-1, force_float32=True)` implementa entmax de
alpha=1.5, no softmax. No cambian sus probabilidades dispersas, constructor ni state-dict.
El backward usa el Jacobiano analítico del simplex, con `s = sqrt(p)`:
`grad_x = s * (grad_p - sum(s * grad_p) / sum(s))`, reduciendo sobre `dim`.
No deriva candidatos de umbral no seleccionados cuyo `sqrt(0)` podía introducir NaN incluso
con logits finitos como `[0, 0, -2, -2]`. Los ceros dispersos tienen sensibilidad cero; no se
añaden epsilon, suavizado de probabilidades, clipping ni `nan_to_num`. Se verifican derivadas
superiores dentro de regiones de soporte fijo, no en las transiciones de soporte.

Para half/bfloat16, umbral y backward siguen calculándose en float32 por defecto y se restaura
el dtype de entrada. Logits en menos infinito usados como máscara conservan peso/sensibilidad
cero si quedan logits válidos sin enmascarar. NaN/+inf o filas enteramente enmascaradas no se
reparan inventando evidencia. Hay regresiones de forward y gradiente analítico en CPU/CUDA,
derivadas numéricas de primer/segundo orden y un paso sintético de optimización. Referencia:
[Peters, Niculae y Martins, 2019, proposición 2](https://aclanthology.org/P19-1146/).

Los Jobs gestionados existentes conservan su runtime LF fijado. Hay que desplegar el build
corregido mediante preparación gestionada normal antes de recuperar explícitamente las Runs
fallidas; cambiar el checkout no parchea un entorno remoto antiguo ni repara un estado del
modelo/optimizador que ya contenga valores corruptos.

<a id="graph-encoders"></a>

### Encoders genéricos de aristas, distancias, direcciones y vectores

Los nuevos bloques se importan desde `lambdaforge.nn.models.graph`, también reexportados por
`lambdaforge.nn.models` y `lambdaforge.nn`. Cada clase tiene su propio módulo. No importan proyectos
consumidores ni datasets y no crean otro runner, registro o configuración científica.

| Mecanismo | API pública | Alcance exacto |
|---|---|---|
| MPNN receptor/emisor/arista | `ECMP` existente | Atributos escalares y embedding relacional opcional; no una MLP independiente por relación |
| GIN con aristas | `GINE`, `GINELayer` | `MLP((1+epsilon)x + sum ReLU(x_emisor + E(arista)))`; proyección de arista sin sesgo |
| Filtros matriciales | `NNConv`, `NNConvLayer` | Matriz completa o restricción explícita `num_bases`, suma/media y raíz opcional; condicionador por bases centrable |
| Filtros radiales continuos | `SchNet`, `SchNetLayer` | Bases gaussianas, SiLU, cutoff opcional explícito y condicionamiento escalar de arista |
| Vectores persistentes | `GVP`, `GVPGNN`, `GVPGNNLayer`, `VectorDropout` | Escalares/vectores polares, dropout por canal y readout escalar invariante |
| Interacción radial/vectorial | `PaiNN`, `PaiNNLayer` | Bases sinusoidales, transporte vectorial, direcciones y mezcla local invariante |
| Mensajes direccionales | `DimeNet`, `DimeNetInteraction`, `DimeNetPlusPlusInteraction` | Interacciones bilineales/factorizadas distintas, Bessel/armónicos reales y tripletas tipadas |
| Atención estructural global | `StructuralGraphTransformer`, `StructuralGraphTransformerLayer`, `GraphStructuralEncoding` | Variante inspirada en Graphormer con grados/caminos y grafos independientes; sin su receta de token global |

Se reutilizan GCN, GraphSAGE, GIN, R-GCN, PNA, GATv2, Transformer disperso y EGNN fija/con coordenadas
latentes existentes. Estos bloques **no** implementan una campaña V9 de WISDOM, sus gates semánticos,
ramas químicas, controles de capacidad, pérdidas, datos o protocolo experimental. Componer capas
con subconjuntos explícitos de aristas permite ramas independientes. Una pareja que pertenece a
dos relaciones conserva dos incidencias; no se fusionan de forma oculta. El embedding de relación
de ECMP no se declara equivalente a una MLP independiente por rama.

**Contrato tensorial.** Aristas enteras `(2,E)` emisor→receptor, nodos escalares `(N,F)` y atributos
escalares `(E,A)` alineados por fila; salida en el mismo orden de nodos. `edge_geometry(pos[N,3],
edges)` devuelve distancias `(E,)` y direcciones unitarias `(E,3)` del **receptor al emisor**.
No crea vecinos, cambia posiciones ni infiere unidades físicas. Rechaza geometría no finita o
de longitud cero. GVP recibe vectores polares de arista `(E,C_arista,3)` y conserva estados
`(N,C_vector,3)`; PaiNN usa igual cantidad de canales escalares y vectoriales.
`forward_with_vectors` expone ambos estados; `forward` termina en embeddings escalares por nodo.
Normas/mezcla de canales respetan O(3), incluidas reflexiones, no escala ni detección garantizada
de quiralidad. El contrato requiere atributos escalares invariantes y vectores polares. Validar
direcciones respeta redondeo bf16/fp16, sin prometer equivarianza bit a bit en precisión reducida.

```python
from lambdaforge.nn.models.graph import GINE, NNConv, SchNet, GVPGNN, PaiNN, edge_geometry

# x: (N,F), edges: (2,E), attributes: (E,A), coordinates: (N,3).
F, A, D = x.shape[1], attributes.shape[1], 64
distance, direction = edge_geometry(coordinates, edges)
gine = GINE(F, D, A, hidden_channels=(64, 64))
ecc = NNConv(F, D, A, num_bases=4, center_conditioner=True)
radial = SchNet(F, D, edge_channels=A, radius=6.0, cutoff=6.0)
vector = GVPGNN(F, D, edge_channels=A)
painn = PaiNN(F, D, edge_channels=A, radius=6.0, cutoff=6.0)
h_gine = gine(x, edges, attributes)
h_ecc = ecc(x, edges, attributes)
h_radial = radial(x, edges, distance, attributes)
h_vector = vector(x, edges, direction[:, None, :], attributes)
h_painn = painn(x, edges, distance, direction, attributes)
```

Las distancias se proporcionan explícitamente, no se reconstruyen ocultamente. Las primitivas
`GaussianRadialBasis`, `SineRadialBasis` y `CosineCutoff` son independientes. `radius` fija la escala
de la base; `cutoff` fija aparte su soporte. En SchNet/PaiNN, `cutoff=None` conserva las relaciones
largas suministradas: no se aplica por defecto el corte espacial a otra relación. Gates y unidades
pertenecen al proyecto. Distancia cero no significa universalmente «apagar geometría»; DimeNet
exige longitudes positivas. Al apagar una fuente, el consumidor debe cerrar también su ruta
derivada en bases. Direcciones/ángulos son entradas geométricas adicionales explícitas. Apagar una
rama relacional/global exige anular toda su contribución, no únicamente su sesgo de atención.

**DimeNet/DimeNet++.** El extra opcional `lambdaforge[graph-directional]` aporta SciPy para calcular
constantes de raíces Bessel al construir las bases. Forward y gradientes son Torch; no requieren
PyG, torch-scatter ni SymPy. No se instala nada automáticamente. Límites validados: hasta 64 bases
radiales y 8 órdenes esféricos. Se usa el envelope polinómico de DimeNet y armónicos reales m=0.
El encoder genérico sustituye embeddings de número atómico por descriptores de nodo arbitrarios;
usa SiLU, inicialización Torch y readouts por nodo, no pesos preentrenados, energías/fuerzas ni
paridad con el protocolo QM9. SchNet usa SiLU en vez de atribuirse toda la receta shifted-softplus.

```python
from lambdaforge.nn.models.graph import DimeNet, GraphTriplets

triplets = GraphTriplets.from_edges(edges, len(x), max_triplets=1_000_000)
angles = triplets.angles(coordinates, edges)
encoder = DimeNet(F, D, edge_channels=A, cutoff=6.0,
                 interaction_type="factorized")  # interacción DimeNet++
h = encoder(x, edges, distance, triplets, angles, attributes)
# interaction_type="bilinear" selecciona la interacción original DimeNet.
```

Tripletas `k→j→i`, sin `k=i` ni autoconexiones, conservando aristas tipadas duplicadas. Se construyen
explícitamente sin inventar vecinos. `max_triplets` limita precomputación y `max_triplet_bytes`
comprueba tensores principales antes de expandir bases/interacciones. No son una predicción de
toda la memoria de backward. Los bloques de NNConv completa limitan matrices temporales, pero no
garantizan memoria constante de todas las activaciones retenidas por autograd.

**Atención estructural global.** Entradas `(B,N,F)`, códigos de camino `(B,N,N)`, grados de entrada/
salida `(B,N)` y máscara Booleana opcional `valid_nodes[B,N]`. Código `0`=inalcanzable y longitud
de camino `d` usa `d+1`, incluida diagonal `1`. El preprocesado satura en el último código; grados
fuera del codebook declarado se rechazan sin recorte oculto. `edge_bias[B,H,N,N]` opcional debe ser
finito y equivarante a permutación. No se elige un camino mínimo por índice para sus atributos.
`GraphStructuralEncoding.from_edges` usa BFS dirigido independiente de empates: preprocesado CPU
explícito O(N*(N+E)) y matriz O(N²) acotada por bytes. Nunca hay atención entre elementos del batch;
la salida del padding es cero. Los modelos dispersos usan unión disjunta; los globales, batch
independiente. Grados/caminos se incorporan en cada bloque como variante genérica declarada.

```python
from lambdaforge.nn.models.graph import GraphStructuralEncoding, StructuralGraphTransformer

encoding = GraphStructuralEncoding.from_edges(edges, len(x), num_spatial_codes=32)
global_encoder = StructuralGraphTransformer(F, D, num_heads=4, num_spatial_codes=32)
h = global_encoder(x[None], encoding.spatial_indices[None],
                   encoding.in_degree[None], encoding.out_degree[None])[0]
```

`max_attention_bytes` comprueba la matriz de scores antes de materializarla, no toda la memoria de
entrenamiento. Atención global cambia realmente el alcance; no se sustituye silenciosamente por
otra dispersa. No se promete toda la receta molecular/de atributos de caminos del Graphormer
original. Ejemplo sintético ejecutable: `python examples/graph_encoders.py [--directional]`, sin
Jobs, mutaciones de datasets ni entrenamiento científico. Pruebas CPU de ecuaciones, gradientes,
permutación/multiplicidad, O(3), grafos vacíos/aislados, autocast bf16, state_dict, geometría inválida,
límites de memoria y dependencias opcionales. No acreditan rendimiento CUDA ni mejora de WISDOM.

Fuentes primarias: [GINEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html),
[ECC](https://openaccess.thecvf.com/content_cvpr_2017/html/Simonovsky_Dynamic_Edge-Conditioned_Filters_CVPR_2017_paper.html),
[SchNet](https://arxiv.org/abs/1706.08566), [referencia GVP de sus autores](https://github.com/drorlab/gvp-pytorch),
[implementación PaiNN de sus autores](https://schnetpack.readthedocs.io/en/latest/_modules/representation/painn.html),
[DimeNet](https://arxiv.org/abs/2003.03123), [DimeNet++](https://arxiv.org/abs/2011.14115) y
[Graphormer](https://arxiv.org/abs/2106.05234). Las adaptaciones genéricas anteriores no se presentan
como reproducción literal de modelos completos; no se incluyen pesos ni código fuente ajeno.

<a id="graph-operator-controls"></a>

### Composición pública de mensajes y controles de fuentes geométricas

El consumidor sigue definiendo relaciones, normalización química, valores de moduladores/gates
y protocolo experimental. LF aporta ecuaciones y etapas reutilizables, no una campaña de dominio.

| Capacidad auditada | Existente / faltante | Interfaz pública |
|---|---|---|
| GCN / GraphSAGE | Existían; faltaban moduladores escalares alineados | `message_weights` opcional en capas y stacks |
| GIN / GINE | Existían; faltaba separar la actualización común | `messages`, `aggregate`, `update` |
| MPNN receptor/emisor/arista | ECMP ya existía | `project_inputs` y etapas públicas con `layer_index` |
| PNA/estadísticas de grado | Ya existían, incluidas estadísticas externas | Separación pre-MLP, agregación estadística y post-MLP, sin otra ecuación |
| Bases/cutoffs radiales | Ya existían; faltaba cerrar toda la ruta | `distance_weights`, `radial_inputs`, `forward_from_radial` de las capas |
| Tripletas dirigidas tipadas | Multiplicidades/mezclas ya conservadas | Pesos diferenciables por etapas en DimeNet/ambas interacciones |
| Soporte direccional por relación | Bases existentes; faltaba inyección | `DimeNet.forward_from_basis` |
| Atención global estructural | Ya existía; conflicto de política del consumidor | Se conserva; límites y dtype promovido explícitos |

**Modulación escalar, no adyacencia ponderada.** Firmas nuevas compatibles:

```python
GCNLayer.forward(x, edge_index, message_weights=None)
GraphSAGELayer.forward(x, edge_index, message_weights=None)
GCN.forward(x, edge_index, message_weights=None)
GraphSAGE.forward(x, edge_index, message_weights=None)
```

`message_weights` es un tensor flotante finito `(E,)`, alineado, con signo permitido y sin detach,
recorte ni normalización. GCN aplica `a_e*w_topológico/sqrt(D_out(emisor)*D_in(receptor))` sin
recalcular grados con `a_e`: **no es GCN de adyacencia ponderada**. Los grados usan incidencias
originales y pesos de autoconexión declarados. Con `add_self_loops=True`, autoconexiones suministradas
y sus moduladores se sustituyen por una canónica por nodo, modulación **uno**. Desactivado, las
autoconexiones explícitas son incidencias normales moduladas. El sesgo se añade una sola vez.
GraphSAGE modula mensajes antes del agregador; Mean divide por número de incidencias, **no** suma
de moduladores. Su raíz independiente no se modula ni se añaden autoconexiones. Una autoconexión
suministrada junto a `root_weight=True` representa dos contribuciones declaradas, no inferidas.
`None` y factores uno conservan propagación previa; los duplicados conservan su multiplicidad.

```python
a = 1 + C(edge_attributes).squeeze(-1) - C(torch.zeros_like(edge_attributes)).squeeze(-1)
h_gcn = gcn_layer(x, edges, message_weights=a)
h_sage = sage_layer(x, edges, message_weights=a)
```

**Mensajes independientes y una actualización.** Firmas públicas:

```python
GINLayer.messages(x, edge_index)
GINELayer.messages(x, edge_index, edge_features=None)
GINLayer.aggregate(messages, edge_index, num_nodes)
GINELayer.aggregate(messages, edge_index, num_nodes)
GINLayer.update(x, aggregated)
GINELayer.update(x, aggregated)
ECMP.project_inputs(x)
ECMP.messages(h, edge_index, edge_attr=None, edge_type=None, *, layer_index=0)
ECMP.aggregate(messages, edge_index, num_nodes)
ECMP.update(h, aggregated, *, layer_index=0)
PNALayer.messages(x, edge_index, edge_features=None)
PNALayer.aggregate(messages, edge_index, num_nodes, *, degree_dtype=None)
PNALayer.update(x, aggregated)
```

GIN/GINE agregan mensajes `(E,F)` por suma y aplican `(1+epsilon)*x` y MLP **una vez**. GINE
conserva ReLU por mensaje y proyección de atributos sin sesgo. ECMP conserva el orden
`[emisor,receptor,arista,relación]`, estados previamente proyectados `(N,H)`, y actualización con
su MLP, normalización, dropout y residual opcional. PNA conserva `[receptor,emisor,arista]`;
agrega `(E,M)` a `(N,M*agregadores*escaladores)` ordenando primero escalador y después agregador.
Los grados son topológicos. Para equivalencia exacta del split bajo AMP, usar `degree_dtype=x.dtype`,
como forward; por defecto se usa dtype de mensajes. Estadísticas medias de grado/log-grado se
siguen declarando desde entrenamiento. El post-MLP concatena una sola raíz.

```python
ac = c.aggregate(c.messages(x, edges_c, attrs_c), edges_c, len(x))
as_ = s.aggregate(s.messages(x, edges_s, attrs_s), edges_s, len(x))
h = c.update(x, z_c * ac + z_s * as_)  # GINE/PNA, una actualización común

h0 = c_mpnn.project_inputs(x)
ac = c_mpnn.aggregate(c_mpnn.messages(h0, edges_c, attrs_c), edges_c, len(x))
as_ = s_mpnn.aggregate(s_mpnn.messages(h0, edges_s, attrs_s), edges_s, len(x))
h = c_mpnn.update(h0, z_c * ac + z_s * as_)
```

Elegir anchuras/layouts compatibles; en PNA también coincidir en el orden de agregadores y
escaladores. El multiplicador afecta al **agregado completo de la relación**, después de min/max/std,
no antes: esas operaciones no conmutan en general. Sumar encoders completos duplica raíz/MLP y
no equivale a esta composición. Objetos de relación separados conservan parámetros de etapas
que no se llamen; la contabilidad de capacidad/optimizador debe distinguir parámetros ejecutados,
no afirmar que participan todos los registrados. Atributos cero no eliminan relación: GINE
todavía emite `ReLU(x_emisor)`. Relación cero deja intacta la raíz común. Para anular raíz GIN/GINE,
fijar explícitamente `epsilon=-1`; PNA/ECMP pueden pasar estados raíz cero a `update`, tras calcular
mensajes con los originales. `residual=False` en ECMP solo quita residual, no la entrada raíz del
MLP. Los sesgos de actualización aún pueden producir una constante.

**Fuentes de distancia derivadas.** Firmas adicionales, sin cambiar constructores/estado:

```python
SchNet.forward(x, edge_index, distances, edge_features=None, *, distance_weights=None)
SchNetLayer.forward(x, edge_index, distances, edge_features=None, *, distance_weights=None)
PaiNN.forward(x, edge_index, distances, directions, edge_features=None, *, distance_weights=None)
PaiNN.forward_with_vectors(x, edge_index, distances, directions, edge_features=None, *, distance_weights=None)
PaiNNLayer.forward(scalars, vectors, edge_index, distances, directions,
                  edge_features=None, *, distance_weights=None)
SchNetLayer.radial_inputs(distances, *, distance_weights=None)
PaiNNLayer.radial_inputs(distances, *, distance_weights=None)
SchNetLayer.forward_from_radial(x, edge_index, radial, support=None, edge_features=None)
PaiNNLayer.forward_from_radial(scalars, vectors, edge_index, radial, directions,
                              support=None, edge_features=None)
```

`distance_weights=g`, tensor finito escalar o `(E,)`, sin recorte, aplica `radial=g*R(d)` y
`support=(1-g)+g*cutoff(d)`. Sin cutoff, su valor es uno. Con `g=0` desaparece exactamente toda
dependencia de distancia por radial y cutoff, pero la geometría sigue validándose. LF no infiere
qué columna arbitraria de atributos duplica esa distancia: el consumidor debe cerrar también
esa entrada escalar con su control de fuente. Permanecen
sesgos del filtro, otros atributos y operaciones propias del nodo. En PaiNN las direcciones
unitarias explícitas **siguen activas**; no se afirma apagar toda geometría ni se usa distancia
cero como sentinel inválido.

`forward_from_radial` consume `(E,R)` y multiplicadores completos `(E,)` opcionales, **sin volver
a aplicar cutoff**. Permite combinar bases/soportes nativos por relación antes de una sola
interacción/actualización y conservar relaciones largas fuera del corte de otra rama espacial.
`support=None` es identidad. Multiplicar soporte por el modulador relacional elimina todos sus
mensajes escalares/vectoriales, no la mezcla local ni los sesgos sin mensajes. El consumidor de
bases precalculadas posee su procedencia/geometría; LF valida tensores, no una fuente física que
esa interfaz no recibe.

**Control direccional y de relaciones tipadas.**

```python
DimeNetBasis.forward(distances, angles, triplets, *, distance_weights=None)
DimeNet.forward(x, edge_index, distances, triplets, angles, edge_features=None,
                *, distance_weights=None, message_weights=None)
DimeNet.forward_from_basis(x, edge_index, radial, spherical, triplets,
                           edge_features=None, *, message_weights=None)
DimeNetInteraction.forward(x, radial, spherical, triplets, *, message_weights=None)
DimeNetPlusPlusInteraction.forward(x, radial, spherical, triplets, *, message_weights=None)
```

DimeNet usa neutral **uno**: `rbf_g=(1-g)+g*rbf(d)` y
`sbf_g=[(1-g_entrante)+g_entrante*Bessel_envelope(d_entrante)]*Y(ángulo)`. Al cerrar distancia no
queda ninguna dependencia radial/envelope/cutoff, pero los ángulos permanecen explícitos y útiles.
Cero neutral apagaría indirectamente mensajes/ángulos/readout a través de filtros sin sesgo.
Siguen nodos, sesgos y geometría angular; no equivale a pruning. Se exigen distancias positivas
finitas originales incluso cuando la fuente esté cerrada.

Por cada incidencia tipada `m_e`: inicialización de estado de arista multiplicada por `m_e`;
tripletas ponderadas por **`m_entrante*m_saliente`**; estado final de cada interacción multiplicado
por `m_saliente`, incluidas rutas afines/residuales; mensaje de readout por arista multiplicado
por `m_e`. Así un sesgo no revive una arista cerrada ni su camino mixto. Sin detach, umbrales,
filtrado de índices, vecinos nuevos o reconstrucción topológica. Pesos fraccionarios/con signo
se componen por etapa/profundidad: no son una única ponderación de importancia ni probabilidad.
Factores uno recuperan la ecuación anterior. Todos cero dejan `root(x)` más las constantes
históricas de los readouts por nodo evaluados en cero, no información de relaciones cerradas.

`forward_from_basis` recibe radial `(E,R)` y esférica `(T,L*R)`. Con bases `DimeNetBasis` separadas
por soporte relacional, elegir cada fila radial según la arista y cada fila esférica según la
**arista entrante** de la tripleta. Las tripletas mixtas se mantienen; los pesos eliminan caminos
que usen cualquiera de las relaciones cerradas. No se aplica indebidamente un cutoff espacial
global a todo. Permanecen los controles de forma/memoria antes de expandir.

Ejemplo mínimo ejecutable: `python examples/graph_operator_controls.py [--directional]`, solo
métodos públicos, sin mutar submódulos privados ni reimplementar mensajes de grafos. Constructores,
forward por defecto y claves de state_dict se conservan. ECMP rechaza ahora topología flotante/
Booleana e identificadores de relación inválidos, en lugar de truncarlos. No se reescriben
checkpoints históricos ni identidades científicas.

**Frontera global.** StructuralGraphTransformer sigue siendo denso: grados dirigidos, códigos BFS
y sesgos opcionales del consumidor, no la receta original de atributos de camino de Graphormer.
No cruza elementos del batch y enmascara padding. El límite de scores contempla promoción de dtype
por sesgos estructurales bajo autocast; **no** acota todos los tensores simultáneos, backward,
optimizador ni encodings del llamador. El límite BFS acota su matriz, no toda la lista/colas Python.
Un consumidor que prohíba matrices densas N×N debe revisar explícitamente esa política para la
comparación global o declararla no soportada/inviable. LF no la evade, inventa atributos de camino
ni llama igual a un fallback disperso.

## 15. Arquitectura y extensiones

`WorkConfig` posee parsing, introspección y expansión. `WorkRunner` enlaza runtime, llama una vez a
`run()` y finaliza resultados. `ControlPlane` selecciona destino, evita duplicados y prepara entorno,
bundle y scheduler. `Transport` y `Scheduler` son fronteras reales. Servicios de Job, dataset y
storage conservan ownership durable independiente.

| Componente | Responsabilidad | Por qué le pertenece |
|---|---|---|
| `WorkConfig` | esquema, firma Python, expansión y valores enviados | la configuración termina antes del scheduler |
| `WorkRunner` | directorios, identidad, entradas, una llamada a `run()` y resultado | es la frontera local del lifecycle científico |
| `AdaptiveSearchPolicy` y controlador WorkRunner | presupuestos de seeds, promoción conservadora, slots GPU y stops | la decisión adaptativa queda sobre cada Work y bajo una reserva externa |
| `WorkRuntime` | servicios/vistas enlazados una vez | evita estado global mutable |
| `ManagedFileStore` | claves, records, lock, validación, huella y promoción | cache/checkpoint comparten invariantes de bytes |
| `WorkCache` | valores tipados, ficheros/fetch/rate limit y lease GC | su vida difiere de checkpoint |
| `CheckpointCollection` | JSON/ficheros del Run | resume no pertenece a outputs ni cache GC |
| `OutputCollection` y retención | outputs, publicación atómica, evidencia exacta y compactación terminal | solo metadata prueba redundancia; el proveedor cubre interrupciones |
| `ToolService` | resolución, proceso argv, logs, env hijo y ledger | el proyecto elige comando; framework ejecuta seguro |
| `EnvironmentManifest` | procedencia software/hardware/Git/plugin/tool | una sola fuente de verdad |
| `SubmissionService`/`ControlPlane` | preparación asíncrona y destino | responsabilidad operacional |
| `NativeEnvironmentSpecification`/`NativeEnvironmentPlanner` | declaración estricta, solve de plataforma e inventario Conda | software nativo pertenece al despliegue, no a parámetros/hooks del Work |
| `ManagedEnvironmentProvider` | instalación Conda/venv en prefijo definitivo, pip/Torch, verificación y recibo listo atómico | convierte un plan completo en entorno ejecutable |
| `GpuAccessPolicy` | admisión scheduler/exclusive/shared/wrapper de claim | la política del clúster no entra en YAML científico ni Work |
| `StorageOperations` | GC exacto, poda de entornos, borrado de Job y compactación | toda mutación queda acotada, idempotente y auditable |
| `Transport` | conexión y transferencia | no decide semántica del scheduler |
| `Scheduler` | submit/observe/signals de Jobs | proceso directo y SLURM tienen autoridades distintas |
| `DatasetPublisher`/`DatasetRegistry` | bytes/índice inmutable y placements | dataset sobrevive al Attempt |
| adapters `Clusterer` | input, capabilities y resultado normalizado | ciencia Python independiente del runtime/YAML |

Un Attempt resuelve entradas/identidad, crea servicios, captura entorno, llama `run()` redirigiendo
logs, valida el JSON primario, finaliza todo output, añade las herramientas usadas al mismo manifest,
elimina temporales y persiste `WorkResult`. Un fallo no publica outputs gestionados ni staging de
dataset incompleto, pero conserva logs y diagnóstico.

El proyecto extiende Python normal: módulos PyTorch, losses, métricas, lectores y helpers se crean
dentro del Work. La infraestructura permanece en servicios cohesionados
`outputs/metrics/checkpoints/cache/tools/progress`. No se introduce otro ejecutable, DSL, Task,
Source/Transform/Sink ni selector global. Un campo YAML nuevo solo se justifica por una decisión de
planificación del investigador, no por un objeto interno que puede seguir en Python.

## 16. Análisis de estudios

Los sweeps fijos completos con seeds explícitas pueden ampliarse editando el YAML original:
`lf studies extend JOB --config experiments/sweep.yaml --dry-run`, seguido del plan revisado
`--apply --plan-hash HASH`. **Extend Study…** en consola comparte el servicio. Solo se ejecutan
celdas nuevas; revisiones inmutables y Runs/análisis/historial acumulados usan un ID de serie.
Se admite CPU local y CPU/GPU gestionada en el mismo destino. Por defecto, código cambiado o no
verificable crea una nueva revisión de código/runtime revisada explícitamente: conserva evidencia
previa, pero separa la inferencia estadística por revisión. `--code-policy strict` exige equivalencia
acreditada. No es retry, recuperación del optimizador adaptativo, migración ni ampliación Fleet. Consulta
[ampliaciones incrementales](#guide-study-extensions) para límites de compatibilidad, presupuesto,
caídas, exportación y continuación desde checkpoints.

Hay puntos individuales por Trial, curvas escalonadas/áreas, violines, barras horizontales,
histogramas y CDF empíricas, con ejes compartidos identificados por métricas. El proyecto añade HTML
autocontenido con `self.outputs.html_section("proteins", title="Proteins").write_text(html_text)`;
generar informe lee artifacts finalizados verificados, nunca durante planificación. Varios Runs
comparten selector de documento. Iframes aislados offline separan JavaScript del proyecto; incrusta
recursos/datos (16 MiB/documento, 64 MiB/informe). Consulta el contrato y API directa en
[pestañas HTML del proyecto](#guide-research-analysis-pestañas-html-del-proyecto).

El HTML de Study empieza en **Overview / Resumen**, con estados, importancia contextual y gráficos
recomendados. Parámetros condicionales muestran rama/soporte aparte, nunca responsabilidad global.
Interpretaciones/preguntas usan carruseles, hallazgos completos se paginan y **ⓘ** explica términos.
La cabecera permite inglés/español sin traducir nombres ni afirmaciones guardadas. Los controles
son desplegables buscables, no modales; en **Parameters**, marcar/desmarcar superpone curvas en un
gráfico (las escalas automáticas agrupan rangos sustancialmente solapados, nunca unidades conocidas
distintas). Puedes elegir líneas, puntos, barras agrupadas, distribuciones de candidatos o mapa de
calor. **Distribución** cambia entre gráfico combinado y un panel por métrica. **Escalas Y** permite
rangos independientes o compartidos explícitamente. **Dispersión** oculta la desviación típica o
usa bigotes/bandas (solo líneas con X numérico; las demás vistas usan bigotes). Se calcula entre
candidatos, no es incertidumbre entre semillas, y falta si hay menos de dos observaciones. El mapa
normaliza colores por métrica; el hover conserva medias originales y lo no observado queda vacío.
Los ajustes se conservan al reabrir el mismo HTML generado. Explore utiliza
**Analyze / By / Compare with** y chips de comparación que se pueden quitar. Los ejes y ajustes
explícitos quedan en **Advanced visualization options**. Las preguntas configuradas son tarjetas
ampliables; Metrics & health ofrece categorías jerárquicas plegables. **Evidence** sustituye la
pestaña duplicada de hallazgos y conserva conclusiones, diagnóstico, seeds, poda y JSON. Las
interacciones observadas de cualquier métrica están separadas del modelo HPO persistido.
Consulta [HTML de investigación](#guide-research-analysis).

Es una capa determinista a posteriori sobre evidencia persistida; no propone Runs ni modifica el
controlador. Un estudio terminal escribe `EXECUTION/analysis.json` automáticamente y estas rutas
reutilizan o recalculan el mismo documento:

```bash
lf results analyze EXECUTION
lf results analyze EXECUTION --recompute --json
lf results report EXECUTION --output informe.html
lf export ESTUDIO_O_WORK --output ./exportaciones
```

`lf results report` escribe únicamente una vista HTML de una Execution local. `lf export` es la
operación de evidencia portable: resuelve el historial semántico del Work en el proyecto actual,
elige el Attempt más reciente en cualquier estado y recupera su evidencia acotada desde proveedor
local o remoto. El éxito final usa `NOMBRE--EXECUTION_ID`; el resto usa un nombre de snapshot con
timestamp. Siempre se prepara y renombra atómicamente, sin mezclar ni sobrescribir, y manifest y
comando muestran el estado capturado.

El paquete contiene `manifest.json` (inventario SHA-256/bytes por fichero), `execution/` (resultado,
configuración, logs/scalars por Run, decisiones, checkpoints y artefactos retenidos), `study/`
(telemetría completa persistida del Study/controlador), `control-plane/` (ciclo de vida del Job),
`published-artifacts/` (outputs finalizados explícitos externos a la Execution) y `reports/`. Se
intentan análisis JSON y replay cuando la evidencia persistida lo permite. Un snapshot temprano
puede contener solo configuración enviada y ciclo de vida del Job. Cuando aplica el análisis,
`analysis-report` activa el dashboard Plotly y la instalación base genera una alternativa
estructurada con evidencia exacta. Se
excluyen Dataset compartidos, entornos gestionados, caché y árbol staged del proyecto: sus
referencias inmutables siguen en provenance y copiarlos a ciegas sería inseguro y potencialmente
enorme. El empaquetado remoto rechaza enlaces/ficheros especiales, la extracción local rechaza
traversal/enlaces, el ZIP temporal se borra incluso ante fallo y una carpeta parcial nunca se
publica. Archivo y extracción local viven bajo el destino elegido, no en `/tmp`, y el ensamblado
reutiliza de forma segura los bytes temporales propios cuando comparten filesystem.

HTML requiere `lambdaforge[analysis-report]`; JSON y Consola no. El informe incluye datos y runtime
Plotly y no usa red. El informe de Run aporta catálogo de métricas buscable y plegable, vistas de
curvas originales/normalizadas, snapshot, correlación y relación X/Y, además de paletas accesibles.
El informe de Study separa comparación de Trials, respuestas y resúmenes por valor, interacciones
heatmap/3D, cobertura, recursos y findings. El navegador solo describe evidencia persistida y nunca
vuelve a ajustar el modelo HPO. Las preferencias se conservan para ese fichero; regenerarlo crea
un ámbito limpio. La barra lateral de Run y los paneles de análisis se redimensionan arrastrando.
Run permite crear categorías de métricas y guardar gráficas de la selección actual; Study guarda
vistas observadas 2D/3D con ejes de parámetro, métrica numérica registrada o recursos por candidato.

Para comparar métricas frente a hiperparámetros de un search adaptativo o sweep:

1. En la consola abre **Study → Analysis → Interactive HTML**, o genera un informe local con
   `lf results report EXECUTION --output study-report.html`.
2. Abre **Explore**. Despliega **Advanced visualization options** para elegir representación. Para puntos, líneas o barras, elige un parámetro como `hidden_dim` en
   **X axis** y una métrica registrada como `val_accuracy` en **Y axis**. La gráfica se
   previsualiza al cambiar controles: guardar es opcional para conservarla, no para dibujarla.
   **Add another Y metric (2D)** tiene casillas buscables para añadir curvas sin Ctrl/Cmd-click.
   Son resúmenes por Trial, no por epoch.
3. En **3D scatter** u **Observed heatmap**, elige dos parámetros (por ejemplo `hidden_dim` y
   `dropout`) en X/Y y la métrica en **Z / cell metric**. **Observed 3D surface** requiere ejes
   numéricos; puntos 3D y mapas de calor también aceptan categorías/booleanos con sus nombres.
   Elige paleta/inversión, pon nombre a la vista y pulsa **Save chart**.
4. **One point per trial** conserva las combinaciones probadas distintas. **Mean per exact X
   value + SD** agrupa valores X iguales dando el mismo peso a cada Trial: la desviación es
   empírica entre Trials, no un intervalo de confianza entre seeds ni un efecto controlado del
   parámetro. Las celdas del mapa/superficie también promedian resúmenes observados por pareja
   X/Y exacta. Parejas no probadas quedan vacías, incluidos niveles pendientes de un dominio
   finito declarado; no se ajusta un modelo ni se interpolan respuestas ausentes. Si la rejilla
   es demasiado grande se pide usar puntos 3D.
5. Por defecto se muestra evidencia completada. Activa **Include partial / pruned observations**
   para inspeccionar datos parciales, identificados mediante marcas/hover. Elige explícitamente
   **Best observed objective** o **Current observed objective** si eso es lo que buscas. Una
   selección final ausente sigue ausente aunque exista un mejor valor parcial.

Los resúmenes de selección/componentes conservan su semántica persistida. Otras métricas numéricas
usan resúmenes diagnósticos persistidos o los últimos valores registrados de Runs de screening
comparables de una misma fidelidad, no nuevos óptimos por métrica. El hover muestra valores exactos,
IDs de Trial y soporte de cada celda. Vistas y paletas se conservan al reabrir el mismo HTML en el
mismo navegador; no modifican HPO, YAML ni evidencia. Regenera el informe para incluir resultados nuevos.
Si la vista está vacía, el mensaje distingue parámetro sin datos, selección final ausente y falta
de métrica/evidencia para el filtro. Nunca sustituye selección por un mejor valor parcial.
En **Parameters**, elige **Y metric** para graficar cualquier métrica numérica registrada y
actualizar la tabla por valor. La gráfica observada también funciona en sweeps sin curvas modeladas.
La sección plegable **Persisted adjusted selection response** sigue ligada al objetivo: cambiar
la métrica diagnóstica no convierte ni renombra el modelo persistido como un modelo de accuracy.

### 16.1 Semántica de evidencia

`current_observed_objective` es el último step con todos los componentes,
`best_observed_objective` el mejor, `final_objective` exige estado terminal y fidelidad completa, y
`selection_objective` es la evidencia del candidato. Un compuesto incompleto declara componentes
ausentes y último step completo. Una Run podada conserva curva parcial y mejor observación como
censurada, pero no recibe score final inventado. La estabilidad empírica es
`insufficient_evidence` si algún candidato líder tiene menos de dos seeds finales comparables: no
se presenta un bootstrap de un elemento como estabilidad. La incertidumbre pooled/modelada aparece
por separado si fue persistida. Las seeds compartidas permiten comparaciones pareadas; screening y
confirmación con seeds nuevas permanecen separados. La confirmación usa el margen de equivalencia
persistido (solo registros antiguos caen al fallback documentado del 2 %).

### 16.2 Efectos, cobertura y fiabilidad

Un surrogate determinista de espacio mixto se valida mediante predicciones
leave-one-candidate-out reales, no épocas/seeds duplicadas ni metadata de otro esquema. Publica
método/folds exactos, RMSE, MAE, correlación, cobertura y calidad. Un único orden min/max gobierna
ganadores, respuestas, regiones y confirmación. Los predicados `when` originales se conservan y
todo punto generado valida su dominio condicional. La importancia funcional global y la región
superior observada responden preguntas distintas. Respuestas y superficies muestran no aditividad,
incertidumbre y extrapolación.

La cobertura describe bins/categorías y resolución conjunta observados; no afirma cubrir un pool de
propuestas que no fue persistido. El coste intrínseco mediano por Run comparable se separa del gasto
total del controlador (Runs, seeds, promociones y tiempo); solo el primero alimenta el Pareto de
eficiencia, separado a su vez del Pareto científico. Constraints, pruning y confirmación se
conservan. Los findings son predictivos/observacionales, no causales. La fiabilidad combina soporte,
calidad CV, cobertura y extrapolación: mala validación nunca produce confianza alta solo por tener
muchas filas.

En vivo el documento es `provisional` y más barato; al terminar es `final`. Un fingerprint hace
idempotentes las lecturas sin `--recompute`. La escritura es atómica/versionada y admite estudios
0.13 con metadata suficiente.

### 16.3 Evidencia de admisión

La telemetría persiste Runs activas/en cola, máximo paralelo, contexto CPU/RAM, VRAM libre/requerida
por GPU y motivo exacto de espera. `runs_per_gpu` es un máximo. La presión espera y se sondea, no
falla una Run; se escalonan lanzamientos por GPU y las demás continúan. Con dos GPU, cinco slots por
dispositivo y recursos suficientes pueden admitirse diez Runs hijas; los probes efímeros no ocupan
slots científicos. Un error transitorio del probe es una caída de observación, no un fallo
científico: se pausa la admisión, no se tocan los hijos activos y se reintenta. Su salida acotada se
conserva para diagnóstico. Solo pasa a terminal si persiste, no quedan Runs activas y se agota el
presupuesto acotado de reintentos.

### 16.4 Espacio de investigación y declaraciones

El HTML offline empieza en **Overview**, con **Research**, **Metrics & health**, buscador global
(**Ctrl/⌘ K**), familias y **Explore**. Python persiste perfiles, relaciones exploratorias acotadas,
diagnósticos de multiplicidad y hallazgos de inspección; JavaScript solo presenta/filtra evidencia.
YAML `analysis` o `Work.analysis_profile` declaran etiquetas/unidades/derivación, familias y
preguntas. Antes de ejecutar se congelan en `analysis-semantics.json`, junto con defaults conocidos
del registro; recuperación/export los conservan. El esquema 8 añade `research` sin eliminar campos.

Test, incluidas derivaciones/componentes compuestos, no puede gobernar objetivo/restricciones ni
discovery provisional; el análisis terminal sí permite inspeccionarlo. Semántica desconocida
sigue explícita; discovery no es causalidad ni confianza nueva del HPO. Unidades Y diferentes
usan gráficas separadas en Explore y ejes Y independientes del mismo gráfico en Parameters.
Hay normalización visual opcional y vistas con notas/export/import JSON
validado. La [guía completa](#guide-research-analysis) reúne pasos numerados, YAML, límites de
algoritmos, cautelas estadísticas y comportamiento de Studies antiguos.

## 17. Consola de investigación

`lf` abre la Consola Textual solo con entrada/salida interactiva; en otro caso imprime ayuda y sale.
Las pantallas son Overview, Work, Studies, Clusters, Datasets y Results. Enter/derecha apila el
workspace real, Esc/izquierda retrocede un nivel, Home vuelve a raíz y `?` explica el contexto. Se retiraron `lf top`, `clusters setup` y
`clusters modify`; los scripts conservan todos los comandos no interactivos.

Overview tiene tres paneles independientes para Clusters, Work ordinario y Studies. Usa frases de
progreso breves y un detalle semántico formateado, no volcados JSON. La selección de clúster acumula
historial vivo acotado de CPU/RAM/GPU y separa carga total, uso personal observable y recursos
solicitados. Cada refresh reconstruye los read models sin cambiar el panel enfocado ni la entidad
exacta. El estado de carga centrado solo se usa hasta obtener el primer snapshot correcto. Cada
refresh posterior mantiene esa vista, indica su antigüedad abajo y marca como obsoleto cualquier
fallo transitorio. El navegador Work y sus logs acotados se sondean en vivo con una sola petición
activa por vista. Los datos siguen la jerarquía de navegación. Overview hace una lectura de
inventario por proveedor directo —o solo del estado de Jobs activos si el scheduler no ofrece
inventario— y devuelve proyecciones compactas de Job y Study; no descubre manifiestos
remotos de datasets ni incluye índices de candidatos/Runs. Las colecciones Work y Studies no
sondean recursos ni datasets. Abrir un Study carga su índice acotado; análisis, historial completo
del controlador y logs externos son lazy por pestaña, y curvas/artefactos/logs de seed solo se leen
para la Run seleccionada. La progresión máquina equivalente es `overview --json`, `show WORK` y
`show WORK --run CLAVE --json`. El inventario directo aísla registros individuales malformados,
ausentes o inseguros: los Jobs válidos siguen observables y sólo el conocido afectado muestra
degradación, conservando resultados terminales guardados. Las entradas de error no acreditan
propiedad, no crean Jobs ni autorizan recovery. Fallos de raíz/transporte siguen siendo globales;
observar no repara archivos persistidos. El mismo dashboard aparece en el navegador Clusters y en el
workspace del clúster. Los
Studies terminales siguen siendo Studies y conservan el Attempt que produjo su última telemetría;
un read model terminal con `study: null` es válido, abre una vista degradada de Study/logs e intenta
una recarga acotada. El botón Exit lateral y `q` en raíz son explícitos.

El recorrido central es `Study → Trial → Seed → Epoch`. Trials permite buscar/ordenar; Seeds separa
current/best/final, marca evidencia censurada y muestra GPU/recursos/log aislado. Curves pagina cuatro
métricas con `n`/`p`, conserva todas en Metrics y marca época seleccionada, mejor y última. Epoch
muestra escalares exactos del mismo step. Analysis navega Summary, Parameters, Interactions,
Coverage, Seeds, Pareto y Findings sobre el JSON persistido. Clusters ofrece test/doctor/bootstrap,
credenciales redactadas, GPU, recursos y storage. Summary de Dataset lee de forma asíncrona los
conteos exactos de splits y target principal por split desde el índice lógico, sin recorrer assets
pesados. Members carga al abrir hasta 200 registros; Stats físicas e Integrity explican su coste y
exigen una acción dentro de su pestaña. El borrado de DatasetVersion está siempre visible, informa
inmediatamente de su estado, bloquea repeticiones, previsualiza todas las ubicaciones y exige
confirmación. Un estado `registered_but_missing` converge limpiando
registros obsoletos sin fallar porque los bytes ya no existan. Results reutiliza análisis y lo
recalcula/exporta.

Study y Seed muestran un indicador explícito durante una lectura remota y revelan el contenido de
forma atómica o informan del fallo; una tabla vacía nunca hace de placeholder de carga. Todos los
segmentos anteriores de las migas de pan son pulsables y usan la misma pila real que Back.

El Overview del Study combina estado, candidatos/Runs, líder, tiempo de cómputo, historial del
objective, distribución de estados y parámetros líderes. Los conteos exactos aparecen bajo las
barras; la gráfica nunca obliga a estimar un grupo pequeño en un eje grande. Las marcas tienen columna propia y no
tapan el trial. HPO resume estrategia, objective humano, evidencia y scheduler; su tabla separa por
parámetro dominio declarado y observado, región mejor respaldada, importancia y confianza. Al
abrirlo muestra respuesta predictiva con incertidumbre, soporte, dispersión e interacciones por
pares persistidas. Antes de existir una Execution final, el mismo panel consume el snapshot HPO
vivo acotado, etiqueta la respuesta observada y la confianza numérica/textual y se refresca sin
esperar al final. Son asociaciones predictivas, no efectos causales. Las gráficas preservan
etiquetas categóricas y al pulsar muestran valores exactos. Omiten pseudolíneas ambiguas de
incertidumbre; la tabla conserva el número. Una acción explícita **Interactive HTML** usa Plotly
opcional para curvas con hover, incertidumbre sombreada real, mapas de calor y superficies 3D
numéricas. Coverage usa tarjetas, barras marginales y tablas exactas de soporte
declarado/observado, no JSON crudo. La ayuda contextual `?` define objetivo, importancia,
fiabilidad, confianza, ganancia conjunta, cobertura y parada. El historial completo
persistido usa Acción / Trial / Motivo y cada fila abre sus umbrales e inputs guardados.

La Consola importa servicios de dominio directamente y nunca ejecuta `lf` por subprocess. I/O,
análisis y proveedores lentos van en workers daemon; los refresh de una misma vista no se solapan,
por lo que un probe remoto lento no multiplica llamadas SSH/proveedor. Las pantallas raíz ocultas no
cargan hasta seleccionarse. **Run Work** ofrece un árbol de directorios filtrado a YAML y una lista
acotada compuesta por el historial local de Jobs existente y selecciones MRU persistentes. Tanto
explorar como elegir un reciente solo rellenan un único selector. La única acción Submit valida,
explica y solo después envía al clúster visible capturado; muestra la fase activa y deshabilita
acciones repetidas. La validación enlaza directamente con el encolado: un plan complejo que no pueda
representarse en el preview nunca oculta ni impide el envío o su resultado. El fichero MRU solo contiene
rutas locales absolutas, nombres y timestamps, filtra archivos ausentes y nunca copia YAML ni datos.
Un YAML inválido permanece en el diálogo de lanzamiento y se muestra una sola vez con ruta, línea y
columna exactas, fragmento acotado e indicación accionable. Los nombres visibles de Work no son
identidades: ejecuciones locales/remotas simultáneas tienen claves de fila `work_id` distintas, y
cancelar/borrar entrega esa identidad exacta al servicio.
Un único factory reutiliza el transporte de la sesión; las operaciones
de un clúster se serializan y sus probes de recursos se pausan durante doctor/bootstrap. El panel de
actividad muestra fases y tiempo transcurrido, ofrece tres alturas y admite arrastrar su separador.
La parte superior es un viewport independiente con scroll y altura mínima protegida; ampliar el log
no puede dejar acciones, pestañas o evidencia sin una vía de acceso. Aplicar bootstrap exige
confirmación; todas las confirmaciones de mutación y el plan modal de Work usan secciones semánticas
acotadas en vez de JSON serializado. Plan bootstrap sigue siendo de solo lectura.
Una caída mantiene el último
snapshot marcado stale. Las acciones destructivas muestran
el objetivo exacto, exigen confirmación y conservan preview/apply, raíz exacta y seguridad frente a
symlinks. `Ctrl+P` solo muestra handlers funcionales; el inventario etiqueta el resto como solo-CLI
en vez de fingir paridad. Clientes máquina consumen `--json`, no texto de pantalla. Plotly sigue
opcional.

La modal Add/Edit cluster es un editor completo del `ClusterProfile` durable, no un asistente
reducido. Sus pestañas guiadas cubren transporte/referencias de autenticación, backend, identidades
de workspace/proyecto/datos, políticas Python/environment/PyTorch, retención de almacenamiento,
conexión SSH y `gpu_access`. El modo command expone su prefijo argv y la pareja atómica claim/release;
se analiza con `shlex` y nunca se ejecuta mediante shell. Advanced conserva como YAML validado
`ssh_options`, el prefijo general, opciones/mapeo/directivas/comandos del scheduler y la política del
script SLURM. Los campos guiados prevalecen explícitamente. Antes de persistir atómicamente se
reconstruye `ClusterProfile`; los secretos permanecen exclusivamente en Credentials.

El historial de recursos y las curvas de aprendizaje se dibujan mediante el widget nativo declarado
`textual-plot`, con ticks numéricos automáticos, líneas Braille y zoom/desplazamiento interactivos.
Al cambiar de página de métricas se restauran los límites automáticos de cada gráfica reutilizada;
el pan/zoom manual solo permanece durante la página visible.
Las gráficas de recursos mantienen una ventana fija de cinco minutos etiquetada desde `5m` hasta
`now`; su clave de color queda encima para no tapar datos. Las gráficas de seed conservan páginas de cuatro métricas y
superponen en rojo la época seleccionada y en verde la óptima. LambdaForge solo conserva muestras
acotadas y series semánticas; no reimplementa ejes ni rasterización. Cada workspace anidado ofrece
un botón visible **Back to …** además de la navegación por teclado. La barra lateral separa
visualmente Action, Browse y Session. Clusters y Datasets
presentan tarjetas compactas y secciones semánticas acotadas en vez de JSON de proveedor o
manifiesto sin formato.
Las celdas de Runs/seeds podadas dicen `not final · pruned` o `not observed` cuando falta evidencia
de forma esperable; `unavailable` queda reservado para fallos reales de recuperación o capacidad.

La acción **Interactive HTML** de una seed convierte esas mismas series persistidas exactas en un
dashboard de análisis responsive y offline. Empieza con un máximo de cuatro métricas de
objetivo/validación; una barra agrupada y buscable permite añadir cualquier escalar registrado. La
misma selección alimenta curvas originales, comparación de tendencias normalizada 0–1 por métrica,
barras de último valor/cambio, correlación de Pearson por pares sobre epochs comunes y una vista
estadística con último/mínimo/máximo/cambio/número de muestras. La normalización solo compara forma,
no modifica ni ordena valores científicos; la correlación se etiqueta como descriptiva. Las épocas
mejor y seleccionada siguen como referencias. El fichero incluye Plotly y todos los datos, no usa
red ni se convierte en otro almacén de resultados.

La implementación antigua `LiveJobMonitor` no es una interfaz pública. Se conserva temporalmente
como oráculo de regresión mientras existan rutas avanzadas clasificadas `CLI-ONLY BY DESIGN`; solo
se borrará cuando Textual tenga flujos y pruebas equivalentes respaldados por servicios.

## 18. Aislamiento por proyecto

LambdaForge selecciona un proyecto consumidor buscando el `pyproject.toml` más cercano desde el
directorio actual. El entorno virtual selecciona paquetes Python; **no** selecciona el estado de
LambdaForge. Usa `lf project` (o `lf project --json`) antes de operar desde una shell desconocida
para ver la raíz y el identificador exactos.

Los proyectos creados con `lf init` reciben un identificador estable. Un proyecto existente puede
declararlo antes del siguiente envío:

```toml
[tool.lambdaforge]
project_id = "diseno-proteinas"
```

Debe tener entre 1 y 80 letras, dígitos, `_` o `-`, empezando por letra o dígito. Sin este campo se
deriva de la ruta local resuelta: dos checkouts con el mismo nombre siguen aislados. Mover el
checkout cambia el ID derivado; añade antes un ID explícito si debe conservarse el namespace
remoto. Usar deliberadamente el mismo ID explícito en dos checkouts comparte su namespace.

La frontera de propiedad es:

| Estado | Ámbito |
|---|---|
| perfiles de clúster y referencias a credenciales | compartidos; el clúster se configura una vez |
| leases cooperativos de GPU/procesos | compartidos por host; dos proyectos no sobreasignan hardware |
| Jobs del controlador, grupos y YAML recientes | solo el proyecto actual |
| resultados, datasets y caché local | proyecto actual bajo `.lambdaforge/` |
| Jobs, estado, datasets, entornos y caché remotos | namespace del proyecto actual |

Con almacenamiento remoto por defecto, el trabajo nuevo usa
`<workspace>/.lambdaforge/projects/<project-id>/{state,cache,jobs}`. Una raíz personalizada mantiene
la frontera añadiendo `projects/<project-id>`. Son rutas operativas: el ID de proyecto no entra en la
identidad científica. La raíz compartida de leases queda intencionadamente fuera.

Esto también se aplica a `storage.dataset_root`: una raíz declarada como `/datos/datasets` se
convierte en `/datos/datasets/projects/<project-id>` para publicaciones nuevas. Una DatasetVersion
antigua puede seguir resolviéndose desde una ruta absoluta sin scope si esa colocación exacta ya
estaba registrada y su manifiesto e identidad de contenido verifican. LambdaForge conserva esa
colocación durable heredada; no deduce que cualquier proyecto pueda usar bytes arbitrarios de la
raíz antigua. El directorio hexadecimal bajo `NOMBRE/VERSION/` es el prefijo del ID de contenido,
no un proyecto ni un Run aleatorio. Publicar bytes distintos con el mismo `NOMBRE@VERSION` se
rechaza; debe incrementarse la versión.

`lf datasets reconcile NOMBRE@VERSION --on CLUSTER` previsualiza reparaciones solo de índices. Un
registro remoto conflictivo solo se puede eliminar con `--apply` si se demuestra que su directorio
registrado ya no existe; bytes válidos o inaccesibles conservan el conflicto duro. Los fallos de
publicación revierten los bytes recién comprometidos, por lo que rechazar una versión inmutable no
deja un nuevo directorio gestionado sin registrar. La réplica entre dos remotos nunca cae en
retransmitir un tamaño arbitrario por el disco del controlador: se usa el servicio de transferencia
del centro y después se reconcilia la colocación exacta verificada en destino.

El catálogo de usuario `~/.config/lambdaforge/clusters.yaml` sigue siendo la fuente común para host,
autenticación y políticas del centro. El `lambdaforge.clusters.yaml` del proyecto es un overlay
recursivo, por lo que puede declarar solo su mirror remoto:

```yaml
clusters:
  gpu:
    project_root: /scratch/usuario/diseno-proteinas
```

La Consola, `lf overview`, las operaciones Work, Results y Datasets usan el mismo contexto. Un
selector de otro proyecto no existe en el actual y no puede accionarse. Así, dos Works homónimos de
proyectos distintos nunca son el mismo objeto del control plane; dentro de cada proyecto mandan los
IDs exactos.

La actualización no destruye ni mueve datos. Un Job legado solo se descubre si la ruta YAML que
registró pertenece al proyecto actual, y se reconecta usando sus rutas de proveedor/Work ya
persistidas. No se copia entre proyectos ni se trasladan directorios remotos automáticamente. Los
nuevos envíos usan el layout con namespace; perfiles, credenciales y leases compartidos no requieren
migración.

## 19. Fiabilidad operativa

`lf health [JOB] [--refresh] --json` y `lf incidents --json` inspeccionan evidencia operativa aparte
del resultado científico. `lf monitor start/stop/status/config/diagnose/once/install` controla el
observador independiente del proyecto; SMTP opcional usa referencias y deduplicación persistida.
Bare `lf` ofrece **Operational health…** con los mismos servicios.

`lf retry STUDY --config NUEVO_YAML --dry-run --json` compara el YAML trasladado con identidad
original sin iniciar cómputo. Apply verifica de nuevo propiedad y fija el archivo elegido. Eventos
Job conservan ID de petición y distinguen admisión, runtime y progreso científico. Consulta la
[guía operativa y de emergencia](#guide-operational-reliability) para configuración, plataformas,
correo, semántica de recovery y requisitos Fleet/checkpoints en vivo que siguen pendientes.
Checkpoints gestionados ofrecen cadencia/retención optativas `configure/should_save` e inventario
paginado `inventory(verify=True)` en host. Disponibilidad no prueba restauración del consumidor.
Sondas concurrentes conservan última observación y antigüedad real; un recovery auditado resuelve
incidentes del origen tras progreso científico fresco, nunca solo al aceptarse.
Cargar Runs válidos reutilizados es una referencia, no progreso nuevo. `lf jobs recovery JOB
--json [--offset N --limit N]` y Study → **Recovery history** consultan eventos nativos paginados,
Jobs enlazados y peticiones rechazadas incluso sin conexión. Validar en consola registra inspección
operativa; CLI dry-run sigue sin mutaciones. El diálogo persistente conserva petición/fase y
accesos a logs, historial y el mismo plan, sin avisos repetidos de sondeo.

## 20. Distancia Fused Gromov–Wasserstein (FGW)

FGW compara **dos objetos completos ponderados con atributos**, no nodos individuales ni clusters.
Instalación opcional explícita: `python -m pip install 'lambdaforge[optimal-transport]'`.
Importar esta API no carga ni instala POT, SciPy, sklearn o Torch. Se utiliza el solver de
gradiente condicional y búsqueda de paso de Python Optimal Transport (POT), sin otro optimizador.

### Entradas y ejemplo

`lambdaforge.distances.StructuredMeasure(structure, features=None, weights=None)` representa
un objeto. `structure` es una matriz N×N finita, simétrica, no negativa y con diagonal cero;
no se verifica la desigualdad triangular. Admite geometría euclídea, caminos mínimos de grafos
u otras relaciones intrínsecas no dirigidas elegidas explícitamente. Las distancias infinitas
de componentes desconectados se rechazan, no se sustituyen por constantes arbitrarias.

Los pesos relativos deben ser finitos, no negativos y sumar una masa positiva. Se normalizan
a probabilidades y `total_mass` conserva la suma original. Es **transporte balanceado de
probabilidades**, no comparación de masas absolutas. Se permiten pesos cero, cantidades de
puntos distintas y geometrías con distinta dimensionalidad.

```python
from lambdaforge.distances import FusedGromovWassersteinDistance, StructuredMeasure

a = StructuredMeasure.from_points(
    [[0, 0], [1, 0]], features=[[0, 1], [1, 0]], weights=[1, 3],
)
b = StructuredMeasure.from_points(
    [[0], [1.5], [3]], features=[[0, 1], [0.5, 0.5], [1, 0]], weights=[1, 1, 2],
)
solver = FusedGromovWassersteinDistance(alpha=0.5)
result = solver.solve(a, b)
value = solver(a, b)  # Solo escalar; solve() incluye acoplamiento y diagnóstico.
```

`from_points` construye distancias intrínsecas con SciPy (euclídeas por defecto); no convierte
las coordenadas automáticamente en atributos. `features` se proporciona explícitamente.
El [ejemplo ejecutable](../examples/fgw.py) comprueba también los marginales del transporte.

### Atributos estructurados y distancias precalculadas

Cada atributo puede ser un vector numérico o un objeto arbitrario del proyecto. Para objetos,
use `FusedGromovWassersteinDistance(feature_metric=lambda a, b: ...)`, que recibe un atributo
de cada objeto y devuelve su distancia escalar. Puede operar sobre diccionarios o estructuras
sin que LambdaForge interprete proteínas ni significado científico específico.
`solver.solve(a, b, feature_distances=D)` permite una matriz N×M finita y no negativa.
Son **distancias sin elevar al cuadrado**: FGW q=2 las eleva exactamente una vez. Los nombres de
métricas de SciPy tienen el mismo contrato; `sqeuclidean` produciría potencias cuartas y no suele
ser adecuado. Los vectores numéricos deben ser finitos N×F/M×F, con la misma dimensión F.
`alpha=1` compara solo estructura y no exige atributos; `alpha=0` usa solo atributos.
No se infieren escalado, unidades ni imputación.

### Objetivo y diagnóstico

```text
min_T (1-alpha) Σ_ij d_atributo(i,j)² T_ij
    + alpha Σ_ijkl (C_origen(i,k)-C_destino(j,l))² T_ij T_kl
sujeto a T >= 0, T 1 = origen.weights, Tᵀ 1 = destino.weights.
```

`FGWResult.value` es este objetivo q=2 **sin raíz**. `feature_cost` y `structure_cost` son
componentes sin ponderar; `transport` es el acoplamiento N×M, no etiquetas de clustering.
`alpha` está entre cero y uno. Las unidades cambian el balance; el proyecto debe normalizarlas
explícitamente si procede. `iterations`, `objective_history`, `marginal_error`, `converged`
y `backend` permiten inspeccionar el resultado. Un fallo del solver o marginales inválidos
producen error; alcanzar `max_iter` puede devolver `converged=False`. Compruebe ese indicador.

El cálculo es denso NumPy/CPU y no diferenciable. No implementa la distancia neuronal punto a
punto `lambdaforge.nn.distances.Distance`, clustering, transporte no balanceado ni baricentros.
Para alpha intermedio el problema no es convexo: convergencia local **no** certifica óptimo global
ni una métrica matemática sobre disimilitudes arbitrarias. Distintos inicios pueden producir
distintos mínimos. `initial_coupling` admite un acoplamiento factible N×M explícito. Por defecto
se usa el producto determinista de marginales de POT; en autocomparación, el diagonal.

### Límites y comparación de varios objetos

`StructuredMeasure.from_points(..., max_structure_bytes=256*1024**2)` comprueba el espacio N²
antes de construirlo. `FusedGromovWassersteinDistance(max_memory_bytes=512*1024**2)` comprueba
una estimación conservadora antes de reservar distancias cruzadas, no impone un límite físico
al proceso de POT. El término estructural se contrae sin reservar un tensor N×M×N×M.
Los objetos enormes requieren reducción científicamente justificada en el proyecto, nunca
submuestreo oculto. `solver.pairwise(objects)` devuelve una matriz objeto×objeto, sin clustering.
Por defecto calcula ambas direcciones por separado, pues los mínimos locales pueden diferir.
`symmetric=True` supone explícitamente distancias de atributos simétricas y refleja el triángulo.

Referencias: [Vayer et al., ICML 2019](https://proceedings.mlr.press/v97/titouan19a.html),
[API FGW de POT](https://pythonot.github.io/gen_modules/ot.gromov.html).


<a id="guide-work-results"></a>

## 21. Resultados de Work, evidencia histórica y productos independientes

[English](MANUAL.md#guide-work-results) · [Manual](MANUAL.es.md) · [Productos](#guide-products)

Un Work es una ejecución científica, no necesariamente un entrenamiento. Un Study añade un diseño
de parámetros/semillas y coordinación científica. Comparten runner, Execution/Run/Attempt, outputs
gestionados y las transacciones de exportación/importación. El informe de Work no inventa objetivos,
ajusta modelos estadísticos ni mezcla evidencias de Attempts distintos.

<a id="guide-work-results-inspección-e-informes"></a>

### Inspección e informes

Abre un Work en la consola y entra en **Results**. Selecciona un Run/Attempt para ver Overview, Metrics,
Outputs, Visualizations, Products, Logs, Resources y Provenance. Cada pestaña carga únicamente su evidencia.
Los listados usan índices compactos, sin descargar curvas, modelos ni HTML. Metrics trae curvas acotadas
del Attempt seleccionado; Outputs muestra descriptores, valores y nombres de checkpoints, no sus bytes.
Los checkpoints se identifican como estado actual compartido del Run, no como copias históricas de un Attempt.
Las rutas no conservadas se identifican como tales. Para consultar evidencia en curso debe existir la
atestación persistida del Execution del Job; durante la preparación siguen disponibles los logs y controles.

Provenance incluye una tabla seleccionable de entradas históricas/productos exactos. Enter abre la
fuente registrada si está en el catálogo local, nunca otra ejecución con el mismo nombre. Si falta,
se explica que hay que exportar/importar la evidencia; no se ejecuta su productor.

```bash
lf results list --json
lf results show EXECUTION --view overview --json
lf results show EXECUTION --view metrics --run RUN_ID --attempt 1 --json
lf results show EXECUTION --view outputs --run RUN_ID --json
lf results preview-output EXECUTION OUTPUT_NAME --run RUN_ID --attempt 1 --json
lf results report EXECUTION --output work-results.html
```

En Outputs, Enter solicita explícitamente una previsualización del archivo seleccionado solo si es
regular, está retenido y ocupa como máximo 64 KiB. Verifica tamaño y checksum exactos; el UTF-8 se
muestra como texto inerte, incluido el HTML. Para archivos mayores, directorios o contenido ausente
se muestra metadata y una explicación sin descargar bytes. Las visualizaciones HTML registradas se
ejecutan únicamente en el visor aislado del informe.

**Open HTML report** genera explícitamente un informe offline desde registros persistidos: logs finales
acotados, curvas reducidas, Attempts separados, outputs, errores, configuración y procedencia.
La biblioteca gráfica se incluye una sola vez: cada Run/Attempt tiene su propio grupo de series y ejes
de paso/observación. No se mezclan Attempts ni se atribuyen correlaciones medidas entre Attempts diferentes.
El export conserva además los logs/escalares completos y checkpoints/artefactos retenidos. No se importa el antiguo
código científico para visualizarlo. Un Work sin métricas o HTML también tiene informe. La evidencia
parcial se identifica honestamente y no se reconstruyen artefactos eliminados por compactación.

Para visualizaciones del proyecto usa la API existente:

```python
viewer = self.outputs.html_section("predictions", title="Predictions")
viewer.write_text("<html><body><h1>Vista científica persistida</h1></body></html>")
```

El proyecto implementa su propia lógica. LambdaForge verifica el archivo y lo integra en un iframe
con origen opaco y la misma política de seguridad de Study. El HTML debe ser autocontenido; utiliza
assets incluidos/data URL. Siguen bloqueados las peticiones externas y el acceso privilegiado al sistema
de archivos. Límites: 16 MiB/documento y 64 MiB/informe. Los listados nunca abren esos archivos.

Para arrays completos mayores usa `html_bundle` y su loader lazy acotado. Consulta
[pestañas gestionadas](#guide-managed-reports): API, visor remoto/offline, integridad de export/import
y regeneración desde inferencia retenida sin repetir entreno ni forward passes.

<a id="guide-work-results-seleccionar-dependencias-históricas-exactas"></a>

### Seleccionar dependencias históricas exactas

Los nombres son selectores, no identidades. Repetir un nombre es válido; con varias coincidencias hay
que elegir el Execution exacto. La consola muestra sufijos locales `#n` deterministas; no son referencias
científicas portables. Los imports conservan su identidad/procedencia y nunca se convierten en Jobs.

```bash
lf results reference previous-work --json
lf results reference EXECUTION --run RUN_ID --attempt 1 --json
lf results reference EXECUTION --product selected-models --json
lf results reference EXECUTION --product selected-models --artifact model-1 --json
```

Estos comandos solo consultan metadata y devuelven un marcador tipado con identidades fijadas.
Una ejecución posterior con el mismo nombre no lo modifica. En **Run Work**, introduce el nombre del
parámetro consumidor, pulsa **Choose historical input…**, selecciona una fila exacta y confirma con
**Use exact reference**. Submit valida, explica y prepara esa configuración sin editar el YAML original.
El selector permite fijar además un Run/Attempt concreto y elegir un artefacto registrado del producto.
CLI equivalente: `lf run CONFIG --input-ref 'PARAMETRO=MARCADOR_JSON'`. La selección al lanzar se aplica
a un Work/Study simple; en composición o Fleet las dependencias se declaran en su YAML. Si la entrada
cambia antes de preparar, se rechaza explícitamente en lugar de sustituir el productor.

<a id="guide-work-results-resultinput-evidencia-de-un-execution"></a>

#### ResultInput: evidencia de un Execution

```yaml
with:
  previous:
    result:
      execution: previous-work
      # run: run-...     # obligatorio para leer un Run individual de una ejecución con varios
      # attempt: 1       # opcional; sin él, la identidad del envelope fija el último Attempt
```

El resolver incorpora `evidence_id`, SHA-256 canónico del resultado/configuración originales exactos.
El Work recibe `lambdaforge.work.ResultInput`: metadata/configuration/metrics/result/outputs inmutables,
`metric_curves()` acotado y `artifact(NOMBRE)` con verificación explícita. Es serializable para workers
spawn. Su identidad participa en el fingerprint y la procedencia del consumidor. Acepta snapshots
finalizados nativos/importados, incluidos fallidos, no resultados vivos mutables. Para leer un Run de una
ejecución con varios hay que seleccionarlo. Eliminar su evidencia productora puede invalidar la entrada:
no es un contrato de modelo autónomo.

<a id="guide-work-results-productinput-contenido-científico-duradero"></a>

#### ProductInput: contenido científico duradero

La sintaxis existente sigue siendo válida:

```yaml
with:
  model: {product: {name: selected-models, contract: example/models:v1}}
```

Para un producto publicado de una ejecución histórica concreta:

```yaml
with:
  model:
    product:
      from: {execution: previous-study, output: selected-models}
      artifact: model-1  # miembro registrado exacto opcional, no una ruta física
      expect: {dataset: example-data@1}
```

Solo se puede omitir el contrato cuando el recibo exacto de publicación lo declara inequívocamente.
Sigue validándose, y las expectativas son explícitas. La preparación fija el content ID sellado.
Las referencias directas a productos funcionan incluso tras borrar su productor. Un output temporal
no publicado no es un producto. DatasetInput y `from` entre steps conservan su semántica.

Si declaras `artifact`, `ProductInput.selected_artifact` identifica la selección y `artifact()` verifica
y abre ese miembro exacto. Solicitar otro se rechaza, no se sustituye. Sin selección, la API existente
`artifact(NOMBRE)` no cambia. No es un subproducto inventado: mantiene contenido y contrato del
producto sellado. El nombre lógico seleccionado participa en la identidad del consumidor y se conserva
al fijar referencias, preparar destinos remotos/Fleet y exportar/importar.

<a id="guide-work-results-promoción-explícita-desde-un-work-ordinario"></a>

### Promoción explícita desde un Work ordinario

```yaml
products:
  evaluation-report:
    kind: ScientificReport
    contract: example/evaluation:v1
    scientific_meaning: {protocol: evaluated-on-fixed-test-v1}
    outputs: [report, counts]
```

Tipos ordinarios: ScientificReport, AnalysisResult, ModelArtifact y Selection. El significado es una
declaración del publicador, no una equivalencia demostrada. Enumera valores y archivos retenidos
registrados; los miembros de directorios deben registrarse individualmente. La promoción requiere
un único Run lógico cuyo último Attempt sea exitoso. Para selección entre Runs se conserva ModelSet.

La finalización usa el registro y locks nativos para publicar bytes independientes e inmutables.
Primero se persiste el resultado científico. Un fallo operativo de publicación no modifica su éxito
y conserva los outputs para `lf products finalize EXECUTION --apply`, que repite solo la publicación.
Reintentos concurrentes reutilizan la publicación exacta. Borrar el productor no elimina el producto.

<a id="guide-work-results-portabilidad-y-disponibilidad-remota"></a>

### Portabilidad y disponibilidad remota

```bash
lf export WORK_OR_EXECUTION --output ./exports
lf import ./exports/PAQUETE_EXPORTADO --json
lf import ./exports/PAQUETE_EXPORTADO --apply
```

Se reutiliza el formato nativo versión 2, añadiendo metadata opcional `execution_kind`; los anteriores
paquetes Study siguen siendo legibles. Los originales permanecen en `portable/`, separados de la
ubicación e índices locales. Results y los Works importados abren el mismo visor sin contactar al
cluster original. Reexportar un import verifica y conserva su paquete/procedencia originales.

Resolver una identidad no materializa sus bytes en otro equipo:

```bash
lf products materialize EXACT_CONTENT_ID --on cluster
lf products materialize EXACT_CONTENT_ID --on cluster --apply
lf results materialize EXECUTION --on cluster
lf results materialize EXECUTION --on cluster --apply
```

**Materialize…** ofrece destino, preview y confirmación. Preview solo lee metadata del origen. Apply
transfiere ZIP64 comprimido, reutiliza extracción/importación seguras y verifica contenido/evidencia
exactos; reutiliza copias idénticas existentes sin ejecutar al productor. Cada destino, incluidos los
miembros Fleet soportados, debe disponer previamente de sus dependencias. La preparación Fleet incorpora
las dependencias históricas exactas al estrato nativo de inputs; cada miembro las revalida antes de sus Runs.
Las restricciones existentes para datasets distribuidos y continuación permanecen intactas. Un origen solo remoto
debe exportarse/importarse explícitamente al catálogo controlador; esto no es orquestación automática
entre clusters. El Python destino necesita el runtime actual y espacio de staging/almacenamiento.
El scratch está en el proyecto, no en la partición temporal del sistema. SSH, permisos y conectividad
son requisitos externos.

Consulta [el ejemplo ejecutable](../examples/work_results/README.md): Study → ModelSet seleccionado →
Work de visualización → HTML → export/import, sin repetir entrenamiento.

<a id="guide-work-results-evidencia-en-ampliaciones-de-studies-fijos"></a>

### Evidencia en ampliaciones de Studies fijos

Una serie opt-in referencia Execution/Run/Attempt nativos entre revisiones inmutables; no clona
resultados antiguos dentro del delta. `studies show/analyze/report` y la consola presentan evidencia
acumulada exploratoria post hoc. Borrado/GC protegen referencias. Exportar una serie conserva paquetes
nativos independientes, JSON/HTML acumulado y journal; importar cada paquete sigue produciendo
evidencia de lectura, no una serie ejecutable. Consulta [ampliaciones](#guide-study-extensions).


<a id="guide-products"></a>

## 22. Productos científicos durables: base implementada

Promoción desde Works ordinarios, referencias históricas exactas, ResultInput frente a ProductInput,
selección en consola y materialización comprimida explícita: [Resultados de Work](#guide-work-results).

[English](MANUAL.md#guide-products)

Esta es la capa funcional de **modelo/registro/transporte e inputs Work explícitos**, no el sistema
completo de orquestación de Studies. Hay decisiones nativas, selección local de modelos,
dependencias Work tipadas y registros de consumo real. Un Study nativo individual puede declarar
publicación automática de decisiones/modelos al finalizar. Import/export nativo de Study de un host
incluye productos publicados. La consola incluye un navegador de metadata/auditorías e import de
Study con confirmación. Siguen pendientes espera/replanificación y export Fleet completo.
Véase el [registro de reforma](ai/IMPLEMENTATION.es.md#history-architectural-reform). Ejecución,
recovery y publicación exacta de datasets actuales no cambian.

<a id="guide-products-identidades-y-contratos"></a>

### Identidades y contratos

`lambdaforge.products` exporta `ProductContract`, `ProductArtifact`, `StudyProduct`,
`ProductRegistry`, `ProductBundle`, `SelectionPolicy`, `ModelSelection`, `select_models`,
`build_study_decision`, `ProductRequirement`, `ProductInput`, `ProductPublication` y
`publish_declared_products`. Un producto separa:

- contrato explícito versionado, como `project/report:v1`, con campos científicos exactos;
- `scientific_id`: kind, declaración del contrato y significados científicos declarados;
- `content_id`: payload/significado canónico y manifiestos exactos de artifacts;
- provenance original inmutable y attestations independientes de productores posteriores;
- ubicación física del catálogo, que no cambia esas identidades.

La configuración completa del productor puede formar parte de provenance, **no** de la clave de
compatibilidad del consumidor. Los nombres son aliases inmutables. Cambiar la declaración de un
contrato requiere otra versión. Objetivo científico, identidad/etiquetas de inputs, diseño y regla
de selección relevantes deben estar declarados. No se adivinan exclusiones dentro de NPZ,
checkpoints ni formatos arbitrarios. Igual significado declarado no certifica equivalencia entre
bytes distintos.

Cada artifact es un fichero regular independiente con SHA-256 crudo y tamaño. Publicación copia y
verifica antes de commit, sin enlazar a Attempts desechables, borrar originales ni recorrer archivos
no seleccionados. Cada metadata tiene límite de 512 KiB. Listar/mostrar no abre artifacts pesados;
`verify` es explícito. Lecturas ausentes y previews no crean carpetas ni locks.

<a id="guide-products-api-de-publicación-explícita"></a>

### API de publicación explícita

Usarla desde un script de operador/postprocesado, no como runner alternativo oculto. Declaración
nativa y selección post-Study explícita de modelos
ya está disponible abajo.

```python
from lambdaforge.products import ProductContract, ProductRegistry, StudyProduct

contract = ProductContract("my_project/count-report:v1", ("sources", "labels", "method"))
report = StudyProduct(
    name="benchmark-counts-v1",
    kind="ScientificReport",
    contract=contract,
    payload={"train": 317, "validation": 272},
    scientific_meaning={
        "sources": {"dataset": "sha256:<contenido-publicado-exacto>"},
        "labels": {"positive": "binding", "partitions": "audited-v1"},
        "method": "count-audited-members-v1",
    },
    producer={
        "execution_id": "<execution-original-real>",
        "evidence_fingerprint": "<fingerprint-real-de-evidencia-persistida>",
    },
)
registry = ProductRegistry()  # Proyecto pyproject actual/.lambdaforge/products
plan = registry.publish(report)  # Preview sin publicar.
registry.publish(report, apply=True)
resolved = registry.resolve(
    "benchmark-counts-v1", contract=contract,
    scientific_expectations={"method": "count-audited-members-v1"},
)
```

Sustituir los placeholders por evidencia real; el envelope genérico no calcula ni certifica esa
evidencia. Etiquetas de kind como `StudyDecision`/`ModelSet` no seleccionan modelos ni demuestran la
semántica del payload automáticamente. El proyecto es dueño de su declaración explícita.

Para artifacts, pasar descriptores `ProductArtifact(nombre, ruta_relativa, sha256_crudo, tamaño, rol)`
y `files={nombre_artifact: ruta_origen}` a `publish`. `artifact_path(selector, nombre)` verifica los
bytes promovidos por defecto antes de devolver la ruta. No sustituir el SHA-256 crudo por el
fingerprint histórico de ficheros gestionados, que también incorpora su nombre.

<a id="guide-products-cli-nativa"></a>

### CLI nativa

```bash
lf products list --json
lf products show benchmark-counts-v1 --json
lf products provenance benchmark-counts-v1 --json
lf products consumers benchmark-counts-v1 --json
lf products verify benchmark-counts-v1 --json
```

Todas aceptan `--root DIR`; por defecto catálogo del proyecto actual o `LAMBDAFORGE_PRODUCT_ROOT`.
List/provenance/consumers permiten paginar con `--offset`/`--limit`. No adivinar rutas de Jobs remotos.
El control plane existente propaga a workers una raíz de productos propia del proyecto, separada
de Jobs y cachés desechables: `STATE_ROOT/products` remoto y `PROJECT/.lambdaforge/products` local.
Raíces remotas dentro de cache/Jobs se rechazan.

Un operador puede serializar `product.to_dict()` y publicarlo mediante
`lf products publish MANIFEST.json --file ARTIFACT=SOURCE [--apply]`. Es declaración avanzada
explícita, no selección automática ni workaround específico de aplicación. Preview verifica
forma/tamaño del origen; apply verifica bytes exactos. Un origen corrupto nunca se publica como
válido. Un nombre inmutable conflictivo exige otro nombre/versión, nunca un flag de sobrescritura.

<a id="guide-products-selección-post-study-de-modelset"></a>

### Selección post-Study de ModelSet

Registrar un snapshot evaluado desde el ciclo de vida Work normal:

```python
self.outputs.from_checkpoint(
    "model", "best.ckpt", role="model",
    metadata={"metrics": {"auprc": measured_auprc, "accuracy": measured_accuracy}, "step": epoch},
)
```

El proyecto aporta valores evaluados para **esos pesos exactos**, no la última época de otro modelo.
Es un artifact independiente normal del Attempt. Promoverlo al catálogo durable es deliberado.
También admite `outputs.file` con rol model y metadata explícita equivalente. El selector local
todavía no admite modelos directorio ni artifacts publicados externamente.

Crear un YAML normal de selección, por ejemplo `selection.yaml`:

```yaml
artifact: model
rank_by: auprc
mode: max
group_by: [hidden_dim]  # Omitir para top_k global.
top_k: 1
constraints:
  accuracy: {min: 0.6}  # Mismo snapshot registrado, no últimas métricas de la Run.
tie_policy: stable
```

```bash
lf products select EXECUTION_ID --name best-per-width --contract my_project/models:v1 \
  --policy selection.yaml
# Revisar y repetir con --apply.
```

`--results-root DIR` selecciona ResultStore local y `--root DIR` el catálogo de productos. La API
equivalente es `select_models(source, execution_dir, policy, name=..., contract=...)`. Usa últimos
Attempts, excluye failed/pruned/fidelidad parcial, aplica constraints del snapshot y ordena cada
grupo condicional exacto determinísticamente. Parámetro inactivo y valor null explícito son distintos.
`tie_policy: include_equivalent` incluye opcionalmente modelos dentro de un `practical_margin`
autorizado desde el límite top-k; no expresa confianza estadística.

Lee/calcula hash **solo de modelos seleccionados**; incluso preview puede tener E/S local importante
si sus pesos son grandes, pero no crea carpetas/locks. Apply promueve copias exactas independientes.
Evidencia e historial originales no cambian. ModelSet registra Runs lógicas, Attempts, seeds,
parámetros, métricas/step del snapshot, grupos, razón, checksums y provenance. Una Run auxiliar
fallida no invalida snapshots elegibles correctos. Ranking es **entre snapshots observados
elegibles**, no óptimo global, confirmación nueva ni nueva decisión HPO. Si faltan métricas asociadas
al modelo, explica cómo registrarlas; nunca atribuye best/last de una Run a pesos arbitrarios.

Siguen pendientes selección remota/Fleet y publicación dentro de `steps` compuestos.
El selector local no busca modelos fuera de la Execution propia verificada.

<a id="guide-products-studydecision-nativa"></a>

### StudyDecision nativa

```bash
lf products decide EXECUTION_ID --name pooling-decision --contract my_project/pooling:v1
# Revisar la decisión nativa y repetir con --apply.
```

Sella la selección nativa del Study finalizado, objetivo/diseño/significado de inputs, referencias
Run/Attempt y fingerprint de evidencia. No vuelve a ordenar candidatos ni ajusta HPO. Selección y
resolución científica son distintas: si falta Analysis final quedan preguntas explícitamente
pendientes; confirmación incompleta no permite una selección basada solo en supervivientes.
Una Run auxiliar fallida no invalida una selección elegible completa.

Si existe `analysis.json`, debe coincidir con Execution/evidencia finalizadas: conclusiones
provisionales, antiguas o de otra Execution se rechazan y se indica cómo recalcular explícitamente.
Esta operación de metadata no lee historiales escalares, checkpoints ni pesos. APIs equivalentes:
`ResultStore.decision(...)` y `build_study_decision(...)`.

<a id="guide-products-publicación-declarada-y-retry-de-publicación-sin-entrenar"></a>

### Publicación declarada y retry de publicación sin entrenar

Un Study individual (`search`, `sweep` o `seeds` repetidas) puede declarar productos junto a los
campos normales del Work:

```yaml
products:
  pooling-decision:
    kind: StudyDecision
    contract: my_project/pooling:v1
  best-per-width:
    kind: ModelSet
    contract: my_project/models:v1
    select:
      artifact: model
      rank_by: auprc
      mode: max
      group_by: [hidden_dim]
      top_k: 1
```

Validación comprueba esta declaración cerrada antes del cálculo. `lf explain` la muestra y la
Execution registra productos esperados antes de iniciar Runs. Finalización guarda primero evidencia
de entrenamiento, después publica mediante los mismos servicios nativos antes de compactar.
StudyDecision requiere objetivo; ModelSet requiere artifacts evaluados explícitamente. No hay
constructores arbitrarios, callbacks ejecutables ni otro runner.

Un fallo de publicación conserva resultados/checkpoints. `products.json` registra pending/published/
failed con causa/fase; `product-publication-history.jsonl` conserva intentos de publicación. No se
transforma en un fallo científico ni se alteran objetivos. Consultar publicación por separado:

```bash
lf products status EXECUTION_ID --json       # Metadata acotada; no lee pesos.
lf products finalize EXECUTION_ID --json     # Preview sin mutación.
lf products finalize EXECUTION_ID --apply    # Solo publicación; no ejecuta Runs.
```

Apply adquiere lock de propiedad de Execution y después locks de publicación/catálogo. Recibos
correctos se reutilizan aunque ya no estén los artifacts originales; apply verifica los bytes
promovidos. Repetición concurrente idempotente; corrupción/cambio de declaración se rechazan.
Conflictos de nombre inmutable requieren otro nombre/versión, no sobrescribir. `--results-root` y
`--root` seleccionan evidencia/catálogo locales. La CLI todavía no envía recovery de publicación a
remoto: ejecutar en el host dueño de Execution/catálogo. Promoción automática Fleet no implementada.

<a id="guide-products-dependencias-work-tipadas"></a>

### Dependencias Work tipadas

Declarar el producto como argumento normal de Work, no como condición sobre el estado del productor:

```yaml
name: model-visualization
run: my_project.Visualize
with:
  models:
    product:
      name: best-per-width
      contract: my_project/models:v1
```

El proyecto puede añadir `expect: {CAMPO: VALOR_EXACTO}` para exigir significados científicos
explícitos. El resolver comprueba contrato/expectativas y entrega un `ProductInput` pickle-safe:

```python
import lambdaforge as lf
from lambdaforge.products import ProductInput

class Visualize(lf.Work):
    def run(self, models: ProductInput) -> dict[str, int]:
        for model in models.payload["models"]:
            weights = models.artifact(model["artifact"])
            # Leer los pesos verificados y registrar outputs de visualización gestionados normales.
        return {"models": len(models.payload["models"])}
```

Consultar metadata no lee/descarga pesos. `artifact(nombre)` verifica ese fichero promovido exacto
antes de entregar su ruta de input de solo lectura. Compatibilidad e identidad del input usan
contrato/significado/contenido sellados, no hash completo de YAML ni estado actual del productor.
Validación/preview no crean nada; Attempts ejecutados añaden registros inmutables de consumidores
consultables con `lf products consumers`. Reutilizar una Execution cacheada no inventa otro consumo.

Bundles remotos fijan aliases por identidad de contenido y verifican materialización en el catálogo
del proyecto destino antes de staging. Importar allí el bundle explícito primero: no se transfieren
pesos grandes implícitamente ni se usa un catálogo dentro del Job como fallback. Productos ausentes/
incompatibles fallan con diagnóstico; espera automática o replanificación del productor siguen pendientes.

<a id="guide-products-exportimport-portable-de-productos"></a>

### Export/import portable de productos

```bash
lf products export benchmark-counts-v1 --output ./report-export
lf products export benchmark-counts-v1 --output ./report-export --apply
lf products import ./report-export --root ./independent-products
lf products import ./report-export --root ./independent-products --apply
```

Transportan **un producto**, no el Study entero. Export escribe un directorio verificado con
`bundle.json`, `product.json`, bytes seleccionados y attestations completas del productor (no los
registros de consumo downstream). Destinos existentes
quedan protegidos. Import comprueba SHA/identidad del manifiesto, bytes de artifacts e inventario de
provenance; conserva productor original y nunca lanza cómputo. Repetir/importar concurrentemente
el mismo producto es idempotente. Un origen corrupto se rechaza incluso si el destino ya lo tiene.

Publicación/import se serializan con el lock nativo existente. Copias interrumpidas limpian solo su
directorio exacto de transacción no publicado, conservando originales. Un crash tras commit del
objeto pero antes del alias puede dejar un objeto inmutable sin registrar: repetir verifica y
completa el alias, sin iniciar otro experimento.

Un producto portable sobrevive al borrado de su Study original. Esto no implementa DAG de dependencias,
autenticidad criptográfica de provenance, certificados de datasets ni export completo de Fleet.

<a id="guide-products-importar-un-study-portable-sin-ejecutarlo"></a>

### Importar un Study portable sin ejecutarlo

```bash
lf import ./portable-study--execution-ID --json
lf import ./portable-study--execution-ID --apply --json
lf results list --json
lf products show models --json
```

Por defecto verifica/preview; `--apply` registra evidencia en ResultStore del proyecto y productos
sellados en ProductRegistry nativo. `--results-root`/`--products-root` cambian opcionalmente ubicación,
nunca identidad científica. Admite exports nativos versión 2: finales, fallidos/cancelados y snapshots
en ejecución conservan estado capturado. Un snapshot no pasa a ser evidencia completada. Paquetes
de Job previo a ejecución sin Execution propio no se importan como Study.

Verifica todos los archivos, SHA-256/tamaño/cantidad, ausencia de extras, versiones de metadata,
identidad/ownership de Execution/Run/Attempt y productos/atestaciones del productor. Rechaza traversal,
contenido simbólico, especial o hard-linked, inventarios duplicados y corrupción entrante/existente.
Imports concurrentes son serializados/idempotentes. Nunca sobrescribe Execution nativo ni otro
snapshot con igual ID: usar otra raíz de resultados para conservar varias capturas. Checksums no
demuestran autenticidad criptográfica de afirmaciones científicas de un autor no confiable.

Archivos/provenance originales quedan intactos en `portable/`; `import.json` registra nueva ubicación.
ResultStore lista la entrada, relocaliza rutas lógicas solo en su lector de logs y lee Analysis incluido
sin refit. No importa código del proyecto, ejecuta, continúa epochs ni abre HTML automáticamente.
Configuración importada no sirve como recovery nativo. `portable/` ya es el paquete reutilizable;
no necesita generar otra procedencia de export.

En la consola, **Studies → Import Study…** (también desde Products y Ctrl+P) permite seleccionar
una carpeta, verificarla en un worker, revisar su identidad y confirmar el registro. Aparece como
snapshot read-only en Studies/Overview del proyecto actual. Trials/seeds usan proyecciones locales
jerárquicas y el dashboard compartido de Run; métricas/logs/artifacts conservados se leen localmente,
no del cluster original. Un snapshot running no es un controlador vivo. Cancel, retry, borrado de
historial y export del productor quedan ocultos; eliminar el archivo importado sigue siendo una
acción explícita de Results. Los HTML incluidos siguen disponibles en `portable/reports/`.

El agregado nativo `execution/result.json` puede superar 64 MiB en un Study válido grande.
Import verifica primero tamaño declarado y SHA-256 exactos de todos los archivos y después valida
estrictamente el agregado sin aplicarle el límite de metadata individual. Siguen intactas las
comprobaciones de checksum, rutas/propiedad, claves duplicadas y valores no finitos. El refresco
raíz solo lee `import-study.json` pequeños, no el agregado. Reaplica un import antiguo para crear
`import-view/` bajo su lock: registros científicos y recibo original permanecen intactos.
Import copia los bytes conservados y necesita espacio local para el paquete y su índice;
nunca vuelve a descargar artifacts.

Export incluye productos de la declaración/recibo nativo, sus bytes independientes e historial del
productor. Transferencia trae solo esos objetos, no todo el catálogo; muestreo de telemetría nunca
altera bytes de productos. Ausencias fallan explícitamente. Publicaciones de operador no declaradas
usan `products export`, sin recorrer catálogos ajenos para descubrirlas.

Registro del archivo hace commit atómico antes de aliases de productos con locks independientes.
Un fallo posterior deja el archivo verificado disponible: repetir import completa aliases. No es
transacción distribuida entre stores. Borrar evidencia importada conserva productos independientes.


<a id="guide-conditional-studies"></a>

## 23. Estudios científicos condicionales

[English](MANUAL.md#guide-conditional-studies) · [HTML de investigación](#guide-research-analysis)

<a id="guide-conditional-studies-1-una-pregunta-un-study"></a>

### 1. Una pregunta, un Study

Usa un único `sweep.space` para comparar alternativas que tienen distintos parámetros aplicables.
Todas las celdas pertenecen al mismo Study y comparten las seeds declaradas. Usa `steps` para un
flujo real de trabajo, no para separar artificialmente ramas comparables: siguen siendo
invocaciones independientes, no estudios fusionados.

```yaml
name: optimizer-comparison
run: my_project.Compare
seeds: [4, 7, 32, 54]
sweep:
  reference: {optimizer: baseline}
  space:
    momentum:
      values: [0.8, 0.9]
      when: {optimizer: {in: [sgd, accelerated]}}
    optimizer: [baseline, sgd, accelerated]
resources: {gpu: 2, time: 24h}
execution: {max_parallel: 4, max_time: 12h}
```

Son cinco candidatos y veinte Runs obligatorios, no doce candidatos ni un Study por optimizador.
`baseline` no recibe `momentum`; se aplica el valor por defecto de la firma Python. La referencia
debe seleccionar exactamente un candidato activo. Cero coincidencias o varias son errores.

<a id="guide-conditional-studies-2-gramática-de-activación"></a>

### 2. Gramática de activación

Un descriptor admite `when` opcional. Varios padres se combinan con AND:

```yaml
when:
  optimizer: {in: [sgd, accelerated]}
  use_momentum: true
```

`{parent: valor}` conserva la igualdad antigua; `{parent: {eq: valor}}` significa lo mismo. La
pertenencia es explícita: `{parent: {in: [valor1, valor2]}}`. La lista debe contener escalares JSON
únicos, no estar vacía y pertenecer al dominio finito del padre. Se rechazan listas sin operador,
operadores desconocidos, padres inexistentes, ciclos y valores ajenos al dominio. No hay
expresiones, `or`, `not` ni lenguaje recursivo. La igualdad antigua sobre rangos numéricos sigue
funcionando; `in` requiere un dominio finito.

Los hijos pueden escribirse antes que sus padres. `ActivationCondition` y `ParameterSpace`
comparten validación y ordenación de dependencias entre sweeps, generación aleatoria/Sobol/
adaptativa, codificación y análisis. Las claves inactivas se omiten: no son `null` ni multiplican
el diseño para deduplicarlo después. Se rechazan valores repetidos del sweep. El dominio mantiene
el orden declarado; ordenar de otra forma los miembros de `in` no cambia la identidad científica.
La cobertura condicional usa valores observados activos, no una categoría inactiva inventada.

<a id="guide-conditional-studies-3-validar-y-entender-el-plan"></a>

### 3. Validar y entender el plan

```bash
lf validate comparison.yaml
lf explain comparison.yaml
lf run comparison.yaml --on CLUSTER --dry-run --json
```

Validación y explicación incluyen `preflight` acotado: candidatos, seeds compartidas, Runs
obligatorios, conteos de la primera rama categórica independiente, referencia, GPUs solicitadas,
límite paralelo, presupuesto del Study y tiempo del scheduler. La salida humana no enumera cientos
de identidades. `--json` conserva el plan completo para automatización. La validación local indica
que no se ha comprobado la capacidad del destino; un plan remoto de solo lectura también puede
necesitar un entorno/runtime ya preparado. El resumen de ramas no es una API de familias de dominio.
En búsqueda adaptativa, los conteos describen la ventana de propuestas actual (limitada por el
techo de candidatos), no prometen ejecutar todas las propuestas. La replicación se vuelve
obligatoria al proponer el candidato. En un sweep fijo, los conteos cubren todo el diseño declarado.

Para combinaciones extra selectivas, `sweep.include` añade puntos escalares activos completos sin
ampliar los productos de los dominios base. `sweep.candidates` selecciona solo puntos explícitos
dentro del espacio. Ambos usan la misma autoridad de activación, rechazando claves inactivas o
ausentes. Un punto include ya presente en la base no se duplica. Ejemplos y ampliación incremental
de Studies completados: [ampliaciones](#guide-study-extensions). Editar el YAML por sí solo no amplía
una Execution existente: revisa y confirma **Extend Study…**, no Resume.

Solo `resources` raíz se hereda por los pasos salvo sobrescritura. En una composición con `steps`
se rechazan `with`, `seeds`, `replicates`, `search`, `sweep`, `execution`, `objective` y `analysis`
raíz: deben declararse en el paso correspondiente. Nunca se ignora silenciosamente una política.
Preflight comprueba herencia y firmas sin construir un Work ni ejecutar ciencia; no puede detectar
errores semánticos arbitrarios dentro del `run()` del proyecto consumidor.

<a id="guide-conditional-studies-4-tiempo-y-capacidad-del-destino"></a>

### 4. Tiempo y capacidad del destino

`resources.time` es el techo de tiempo del scheduler para la unidad planificable. Los niveles
secuenciales suman; un nivel paralelo aporta el máximo de sus miembros. Seis pasos secuenciales
de 168 h solicitan 1008 h (42 días); tres de 24 h, 72 h. Es un techo, no una predicción del tiempo
de entrenamiento. `execution.max_time` es el presupuesto lógico de despacho del Study; se informa
por separado y no sustituye al límite del scheduler.

En un host directo, un inventario UUID vivo y correcto junto con la restricción heredada de
visibilidad, cuando sea inequívoca, permite rechazar una solicitud superior a la capacidad antes
de preparar el bundle, también en dry-run. Ocupación no es capacidad. Una sonda fallida/ambigua
queda como desconocida, nunca inventa un número. El nodo de login de un scheduler no es su
inventario de asignaciones. Los launchers del centro conservan la comprobación existente en
runtime: un Study adaptativo puede reducirse a menos tokens otorgados, nunca ampliar el grant ni
reservar otras GPUs durante preflight. La validación local no contacta destinos remotos.

<a id="guide-conditional-studies-5-identidad-y-evidencia-existente"></a>

### 5. Identidad y evidencia existente

La igualdad normalizada mantiene serialización e identidad antiguas. `in` amplía el schema actual
sin nueva versión YAML ni runner de compatibilidad. Los predicados permanecen en diseño e
inicialización del Study y viajan por análisis, exportación y retry. Los registros históricos de
igualdad se leen, no se reescriben. Cambiar condiciones, dominios, seeds u objetivo sigue siendo
un cambio científico, no una excepción automática a las garantías de retry. Un sweep fijo sigue
ejecutando cada celda/seed obligatoria sin poda adaptativa ni replanteamiento científico.

La prueba genérica `tests/work/test_conditional_membership.py` produce ramas de 1, 2, 10, 16, 9 y
18 candidatos: 56 × cuatro seeds = 224 Runs obligatorios. Compara el diseño unido con el conjunto
previsto de ramas independientes, sin importar ni modificar ningún proyecto consumidor.


<a id="guide-study-extensions"></a>

## 24. Ampliaciones incrementales de Studies fijos

[English](MANUAL.md#guide-study-extensions)

Edita el **YAML original** y pulsa **Extend Study… → Review changes → Extend Study**.
El diálogo muestra Runs conservados, celdas nuevas, destino, revisión y recuentos. Tras confirmar
se envía solo el delta y se abre la serie acumulada. No necesitas escribir otro YAML.

El origen debe ser un **sweep fijo propio, finalizado correctamente y con seeds explícitas**,
también condicional. Amplía sus candidatos **o** añade seeds; completa cada revisión antes de
ampliar otra vez. En un diseño rectangular sin puntos explícitos, añade valores a una dimensión
existente cada vez. En diseños condicionales/selectivos se comparan candidatos activos exactos.
Reordenar/insertar se admite. Quitar candidatos, cambiar ciencia o añadir valores
y seeds simultáneamente se rechaza. Una reordenación sin celdas nuevas es un no-op.

<a id="guide-study-extensions-cli"></a>

### CLI

```bash
# Remoto: Job exacto del origen; se deduce su cluster.
lf studies extend job-ID_EXACTO --config experiments/regularization.yaml --dry-run
lf studies extend job-ID_EXACTO --config experiments/regularization.yaml --dry-run --json
lf studies extend job-ID_EXACTO --config experiments/regularization.yaml --apply --plan-hash sha256:PLAN_REVISADO

# Opcional: exigir equivalencia en lugar de crear una revisión de código/entorno.
lf studies extend job-ID_EXACTO --config experiments/regularization.yaml --dry-run --code-policy strict

# Origen local directo: también admite Execution exacta.
lf studies extend execution-ID_EXACTO --config experiments/regularization.yaml --dry-run

lf studies show series-ID_EXACTO --json
lf studies refresh series-ID_EXACTO
lf studies history series-ID_EXACTO
lf studies analyze series-ID_EXACTO --json
lf studies report series-ID_EXACTO --output regularization.html
lf studies export series-ID_EXACTO --destination ./exports
lf studies extend series-ID_EXACTO --config experiments/regularization.yaml --dry-run
```

Sin `--dry-run` también se hace solo preview. Este no escribe revisiones, reserva Jobs, consume
seeds ni descarga modelos/checkpoints/artefactos/historial de epochs. Verificar inputs tipados puede
calcular hashes de su contenido. Validar importa la Work real; una sesión de agente READ/PLAN no
autoriza esa importación por sí sola. No se expone apply libre por MCP.

<a id="guide-study-extensions-ejemplo-de-regularización"></a>

### Ejemplo de regularización

Para λ utiliza un argumento Python válido como `lambda_`:

```yaml
name: regularization
run: project.Training
seeds: [11, 23, 47, 59, 71]
sweep:
  space:
    lambda_: [0.0, 0.1, 0.2, 0.3]
objective: {metric: score, mode: max}
```

Tras 20 Runs correctos, cambia esa lista a `[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`:
**20 conservados + 15 nuevos = 35**. Después añade tres seeds:
**35 conservados + 21 nuevos = 56**. Añadir tres seeds directamente al diseño inicial produce
**20 conservados + 12 nuevos = 32**.

SubmissionService ejecuta un delta nativo exacto, rectangular o disperso, no otro runner ni recovery.
LF conserva un snapshot de solo lectura `.lf-extension-UUID.yaml` junto al YAML original para
mantener rutas relativas/contexto. No lo edites ni borres. Preparación asíncrona fija sus bytes
e identidad científica real; recientes sigue mostrando el YAML authored.

<a id="guide-study-extensions-ampliaciones-condicionales-y-selectivas"></a>

### Ampliaciones condicionales y selectivas

Añadir valores a una dimensión condicional existente amplía solo sus ramas activas. Para añadir
valores únicamente a determinadas combinaciones, **mantén el espacio base intacto** y añade
puntos completos mediante `sweep.include` en el mismo YAML:

```yaml
sweep:
  reference: {pooling: max}
  space:
    pooling: [max, mean, lse]
    heat: {values: [false, true], when: {pooling: {in: [mean, lse]}}}
    lambda_: {values: [0.0, 0.1], when: {pooling: {in: [mean, lse]}}}
  include:
    - {pooling: lse, heat: true, lambda_: 0.2}
    - {pooling: mean, heat: false, lambda_: 0.3}
```

Esto añade exactamente dos candidatos, no todas las combinaciones con λ=0.2/0.3. Con cuatro seeds
sin cambios son ocho Runs nuevos. Usa los nombres reales de argumentos de tu Work. Cada punto debe
incluir **todas las dimensiones activas y ninguna inactiva**: el control `max` no tiene `heat`
ni `lambda_`. Include es una unión: un punto ya presente en la base no se ejecuta dos veces;
entradas duplicadas dentro de `include` se rechazan. Los valores son escalares JSON finitos, no
listas ni abreviaturas de dominios por rama. Conserva los include anteriores en próximas revisiones.

La opción avanzada `sweep.candidates` ejecuta **solo** los puntos completos enumerados dentro del
`space` declarado, sin recorrer su producto. Es incompatible con `include`. El delta disperso
nativo usa esta forma para ejecutar la diferencia exacta, sin añadir controles, productos cruzados,
dimensiones ficticias, nulls para claves ausentes ni seeds extra. Space sigue siendo la geometría
condicional declarada; no implica que todos los puntos posibles se hayan medido o sean obligatorios.
Quitar candidatos anteriores o cambiar condiciones sigue rechazándose. Añade seeds en otra revisión.
Nunca se reescriben identidades ni registros históricos.

Un parser conservado debe admitir candidatos dispersos. Por defecto, si falta o es incompatible
en un cluster gestionado, se propone desplegar normalmente el proyecto/framework actuales como
una nueva revisión de entorno. No se modifica el prefijo antiguo. Con `--code-policy strict`
se exige compatibilidad del entorno conservado y se bloquea antes de enviar si no se acredita.

<a id="guide-study-extensions-identidad-compatibilidad-y-evidencia"></a>

### Identidad, compatibilidad y evidencia

Serie, revisión, Execution, Job, Run y Attempt tienen identidades distintas. La evidencia antigua
se referencia: no se copia a otra Execution ni se renumera en disco. Candidato se identifica por
parámetros normalizados exactos; `true`, `1` y `1.0` no se confunden. Se conservan Trial original,
Execution/Run/Attempt, seed, identidad de inputs y checksum. No cambian CodeIdentity ni IDs globales.

La política predeterminada `--code-policy revisioned` no exige que el repositorio original estuviera
limpio. Identidad exacta o prueba de HEAD limpio idéntico con solo el YAML cambiado permiten
análisis compatible acumulado. En otro caso el plan propone explícitamente **el código actual como
nueva revisión de código/entorno**. Cambios de código, documentación, dependencias, HEAD u origen
dirty no impiden por sí solos añadir celdas. La consola lo avisa y confirma con **Extend · current
code**. Aprobar ese plan autoriza la nueva revisión, no demuestra equivalencia de código.

Los Runs antiguos conservan código, Execution, Run y Attempt originales. Los nuevos reciben su
propia identidad científica mediante despliegue nativo normal; no se atribuyen al código antiguo.
El código e inputs de preparación revisados se fijan y comprueban al crear el snapshot y en el
worker desacoplado. Un cambio entre revisión y envío exige revisar de nuevo. Solo se excluyen del
pin snapshots internos propios y acreditados por checksum.
Tras preparar remotamente, LF valida la Work **realmente desplegada**, contenido de inputs,
candidatos/seeds exactos, objetivo, semántica de métricas y diseño antes de enviarla al proveedor.
Registra su identidad científica nativa del host separada del pin de envío del controlador;
el worker vuelve a comprobar checksums de configuración e identidad de código embebida.
Recogida/retry usan esa identidad Execution acreditada, no rutas ni namespace del controlador.
Los lectores de metadatos usan el Python científico registrado, incluidos los preflights nativos.
El argv SSH solo transporta compromisos SHA-256 canónicos por cada campo completo del contrato;
el host los recalcula desde la configuración nativa. Descripciones extensas de métricas no generan
comandos enormes. No se truncan/ignoran metadatos, inputs, seeds ni campos del diseño; un cambio
sigue identificando los campos afectados. No cambia el algoritmo de identidad científica nativo.
Rutas inseguras y evidencia científica
corrupta/ausente siguen bloqueando; Work fuera del proyecto no está admitida. La política
`--code-policy strict` conserva la exigencia anterior de equivalencia.

Se reutilizan checks nativos de configuración/Run para conservar clase, parámetros fijos, contenido
exacto de inputs/datasets/productos, objetivo/dirección, semántica de métricas, seeds y contratos de
publicación. Agregado y últimos Attempts autoritativos deben coincidir. Celdas requeridas
failed/pruned/incompletas se reparan antes con retry; no se convierten en evidencia completa.

El orden histórico de Trials se reconstruye desde el snapshot `authored.yaml` propio de la
Execution, no desde el YAML original ya editado. Sus valores JSON exactos deben coincidir con
`configuration.json`; se mantienen huella científica y checks por Attempt. Así se evita que ordenar
alfabéticamente las claves JSON cambie los ordinales de un sweep condicional/cartesiano. El transporte
paginado conserva ese orden explícitamente. Evidencia de orden ausente/corrupta/insegura no exime
de verificar identidad. Reconstruir el orden no convierte versiones científicas distintas en
equivalentes.

<a id="guide-study-extensions-ejecución-admitida"></a>

### Ejecución admitida

- CPU local: mismo intérprete existente y contrato registrado de Python/framework/paquetes/
  plataforma; sin cambiar entorno ni command prefix.
- CPU/GPU en un cluster: mismo destino y **entorno inmutable nativo gestionado original conservado**.
  Se comprueban en el host prefijo propio, identidad del entorno, marcador listo y Python/paquetes/
  plataforma. No se resuelven otras dependencias. Un preflight pequeño llama al CLI/WorkRunner
  intacto dentro del scheduler/lanzador habitual. Antes de ciencia comprueba tipo/modelo, capacidad
  y memoria de GPU concedida y entorno numérico. Ordinales/cantidad de GPUs no son identidad de
  hardware. Admisión/grants siguen siendo nativos; nunca se amplía CUDA visibility.
- Con código no equivalente o runtime conservado incompatible: el mismo cluster **gestionado**
  usa bundle, resolución de runtime, entorno, scheduler y grants nativos del proyecto actual.
  Preview no construye/instala/envía nada. No se reescribe el entorno viejo ni se usa silenciosamente
  para ejecutar código distinto. Entornos remotos existing/no gestionados no despliegan nuevo código.

No se acredita cada byte de binarios arbitrarios instalados. El modo estricto rechaza otro framework,
entorno o tipo de GPU conservados. Entornos remotos no gestionados, ampliación GPU local, migración,
Fleet, HPO adaptativo, replicación secuencial automática, dimensiones estructuradas y
Works compuestos quedan fuera del contrato inicial; no se simulan mediante reinicio completo.

<a id="guide-study-extensions-consola-y-análisis-acumulado"></a>

### Consola y análisis acumulado

El catálogo científico agrupa revisiones opt-in bajo una serie, con Runs/métricas/gráficas acumulados,
revisiones, Jobs nativos e historial. Cada Job mantiene logs, cancelación y Resume. Los Jobs físicos
siguen en Advanced Jobs y `lf jobs`; no cambian Studies normales ni selectores existentes.

`StudyExtensionService.overview()` proyecta metadatos y `.analysis()` reutiliza StudyAnalysis,
seeds emparejadas y fingerprint acumulado independiente, sin sobrescribir el análisis original.
Elegir candidatos tras ver resultados genera evidencia **exploratoria post hoc**, no confirmación
retroactiva. No se muestran medias finales de supervivientes cuando faltan seeds. Diagnósticos
nativos de cobertura/extremos no demuestran óptimos fuera de los datos ni autorizan ampliaciones
automáticas. No hay otro motor estadístico.

Si alguna revisión no tiene equivalencia acreditada de código/runtime, sigue habiendo una única
serie navegable, pero el análisis se **separa por revisión**. Por defecto se analiza la última;
JSON conserva `code_cohort_analyses` y el HTML añade pestañas **Code · revision-N**. Se mantienen
semánticas originales de métricas. No se mezclan tests, intervalos, efectos de parámetros ni medias
de seeds entre revisiones. Un candidato con seeds de dos códigos no tiene una media final combinada
ficticia; sus Runs siguen correctos y sus métricas visibles. Logs, export y retry nativos siguen
disponibles. Evidencia histórica de un parámetro/seed no demuestra su resultado bajo código nuevo.
Para evaluar *todo el diseño* con código nuevo, envía un Study nativo completo separado: Extend
ejecuta deliberadamente solo las celdas adicionales. Rigen los límites nativos de tamaño de HTML.

<a id="guide-study-extensions-presupuesto-fallos-y-durabilidad"></a>

### Presupuesto, fallos y durabilidad

Attempts/tiempo históricos de todas las revisiones, incluidos fallos reparados, no se devuelven.
El límite global deja su remanente al delta; obligaciones imposibles se rechazan. Cambiar el
ganador no promociona automáticamente productos.

El journal propio `.lambdaforge/study-series/` tiene checksums, revisiones inmutables y eventos
encadenados. El hash revisado vincula ciencia, evidencia, código, inputs, destino y presupuesto.
Apply lo revalida. Repetir el plan aceptado devuelve el mismo Job; revisiones incompatibles se
serializan y series diferentes pueden enviarse independientemente. No-op no escribe nada.

Se preasigna ID de Job antes del handoff. Una caída acreditada anterior a crear Job/request/estado
del proveedor se reconcilia con **ese mismo ID**. Preparación existente no se relanza a ciegas.
Autoridad ausente/corrupta/incierta bloquea. Refresh sella celdas nativas comprobadas, no solo un
éxito del scheduler. La consola actualiza la serie seleccionada. Preview sucesivo puede leer un
Job terminado sin sellar sin escribir; apply reconcilia esa finalización operativa.

Si falla un delta, abre su Job y pulsa **Resume Study…**, o `lf retry JOB --dry-run` seguido del
retry explícito normal. Solo se recupera la Execution del delta; las revisiones anteriores quedan
intactas. La serie sigue genealogía exacta, coste e historial y cuenta cada celda una vez.
Continuar desde una epoch requiere soporte explícito de checkpoints en la Work.
EXTEND no es RETRY, RERUN ni editar resultados antiguos.

Si la preparación falló **antes de existir una Execution nueva**, Resume/retry reintenta solo
la preparación del delta revisado. El nuevo Job conserva vínculo con la serie y pin de envío;
no repite revisiones anteriores correctas. Aquí `resumable: false` indica que todavía no hay
Execution/checkpoint de entrenamiento que continuar, no que haya que reiniciar el Study completo.
La búsqueda de recuperación descarta Executions locales ajenas al Job exacto.

Jobs/Executions referenciados y entornos se protegen de borrado/GC. Exportar la serie genera una
colección de paquetes nativos independientes, journal y JSON/HTML acumulados de solo lectura.
Un fallo conserva el parcial y su diagnóstico. Se pueden importar paquetes individualmente como
evidencia; adoptar/migrar una serie ejecutable no está soportado. Exportar una Execution no finge
contener toda su serie.

<a id="guide-study-extensions-verificación"></a>

### Verificación

Fixtures CPU cubren 20→35→56, 20→32, origen intacto, incompatibilidad, concurrencia/idempotencia,
autoridad corrupta/ausente, reconciliación anterior al Job, retry del delta y análisis acumulado.
Un proveedor loopback crea un intérprete aislado sin instalar dependencias, valida evidencia
en el host y ejecuta solo el delta con entrypoint/procesos nativos. Contratos GPU usan metadatos
sintéticos. No se afirma aceptación en SSH/SLURM/GPU reales ni ampliación de WISDOM.


<a id="guide-dataset-reconstruction"></a>

## 25. Datasets publicados y reconstrucción científica

Español · [English](MANUAL.md#guide-dataset-reconstruction)

`NOMBRE@VERSION` identifica un único content ID inmutable. Una validación científica **PASS** no
demuestra igualdad de bytes ni equivalencia de todos los miembros. LambdaForge nunca sobrescribe
esta versión, relaja checksums de transporte, redondea arrays o adivina qué campos internos de un
NPZ u otro formato son operativos.

<a id="guide-dataset-reconstruction-elegir-la-intención-antes-de-calcular"></a>

### Elegir la intención antes de calcular

```bash
lf datasets preflight corpus@6 --intent publish --on gpu12 --json
lf datasets preflight corpus@6 --intent reuse --on local --json
lf datasets preflight corpus@6 --intent rebuild --on gpu16 --json
```

El preflight lee el índice del controlador y de todos los clústeres configurados sin modificarlos,
descargar assets ni lanzar Works. Devuelve ubicaciones, identidades exactas, errores de descubrimiento
y `allowed`. Publicar se rechaza si la versión existe; reuse/rebuild exige una referencia no
ambigua. Un descubrimiento incompleto no prueba ausencia. No reserva la versión: la publicación
final sigue comprobando identidad bajo el lock del registro.

1. **Reutilizar**: `lf datasets materialize corpus@6 --on DESTINO` previsualiza la colocación exacta;
   revisar y añadir `--apply`. `replicate` elige origen/destino explícitos. Copia y verifica el
   contenido publicado, no reconstruye arrays. La réplica nativa comprimida admite ubicaciones
   locales/remotas y prefiere SSH directo entre clústeres; véase abajo.
2. **Reconstruir**: calcular un candidato independiente, sellado y no registrado para compararlo.
3. **Publicar**: usar una versión nueva para bytes distintos, incluso con equivalencia científica.
   No se admiten representaciones alternativas bajo la misma versión ni placements con otro hash.

<a id="guide-dataset-reconstruction-publicar-una-vez-replicar-el-contenido-exacto"></a>

### Publicar una vez, replicar el contenido exacto

```bash
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --json
# Revisar ruta, tamaño y destino; después aplicar:
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --apply
# Exigir ruta directa (envío O descarga), sin intermediación del PC:
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --route direct --apply
# Solo tras aceptar la dependencia del PC/red, solicitar relay explícitamente:
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --route relay --apply
```

En `lf`, abre **Datasets → DatasetVersion → Replicate…**, elige origen/destino, revisa el plan nativo
y confirma. El selector permite **Direct only · no laptop required** (por defecto si el origen es
remoto), Automatic y Relay explícito. Direct only bloquea si ninguna dirección SSH funciona y
muestra una explicación legible de host key/autenticación/herramientas, sin sustituirla por una
copia local. El host se deriva de la ruta real, no es una etiqueta editable: origen en envío directo,
destino en descarga directa y controlador en relay. Jobs relay existentes conservan su ruta;
seleccionar Direct only no migra ni cancela una transferencia enviada anteriormente.
La consola sigue respondiendo y muestra fases/tiempo transcurrido y bytes comprimidos
en la ruta retransmitida. **Transfers** conserva ID, origen/destino, estado, fase, bytes comprimidos,
host de ejecución y logs seleccionados entre sesiones. Cerrar el panel o LF no cancela una
transferencia ya confirmada. Locations se actualiza tras el
registro. El destino requiere `storage.dataset_root` y runtime LambdaForge disponible (existente o
preparado mediante bootstrap); no ejecuta Work científico/claim GPU. Mantiene el scope del proyecto.

`--route auto` comprueba primero envío SSH origen → destino y después descarga iniciada desde
destino ← origen. Las sondas acotadas, sin escritura ni copia de assets, verifican autenticación,
confianza de host, intérprete/imports necesarios, lectura del origen y planificación segura/escritura
del destino. El preview distingue autenticación, host key, red, timeout, permisos, herramientas y
fallo indeterminado por dirección. Una sonda inconclusa no demuestra ausencia de conectividad;
verificar requisitos ahora tampoco garantiza la conexión futura. Las sondas automáticas no reenvían
agentes/claves privadas ni relajan host keys. Un fallo de autenticación interactivo puede ofrecer
por separado la contraseña de un solo uso descrita abajo.

**Huellas de primer acceso.** La confianza SSH pertenece al host iniciador: aceptar una huella
en el portátil no autoriza 16 → 12. Un SSH manual desde 16 al mismo host/IP y usuario usados por
LF puede resolver la huella desconocida, pero si pide contraseña no demuestra que funcione sin
interacción. Alias e IP pueden tener entradas distintas en known_hosts.

Si tras aceptar las huellas el diagnóstico pasa a `authentication / Permission denied`, ya hay
confianza pero falta autenticación automática. Puedes configurar acceso no interactivo aprobado
en una dirección o introducir explícitamente una contraseña del destino para iniciar desde origen.
LF no instala claves ni reutiliza/reenvía secretos del controlador automáticamente. El relay SSH
con contraseña usa lectores binarios para cabecera y payload comprimido.

**Contraseña directa de un solo uso.** No requiere crear claves SSH ni guardar credenciales del SO:

```bash
lf datasets replicate NOMBRE@VERSION --source CLUSTER16 --destination CLUSTER12 \
  --route direct --ask-password --apply
```

El prompt oculto explica que la contraseña del destino se entrega por SSH autenticado/cifrado al
origen. Solo permanece en memoria y pasa al supervisor/worker nativo detached por pipes anónimos:
no se escribe en argv, entorno, solicitudes, registros Job ni logs de LF, tampoco en keyring.
El origen conecta directamente al destino con PasswordSshTransport; conserva el mismo motor de
archivo comprimido, checksums, identidad y publicación atómica. Los bytes no pasan por el portátil.
Mantenlo conectado hasta el acuse del Job; después el Job del origen ya no depende del PC.

En CLI interactivo se ofrece el prompt tras fallos clasificados como autenticación. JSON/no
interactivo nunca pregunta ni acepta contraseñas como argumentos. El preview autentica y revisa
permisos sin copiar datos. En LF elige **Direct · ask destination password**, o usa la modal
ofrecida tras fallar el acceso automático. El campo se oculta y vacía al aceptar/cancelar; el
secreto no aparece en la confirmación de copia posterior, que sigue siendo obligatoria. Huellas
desconocidas/cambiadas exigen su decisión separada: la contraseña no las acepta automáticamente.
Una contraseña incorrecta no provoca reintentos automáticos ni cambio a relay.

El runtime iniciador necesita LF con entrada efímera de Jobs y la dependencia opcional
`lambdaforge[cluster-password]`/Paramiko. Preflight comprueba ambos antes de enviar el secreto;
si faltan, se requiere bootstrap/actualización explícita, no modificar un prefijo inmutable ni
instalar al importar. El destino solo necesita el receptor LF normal. La autorización con
contraseña es origen→destino; nunca se prueba la misma contraseña en sentido inverso. MFA y
keyboard-interactive no están implementados. Tras caer proceso/host el secreto no se recupera:
el retry genérico de Job rechaza esas operaciones. Repite `datasets replicate --ask-password`
con preview/apply para aportarlo otra vez. Se conservan bloqueo de duplicados activos y verificación
exacta del destino.

Si Direct only falla por confianza de host, la consola ofrece una modal separada con cluster
iniciador, host/puerto del destino, huellas SHA256 y ruta remota del archivo de confianza. Contrasta
las huellas con el administrador u otro canal fiable y confirma o cancela. Las observaciones de
`ssh-keyscan` no están autenticadas: [aviso de seguridad de OpenSSH](https://man.openbsd.org/ssh-keyscan).
Confirmar vuelve a obtener/comprobar todas las huellas y añade exclusivamente claves desconocidas
al known_hosts efectivo del usuario en el iniciador. Usa nombres de host hasheados, lock y escritura
atómica conservando entradas/permisos. Nunca sustituye claves diferentes, revocadas o certificadas.
Después repite el preview normal; **iniciar la copia requiere otra confirmación**. Cancelar no cambia
confianza ni datasets. Aceptar la identidad del servidor no concede permiso de inicio de sesión.

La misma operación nativa existe por comando, sin depender de un dataset:

```bash
# Solo lectura: observar huellas desde el cluster iniciador.
lf datasets trust-ssh --source gpu16 --destination gpu12 --json
# Solo tras verificar: repetir --fingerprint para TODAS las claves del preview.
lf datasets trust-ssh --source gpu16 --destination gpu12 \
  --known-hosts-file /home/USER/.ssh/known_hosts \
  --fingerprint SHA256:HUELLA_VERIFICADA --apply
```

Usa la ruta remota exacta del preview, no una ruta del PC. Python ofrece
`DatasetService.replication_ssh_trust(initiator, peer, ...)`; aplicar exige huellas revisadas y
`expected_file`. Ninguna sonda normal de replicación escribe confianza automáticamente. Proxy SSH,
HostKeyAlias, expansiones especiales o archivos inseguros/no propios requieren administración
explícita, no suposiciones. El iniciador necesita SSH/keyscan/keygen e intérprete LF existente;
sin instalar paquetes, reenviar claves privadas/contraseñas ni relajar políticas SSH.

Si no hay dirección directa utilizable, auto **propone** relay, pero nunca lo inicia silenciosamente.
CLI interactivo muestra ruta, motivo, tamaño original y dependencia del PC y pide consentimiento;
en JSON/no interactivo debes elegir `--route relay`. La consola exige el mismo aviso/confirmación.
El tamaño original no es un total comprimido. Se fija la dirección aprobada: si cambia antes del
envío hay que revisar de nuevo, no se pasa a relay a escondidas.

Para rutas automáticas de claves/Kerberos, el administrador debe autorizar SSH y host keys en el iniciador
(origen para envío, destino para descarga), además de runtimes LF y permisos utilizables en ambos.
LambdaForge puede guardar claves públicas desconocidas aprobadas explícitamente como se explica
arriba; nunca cambia confianza en silencio, sustituye claves registradas ni distribuye claves privadas.
`--route direct` admite ambas direcciones y garantiza ausencia de intermediación local: si ninguna
funciona falla antes de copiar. `--route relay` acepta expresamente la dependencia del controlador y
usa sus transportes autenticados con un **stream comprimido en memoria acotada**, no un archivo
intermedio en tu equipo ni en `/tmp`.

La copia usa el ciclo de vida Job/scheduler nativo con tipo `dataset-replication`, no un Work
científico ni Study ficticio. El envío directo se aloja en origen y la descarga directa en destino:
supervisor, progreso, resultado y logs sobreviven allí aunque desconectes el PC. Solo se registra contenido verificado
en destino; al reconectar se reconcilian sus metadatos exactos con el índice del controlador.
Un conflicto de identidad exige reconciliación explícita, nunca elegir una referencia a ciegas.
Relay se ejecuta como Job local independiente de la consola: puedes cerrar LF, pero apagar/suspender
el PC o perder su red puede interrumpirlo. `requires_controller_online` señala esa dependencia del
host, no de la interfaz. Relay desatendido requiere OpenSSH o una referencia de credencial guardada,
sin contraseñas en snapshots ni prompts posteriores.

```bash
lf datasets transfers corpus@7 --json
lf datasets transfer-status job-YYYYMMDDHHMMSS-ID --json
lf jobs logs job-YYYYMMDDHHMMSS-ID --tail 100
```

Las filas separan estado del proceso, última observación confirmada/antigüedad y último avance de
bytes comprimidos/antigüedad. Si el host no es observable, muestran estado no confirmado y último
estado conocido; no running confirmado indefinidamente ni fallo inventado. Verificar/publicar puede
no mover bytes; transferir sin avance indica transferencia/espera, no bloqueo demostrado. No se
inventan porcentajes sin total comprimido conocido. Listado de Datasets y Transfers también
actualizan la observación nativa y conservan estados terminales, causa redactada y comando de logs.
EOF/BrokenPipe conservan la excepción original y stderr/exit del extremo cuando están disponibles;
sin esa evidencia no se puede distinguir pérdida de red de salida del receptor.

Solicitudes repetidas/concurrentes de la misma copia activa reutilizan el Job; estado incierto
no permite duplicarlo. Consultar estado lee metadatos acotados, no assets ni archivos comprimidos.
Una transferencia activa protege origen/destino frente a borrado nativo. No hay reintentos infinitos:
tras un fallo corrige la causa y solicita replicar otra vez. No es continuación por offset; una copia
incompleta se transfiere de nuevo y una ya verificada se reutiliza. Las transferencias iniciadas con
una versión anterior no se convierten retroactivamente en persistentes. En Python se usan
`DatasetService.replication_preview()`, `enqueue_replication()`, `replications()` y
`replication_status()`; `replicate()` de bajo nivel sigue siendo la copia síncrona que usa el worker
relay nativo. No hay un segundo motor de transferencia.
Antes de preparar una transferencia independiente, LF ancla workspace/almacenamiento locales y
las rutas explícitas de intérprete al proyecto fijado del controlador. Solicitud y perfiles locales
serializados usan rutas absolutas, independientes del `work_dir` del worker; las rutas SSH nunca
se resuelven con el filesystem del PC. Estado/historial fallidos incluyen una causa stderr acotada
y redactada y `logs_command`, aunque no haya progreso ni resultado. Cachés antiguas con error vacío
se actualizan una vez. Tras corregir un error de arranque, solicita la misma previsualización
`datasets replicate ...` y después `--apply`: Jobs activos/inciertos siguen impidiendo duplicados,
destinos exactos se reutilizan y se conserva el Job fallido con sus logs. No hay que recalcular datos.

Tar/gzip en streaming (nivel 3) verifica el origen completo, rechaza enlaces/entradas especiales y
rutas peligrosas, usa staging/admisión de almacenamiento en destino y comprueba contenido exacto
antes de promocionar/registrar atómicamente bajo lock. Ambos índices exponen el mismo content ID
a entradas Dataset tipadas. Un destino idéntico se verifica/reutiliza; otro contenido se rechaza.
Las copias interrumpidas no registran contenido parcial. Reintentar reutiliza una colocación exacta
ya publicada o transfiere de nuevo; no es reanudación por offset de bytes. Admite rutas antiguas
registradas de origen. Nunca reconstruye/cambia identidades/borra origen ni reconcilia versiones
conflictivas automáticamente: inspecciona divergencias o publica una versión nueva primero.

LambdaForge no puede inferir la versión de salida desde Python arbitrario antes de `Work.run()`.
El proyecto productor debe usar el preflight público como primera operación:

```python
def run(self, version: str, intent: str = "publish"):
    self.outputs.dataset_preflight(name="corpus", version=version, intent=intent)
    # El cálculo costoso empieza después. Guardar resultados en checkpoints.
    rows = self.compute_members()
    return self.outputs.dataset(
        name="corpus", version=version, members=rows, intent=intent,
        scientific_identity={
            "sources": {"source-id": "sha256:<checksum exacto de la fuente>"},
            "selection": ["member-id"],
            "labels": {"member-id": 1},
            "configuration": {"modes": 64},
            "algorithm": "my-project/spectral-contract-v1",
        },
    )
```

El preflight del Work consulta el registro de su host de ejecución. El comando anterior también
detecta divergencia entre hosts antes del envío. `intent="rebuild"` guarda el candidato en una
colección nombrada y fijada de checkpoints y devuelve su content ID y ruta como valor ordinario
(`publication_status="reconstructed-unregistered"`), nunca como DatasetVersion registrada.
`reuse` sirve para preflight/materialización, no como opción de publicación de `members`.

<a id="guide-dataset-reconstruction-tres-identidades-separadas"></a>

### Tres identidades separadas

- **Científica**: `scientific_identity` explícita del proyecto; exige fuentes, selección, etiquetas,
  configuración y contrato del algoritmo. Manifiesto/registro exponen `scientific_id`. Es una
  declaración, no prueba de reproducibilidad. Los manifiestos históricos sin declaración siguen
  siendo legibles; para bytes diferentes su equivalencia queda sin resolver.
- **Contenido**: `dataset_id == content_id`, checksums exactos de assets, tamaños e índice lógico
  canónico. No cambia transporte ni verificación. Una versión existente tampoco puede adquirir
  silenciosamente otra declaración científica aunque tenga los mismos bytes.
- **Operativa**: procedencia de construcción, rutas, Execution/Run/Attempt, hardware y fechas del
  manifiesto. No pertenecen a content ID. Meterlas dentro de un asset sí cambia sus bytes.

<a id="guide-dataset-reconstruction-comparar-reconstrucciones-completas"></a>

### Comparar reconstrucciones completas

```bash
lf datasets compare /raiz/referencia /raiz/candidato --on gpu16 \
  --verifier my_project.dataset_checks:compare_member --policy comparison-policy.json \
  --output comparison-report.json --json
```

Ambas raíces deben estar en el host seleccionado. El fichero de política y la ruta del informe
son locales al controlador; solo política/JSON cruzan el transporte, no assets. El verificador es
código explícitamente confiado del proyecto, importable en ese host; nunca se ejecuta un módulo
elegido por un manifiesto de datos.

API: `from lambdaforge.data import DatasetComparison, DatasetComparisonContext`;
`DatasetComparison.compare(left, right, verifier=callback, verifier_id="project/check-v1", policy=policy)`.
El callback recibe un contexto por miembro con raíces validadas, ambos `DatasetMember` y política.
Devuelve JSON estricto con `equivalent: bool`, `checked_assets: [nombres_logicos]` y `details`
opcional. Debe cubrir cada asset modificado. Usa las rutas relativas de los descriptores, no rutas
inferidas de la procedencia de máquina.

Para una referencia histórica sin contrato usa `--contracts contracts.json` (Python:
`scientific_contracts={content_id_exacto: declaracion, ...}`). Cada declaración explícita del proyecto
queda ligada al content ID verificado; debe proceder de evidencia fuente/selección/labels/algoritmo
comprobada independientemente, no de asumir equivalencia. El informe conserva `scientific_id` ausente,
registra estas declaraciones y `comparison_scientific_id` separado; sigue exigiendo verificador de
todos los miembros. No se edita manifiesto ni registro histórico. Nunca sustituye una declaración
persistida incompatible.

Ejemplo ilustrativo; las tolerancias deben justificarse desde el contrato real:

```json
{
  "diffusion_eigenvectors": {"method": "spectral-projector", "atol": 1e-8, "rtol": 0.0},
  "metadata_json": {"method": "exact-scientific-fields-operational-differences-reported"}
}
```

LambdaForge valida forma de política y tolerancias finitas/no negativas, no formatos numéricos.
El verificador implementa cada método y documenta diferencias operativas. Debe comparar
identificadores, etiquetas, particiones y bytes fuente internos **exactamente**, nunca con
tolerancias. Cambios de signo o bases en subespacios degenerados pueden exigir proyectores,
residuos y agrupación apropiada de autovalores, no redondeo ni `allclose` global.

Antes del callback se verifican ambos datasets completos con sus hashes exactos y se exige igualdad
de contratos científicos, identificadores, particiones, targets/metadata de miembros, nombres de
assets, schema y assets globales. Los URI cambiados necesitan materialización local. Corrupción
produce `invalid`; falta de declaración/verificador, `unresolved`; cambios científicos, `different`.
Una comparación completa aceptada produce `equivalent` conservando ambos content IDs distintos y
`byte_equal=false`. Nunca registra ni sustituye contenido. Bytes idénticos producen `exact` salvo
conflicto en contratos declarados. Comparar una proteína no demuestra equivalencia de todo el dataset.

<a id="guide-dataset-reconstruction-recuperar-una-publicación-sin-recalcular"></a>

### Recuperar una publicación sin recalcular

`outputs.dataset` conserva un candidato sellado rechazado e informa de su ruta exacta. Una
publicación fallida/interrumpida también protege artifacts de Attempt, métricas/logs/resultados y
checkpoints frente a compactación automática. Si el volumen de recuperación no admite una copia,
se conserva el árbol sellado en su volumen original y se informa de la ruta. Estos bytes retenidos
consumen espacio intencionadamente; inspeccionarlos antes de una eliminación explícita.

```bash
lf datasets publish-candidate /candidato/conservado --on gpu16 --version 7 --json
# Revisar origen exacto, identidades, destino y versión; después:
lf datasets publish-candidate /candidato/conservado --on gpu16 --version 7 --apply
```

Solo verifica/publica/registra bytes guardados, nunca ejecuta el Work. Omitir `--version` únicamente
si falta el registro original o ya tiene contenido/declaración exactos. Repetir apply es idempotente;
candidatos corruptos o inseguros se rechazan. El historial del Attempt original sigue fallido:
recuperar publicación no fabrica un cálculo correcto. No se borran candidato, versión original ni
checkpoints. `lf retry` ordinario sí invoca código del Work; no equivale a esta recuperación.
Local usa `storage.dataset_root` configurado o `datasets/published` junto al índice local; remoto
exige raíz permanente configurada.

<a id="guide-dataset-reconstruction-fallos-antes-del-sellado-0171"></a>

#### Fallos antes del sellado (0.17.1)

Antes de copiar, `outputs.dataset` guarda `publication-request.json` y un `members.jsonl` streaming
con checksums exactos de las fuentes en su colección de checkpoints de publicación. Si ese volumen
no acepta los metadatos iniciales, usa `.publication-requests` en el volumen de publicación.
El fallo informa de la ruta exacta. El mismo
`lf datasets publish-candidate DIRECTORIO_SOLICITUD [--on CLUSTER] [--apply]` acepta la solicitud:
preview verifica bytes/declaración y apply solo copia/sella/registra bajo los locks existentes.
Fuentes modificadas, rutas inseguras y conflictos de versión inmutable se rechazan. No se invoca
ningún Work ni se convierte el Attempt fallido en correcto.

Un fallo antiguo puede conservar fuentes validadas y su índice, pero no una solicitud completa.
No inventar metadatos científicos ausentes: restituir explícitamente la declaración original de
miembros/assets, schema, metadata y procedencia mediante la API pública de preparación:

```python
from lambdaforge.data import DatasetIndex, DatasetPublisher

# Debe coincidir con la declaración del productor, incluidos assets adicionales de diseño.
original_members = (member.to_dict() for member in DatasetIndex(source / "members.jsonl"))
request = DatasetPublisher().prepare_publication(
    name, version, original_members,
    source_root=source, request_root=recovery_directory,
    build_provenance=original_provenance,
    metadata=original_metadata, target_schema=original_target_schema,
    scientific_identity=original_scientific_identity,
)
print(request)  # publish-candidate sobre esta ruta, primero preview.
```

En el preprocessing antiguo de WISDOM, las evidencias de partida son
`attempt-0001/dataset/members.jsonl` y `dna-validation/dna-validation-report.json` retenidos.
También se debe mantener el asset `dataset_design` del primer miembro y la declaración original;
no inferirlas del interior de los NPZ. Solo se repite verificación/publicación, no geometría/anotación.

<a id="guide-dataset-reconstruction-conflictos-entre-registros"></a>

### Conflictos entre registros

`lf datasets list --all` y Datasets en la consola descubren los índices pequeños de todos los
clústeres configurados. Mismo nombre/versión/content fusiona ubicaciones; contenido distinto sigue
en filas separadas y marcadas **CONFLICT**. Listar no modifica registros y el descubrimiento
incompleto queda visible. Lectura de miembros y borrado global nunca adivinan una identidad en conflicto.
**Datasets → fila de identidad exacta → Manage copies…** sigue habilitado: selecciona destino y
**Keep as project reference**, **Remove registration · keep files** o **Delete managed copy**.
Cada acción previsualiza el hash/ruta y exige confirmación. REFERENCE marca la referencia del
controlador; las otras filas siguen CONFLICT hasta retirarlas explícitamente.

Los mismos comandos nativos solo previsualizan salvo que se añada `--apply`:

```bash
lf datasets list --on gpu16 --json  # obtener el content ID completo
lf datasets adopt corpus@6 --on gpu16 --content-id sha256:ID_COMPLETO_16
lf datasets adopt corpus@6 --on gpu16 --content-id sha256:ID_COMPLETO_16 --apply
lf datasets delete corpus@6 --on gpu12 --content-id sha256:ID_COMPLETO_12  # previsualizar
lf datasets delete corpus@6 --on gpu12 --content-id sha256:ID_COMPLETO_12 --apply
# Alternativa: retirar solo el índice, incluso copia rota/ausente o ruta externa antigua:
lf datasets remove corpus@6 --on gpu12 --content-id sha256:ID_COMPLETO_12 --apply
```

Sustituye los IDs de ejemplo por hashes completos observados. Adoptar verifica todos los checksums
en origen antes de archivar/sustituir la declaración del controlador. Elige la **resolución futura**,
no certifica equivalencia científica: no reescribe otros registros, cambia identidad de bytes,
sobrescribe publicaciones ni modifica Runs previas, inputs fijados, checkpoints o procedencia.
Para conservar una representación divergente como publicación usa otra versión explícita.

Borrar exactamente selecciona la identidad del índice de destino aunque el controlador apunte a otra.
Exige manifiesto coincidente, ruta gestionada y ningún consumidor activo. Retirar registro no borra
bytes ni libera espacio y permite retirar copias físicas corruptas/ausentes; el índice debe ser legible
y válido. Se retiran entradas vacías para que no reaparezcan. Ambas operaciones rechazan cambios de
hash/ruta, bloquean el registro y archivan la declaración previa en `dataset-registry-history/change-*.json`
antes de modificarlo. Ese archivo registra estado previo/solicitado, no prueba éxito de la escritura.
Registros inaccesibles/corruptos no se consideran ausentes. Si la retirada remota termina pero falla la
limpieza del controlador, inspecciona ambos índices de nuevo; nunca reintentes suponiendo una identidad.
No se borra evidencia histórica. Las operaciones sin hash siguen rechazando ambigüedad;
`reconcile` sigue conservando identidad.

<a id="guide-dataset-reconstruction-certificados-de-equivalencia-durables"></a>

### Certificados de equivalencia durables

`DatasetEquivalenceCertificate` reutiliza ProductRegistry inmutable como `ScientificReport` con
contrato reservado `lambdaforge/dataset-equivalence:v1`. Ejecuta la comparación existente de todo
el dataset antes de sellar la aprobación; no acepta un JSON arbitrario como prueba. Fija ambos
content IDs exactos, declaración científica completa, verificador, política por variable y evidencia.
Raíces operativas, fecha de comparación y procedencia de la Execution/operación productora quedan
separadas de identidad científica/contenido.

```python
from lambdaforge.data import DatasetEquivalenceCertificate
from lambdaforge.products import ProductRegistry
from my_project.dataset_checks import verify_member

certificate = DatasetEquivalenceCertificate.build(
    "/published/reference", "/sealed/reconstruction",
    name="corpus-reconstruction-check-v1",
    scientific_contract=science_declaration,  # Contrato completo explícito del dataset.
    verifier=verify_member,
    verifier_id="my_project.verify_member:v1",
    policy=variable_specific_policy,
    producer={
        "execution_id": "<Execution u operación de comparación productora registrada>",
        "evidence_fingerprint": "<identidad registrada de fuentes/comparación>",
    },
)
registry = ProductRegistry()
preview = certificate.publish(registry)  # No crea locks/archivos ni cambia registros.
certificate.publish(registry, apply=True)
restored = DatasetEquivalenceCertificate.load(registry, certificate.product.name)
```

Comparaciones no resueltas, científicamente distintas o corruptas no se certifican. Incluso bytes
idénticos necesitan declaración científica explícita para obtener certificado científico. Los
manifiestos antiguos requieren las mismas assertions `scientific_contracts` ligadas a content IDs
que `compare`. Informes mayores que los 512 KiB de metadata del catálogo fallan explícitamente;
nunca se recorta evidencia. `lf products show/provenance/export/import` y Products de la consola
consultan/transportan estos registros. Leer/importar no ejecuta el verificador registrado.

Son assertions de un proyecto de confianza, no autenticidad criptográfica ni igualdad transitiva.
`accepts(reference_content_id=..., candidate_content_id=..., scientific_contract=...)` solo consulta
el par y contrato exactos registrados; no verifica integridad actual. Antes de usar el candidato
materializado comprueba por separado `DatasetOperations.verify(root, candidate_content_id)`.
Nunca le atribuyas el content ID de referencia ni sobrescribas una versión inmutable.
La resolución automática Work YAML por contrato **todavía no está implementada**: inputs dataset
tipados ordinarios siguen exigiendo contenido exacto. Ningún certificado cambia eso implícitamente.

<a id="guide-dataset-reconstruction-cambios-necesarios-en-wisdom-no-realizados-aquí"></a>

### Cambios necesarios en WISDOM (no realizados aquí)

Los tres hashes comunicados para `wisdom-dna-reduced@6` representan contenido exacto distinto.
Coincidir en 45/47 entradas y tener 27 diferencias pequeñas de autovectores en `10FI_Y` no demuestra
equivalencia de 589 proteínas. WISDOM debe llamar al preflight antes de geometrías/anotaciones,
exponer intención publish/rebuild, checkpointar resultados costosos, declarar fuentes/diseño/labels
exactos y contrato del algoritmo, y mover rutas absolutas/versiones operativas del framework fuera
del NPZ científico a procedencia del manifiesto/sidecars. Los cambios relevantes del algoritmo
siguen formando parte del contrato. Debe aportar un verificador versionado de todas las variables,
con labels/splits/fuentes exactas, tolerancias justificadas por variable e invariantes espectrales.
Comparar todos los miembros, guardar informe y publicar versión nueva si cambian bytes. Nunca
modificar NPZ históricos registrados en sitio, reutilizar `@6` conflictivo ni inferir una tolerancia
universal desde una proteína.


<a id="guide-storage"></a>

## 26. Admisión de almacenamiento y limpieza segura

Español · [English](MANUAL.md#guide-storage)

<a id="guide-storage-consultar-y-limpiar"></a>

### Consultar y limpiar

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

<a id="guide-storage-admisión"></a>

### Admisión

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

<a id="guide-storage-qué-se-puede-eliminar"></a>

### Qué se puede eliminar

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

<a id="guide-storage-checkpoints-y-publicación"></a>

### Checkpoints y publicación

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

<a id="guide-storage-límites-actuales"></a>

### Límites actuales

Están implementadas la admisión de run_root y las reservas de copias de publicación gestionadas.
Staging de bundles y preparación de dependencias todavía no tienen reservas independientes de bytes
en sus volúmenes de destino. Generar índices/manifests de datasets no es una cuota de disco.
Los entornos managed nuevos usan identidad v3: una base de dependencias inmutable y una capa pequeña
de código exacto instalada independientemente (se explica abajo). Los prefijos v2 existentes no se
reescriben ni actualizan bajo Jobs. La orfandad entre hosts sigue sin inferirse. Los markers de runtime Python
y entorno comparten la adquisición atómica con GC y heartbeat acotado, también al verificar reuse.
Un prefijo completo inválido se conserva, no se reemplaza bajo Jobs que podrían referenciarlo.
`lf storage reconcile` ofrece un inventario diagnóstico de medidas, no contabilidad completa de cada
escritura. Esta guía no promete protección frente a todo ENOSPC durante preparación.

<a id="guide-storage-environment-layers"></a>

### Reutilización de dependencias, limpieza de entornos y reparación de referencias

Las nuevas ejecuciones y bootstrap comparten `ManagedEnvironmentProvider`. Un prefijo `deps-*`
resuelve y congela una vez las dependencias pesadas. Su contrato incluye `Requires-Dist`, requisitos
Python/extras de los wheels, Python/runtime/plataforma seleccionados, Torch/CUDA exacto, resolución
nativa y checksums de wheels offline. Cambiar código, YAML, seeds o hiperparámetros no duplica esa
instalación si no cambia el contrato. Comentarios/nombres del archivo nativo y notas de descubrimiento
no invalidan una resolución idéntica. Cambios reales de dependencias, Python/ABI, paquetes nativos o
política Torch/CUDA sí generan otra base.
Observaciones de driver/dispositivos y exigir una prueba CUDA en el Job actual son evidencia de
validación, no bytes de paquetes; la reutilización sigue verificando los grants reales del proveedor.

La base conserva sus wheels iniciales de bootstrap/instalación y el inventario resuelto; **no** son
el código científico seleccionado para nuevas ejecuciones. Cada capa `env-*` instala los wheels
exactos actuales de LF/proyecto con `--no-deps --ignore-installed` en su propio venv, con precedencia
sobre los paquetes iniciales de la base. No actualiza dependencias compartidas: verifica el contrato
combinado y publica atómicamente. Entry points Python de dependencias usan el intérprete de ejecución;
binarios nativos conservan su base inmutable. Los hashes de código siguen formando parte de identidad
y procedencia. La primera resolución compatible queda congelada, no se actualiza silenciosamente
desde el índice. No se instala nada al importar ni se deduplican prefijos antiguos en uso.
`state/dependency-resolutions/<contrato>.json` conserva inventario sellado y
`dependency_resolution_id` aunque se limpie la caché. Reconstruir una base aplica las versiones
exactas y rechaza diferencias antes de publicar. Un recibo corrupto es error diagnosticable, no
permiso para resolver otra vez. Entorno preparado y Job registran la identidad resuelta; cambiar
el contrato de dependencias/resolución genera otra vinculación. Los paquetes de código iniciales
de bootstrap se separan del inventario compartido fijado.
Una referencia nativa protege entorno/bundle/runtime desde staging hasta el hand-off al proveedor,
incluido el intervalo entre terminar la instalación y publicar request.json; no depende solo del
puntero activo, que otra preparación podría cambiar.

```bash
lf clean --on CLUSTER --environments-only --json          # inventario/plan sin borrar
lf clean --on CLUSTER --environments-only --apply --json  # solo tras revisarlo
```

La limpieza explícita selecciona entornos completos obsoletos sin referencias aunque no hubiera
cuota ni edad configuradas. Este modo no compacta Jobs, artefactos ni checkpoints. Protege punteros
activos, intérpretes/procesos vivos, builds, entornos de recuperación fallida/cancelada, dependencias
de recuperación y series. Una capa de código conservada retiene su base. Si no queda ningún consumidor
necesario, elimina primero código y después la base en la misma transacción GC. La limpieza automática
sigue cuotas/edad/presión; `storage.cache_max_size` y `storage.cache_max_age` se configuran en el perfil.
La cuota existente mide **bytes aparentes de archivos**, no promete liberar esa cantidad física.

El plan distingue cantidad, bytes aparentes, bloques únicos de **archivos** y estimación física
liberable. Cuenta enlaces duros una vez y no estima liberar sus bloques si queda un enlace fuera de
la selección. Metadatos de directorios, reflinks y actividad ajena pueden cambiar el resultado. Apply
revalida referencias/propietarios bajo el lock nativo GC, registra eliminaciones exactas y devuelve
bytes aparentes realmente retirados más delta observado de espacio libre por filesystem. Ese delta
no se atribuye exclusivamente a LF. Una operación vacía es `outcome=no-change`, no bytes liberados.

Un Job sin estado válido pero con solicitud legible protege su entorno **concreto**, no todos.
Sin solicitud ni propietario atribuible se conserva la protección y se identifica el Job bloqueante.
Se puede revisar y retirar explícitamente una referencia operativa abandonada:

```bash
lf storage repair --on CLUSTER --retire-job job-IDENTIDAD-EXACTA --json
lf storage repair --on CLUSTER --retire-job job-IDENTIDAD-EXACTA --apply --json
```

No cambia estado del Job, Execution, Run, Attempt ni archivos científicos. Audita el snapshot en
`state/storage-reference-repairs.json`; cambios de estado/solicitud o propiedad viva invalidan la
retirada. Requiere inspección de procesos disponible y no adivina propiedad de SLURM. Jobs remotos
ausentes del catálogo local se incluyen igualmente en el inventario; los argumentos de recuperación conservan
el Job/entorno original aunque no exista registro local.

`storage repair` también acepta repetidamente `--release-build .environment-build-env-ID.lock`.
El PID/nacimiento/muerte del controlador se comprueba **en su host registrado** y los descendientes
instaladores se verifican independientemente en el host del build. No se retiran propietarios vivos,
recientes, ilegibles o extranjeros no verificables. Edad, PID inexistente en otro host o marker
ilegible no prueban muerte. Se audita intención antes de retirar el marker exacto, nunca su prefijo.
Después de liberación verificada, los temporales huérfanos son candidatos GC. Un marker antiguo
ilegible puede quedar protegido hasta recuperar su atribución; no bloquea los demás prefijos.

El preview de repair muestra además un plan **hipotético** de limpieza posterior. No autoriza borrar:
tras la reparación aprobada hay que repetir `lf clean` con las referencias actuales. La consola
ofrece **Clusters → Storage → Repair references…**, selección exacta, preview y confirmación aparte;
**Clear storage…** usa el mismo servicio. Con cero candidatos explica el bloqueo sin ofrecer una
falsa limpieza. El refresco habitual usa metadatos de medición cacheados y capacidad física, sin
recorrer entornos; clean/reconcile explícitos hacen la medición profunda. El colector operativo remoto
usa el código actual del controlador sin modificar los entornos inmutables de trabajos científicos.


<a id="guide-managed-reports"></a>

## 27. Pestañas de informes con assets completos gestionados

[English](MANUAL.md#guide-managed-reports) · [Resultados de Work](#guide-work-results)

`html_section` sigue disponible para HTML autocontenido (16 MiB/documento y 64 MiB/informe).
Para visualizaciones mayores usa `self.outputs.html_bundle`: un directorio artifact gestionado
nativo, sin otro almacén científico ni runner. LF no elige proteínas, seeds, checkpoints,
ganadores, puntos o canales del proyecto.

```python
bundle = self.outputs.html_bundle("predictions", section="proteins", title="Proteins")
bundle.entry.write_text(inspector_html)
bundle.asset("catalogue", filename="catalogue.json", media_type="application/json").write_json(rows)
bundle.asset("surface", filename="surface.f32", metadata={
    "dtype": "float32", "byte_order": "little", "shape": [point_count, 3],
}).build(write_full_surface)
```

Declara cada asset explícitamente. Son bytes opacos completos: float32, NaN/Inf y máscaras de
validez no se convierten a JSON ni se redondean. IDs ASCII estables y nombres relativos NFC
únicos, también ignorando mayúsculas. Rutas absolutas, traversal, symlinks, archivos especiales,
IDs duplicados y archivos no declarados se rechazan.

Al finalizar se sella `lambdaforge-report.json`: versión, entrada, IDs, nombres relativos,
tamaños, checksum gestionado, SHA-256 del contenido, tipos MIME y metadata del proyecto.
El artifact original del Run atestigua el manifiesto; esquema público empaquetado:
[`report-bundle-v1.json`](../src/lambdaforge/schemas/report-bundle-v1.json).
Entrada limitada a 16 MiB y manifiesto a 8 MiB; estos límites no afectan a los arrays.
El directorio íntegro se conserva/exporta como evidencia gestionada normal.

<a id="guide-managed-reports-api-del-navegador"></a>

### API del navegador

```javascript
await lambdaforgeReport.ready;
const controller = new AbortController();
const bytes = await lambdaforgeReport.load('surface', {signal: controller.signal});
const positions = new Float32Array(bytes);
// Al cambiar selección: controller.abort(); liberar renderer/buffers/referencias.
// Al cerrar definitivamente: lambdaforgeReport.dispose();
```

`assets` expone IDs, tamaños, MIME y metadata; versión pública `1`. `load` devuelve ArrayBuffer
o error explícito; cancelar produce `AbortError`. Máximo dos cargas simultáneas, sin cola
ilimitada. El padre transfiere los buffers, sin guardar una caché completa. Cambiar pestaña o
documento aborta peticiones, cierra el puerto y descarga el iframe. El proyecto debe liberar
su renderer y no acumular arrays antiguos.

Se mantiene origen opaco y `connect-src 'none'`: sin URLs externas, DOM padre, rutas ni
credenciales. Un MessagePort se entrega solo al iframe seleccionado. El padre comprueba
tamaño/checksum; el servidor vuelve a verificar el artifact del Run y el manifiesto atestiguado
antes de abrir un archivo. Assets ausentes/corruptos son errores de presentación. Un fallo
al sellar queda visible como `presentation_error`, conservando otros resultados científicos.
Una excepción del consumidor anterior a la finalización sigue siendo una excepción científica
normal, no se oculta.

<a id="guide-managed-reports-visor-nativo-remoto-y-offline"></a>

### Visor nativo remoto y offline

**Open HTML report** en consola inicia automáticamente el lector loopback para bundles.
Listas/HPO/heartbeat no descargan arrays. Abrir informe lee entrada/manifiesto; seleccionar
un asset trae solo sus bytes por transporte nativo, comprimidos gzip y verificados. Sin servidor
del proyecto, CUDA ni entreno. Peticiones remotas tienen cancelación/plazo y spool temporal
junto al informe, no en `/tmp`.

```bash
lf reports serve ./reports/study-analysis.html --open
# Si un único asset supera el límite de visualización de 256 MiB:
lf reports serve ./reports/work-results.html --max-asset-mib 1024 --open
```

Solo evidencia declarada, en `127.0.0.1`, URL privada, Host y origen exactos. No es explorador
de archivos general. Dos lecturas binarias, ocho conexiones, 256 MiB por asset y cierre tras
30 minutos inactivo. Ctrl-C cierra solo el lector. Nunca se recortan puntos/proteínas en silencio.

`file://` no admite lazy fetch seguro de archivos hermanos: muestra cómo abrir el visor nativo.
Export/import conserva bundle completo, informe y sidecar con referencias relocatables. Mantén
todo el paquete; moverlo y ejecutar el visor no necesita cluster ni modelo originales. El sidecar
es metadata de despliegue descartable, no resultado científico.

Para regenerar desde inferencia retenida usa ResultInput y Work públicos: fija Run/Attempt
con `lf results reference`, enlázalo como input `result` y lee `previous.artifact(NAME)`.
Construye el bundle desde predicciones guardadas, sin cargar modelo ni hacer forward passes.
Ejemplo: [`managed_report.py`](../examples/managed_report.py). Si faltan predicciones no se inventan,
no se cambia seed/checkpoint ni se repite inferencia. Pruebas del navegador incluyen asset de
70 MiB y catálogo de 206 registros; selección/razón/procedencia son autoría del proyecto.


<a id="guide-research-analysis"></a>

## 28. Espacio de investigación de Studies

[English](MANUAL.md#guide-research-analysis) · [Manual](MANUAL.es.md#16-análisis-de-estudios)

Versión actual de la aplicación: **0.19.0**. Análisis v8 y vistas guardadas v1 siguen compatibles;
los informes siguientes conservan el contexto histórico de su implementación.

<a id="guide-research-analysis-1-abrir-el-espacio"></a>

### 1. Abrir el espacio

```bash
lf results analyze EXECUTION --recompute --json
lf results report EXECUTION --output research.html
```

El HTML necesita `lambdaforge[analysis-report]`; el análisis numérico no necesita Plotly. Para un
Study remoto usa `lf export STUDY --output ./exports` o Export en la consola. El informe funciona
sin conexión: es una captura, no una conexión viva al clúster. Regenera para incluir nuevas Runs.

**Overview / Resumen** es la primera pestaña y la vista inicial de un informe nuevo. Muestra
estados con recuentos exactos, contexto de selección/confirmación, asociaciones predictivas no
condicionales y recomendaciones que abren el gráfico concreto. El score por número de candidato
queda plegado en **Trials**, no domina el resumen. Al reabrir se conserva la última pestaña.

Los parámetros condicionales no entran en el gráfico de importancia global. Su tabla muestra la
regla de activación, candidatos activos/total, soporte completo, score modelado guardado y
fiabilidad. El score puede mezclar activación de la rama con variación interna; no es un porcentaje
causal, una cuota aditiva de responsabilidad ni una justificación para elegir esa rama.
Interpretation usa un carrusel horizontal desplazable con ratón/teclado. **ⓘ** reúne explicaciones
comunes de estabilidad y ruido; los detalles conservan la conclusión original exacta.

**Language / Idioma** en la cabecera cambia inglés/español inmediatamente y guarda la elección
para ese HTML. Cambian controles, ayudas, estados y etiquetas de gráficos compatibles. Los nombres
declarados y las afirmaciones científicas guardadas conservan su idioma original: ningún servicio
remoto traduce ni modifica la evidencia.

**Research** conserva las tarjetas de salud. Preguntas e inbox exploratorio usan carruseles;
la lista completa está paginada de seis en seis.
**Inspect** explica evidencia, método, soporte, ranking y límites;
**Explore these observations** abre los valores registrados. Una asociación exploratoria no es
una nueva conclusión del HPO ni un efecto causal.

**Metrics & health** busca etiquetas, claves estables, descripciones, alias y tags. Filtra por
categoría y ordena por prioridad, cobertura o dispersión. Las métricas constantes, ausentes u
ocultas se omiten al principio, no se borran: activa **Show constants / missing / hidden**.
El inspector muestra unidades, dirección, agregación y ausencias explícitas. Los grupos de
redundancia son descriptivos; sus miembros siguen accesibles. **Ctrl/⌘ K** busca métricas,
parámetros, familias, hallazgos, Trials y vistas guardadas. Los controles de métrica, parámetro,
ejes, tipo, paleta y filtros usan un desplegable buscable junto al botón, no una modal. El buscador
global **Ctrl/⌘ K** conserva su diálogo con favoritos/recientes. **Categories** es un árbol desplegable: seleccionar
`validation` incluye descendientes como `validation/global` y `validation/surface/quality`.

<a id="guide-research-analysis-2-declarar-significado-no-política-de-ejecución"></a>

### 2. Declarar significado, no política de ejecución

Sin declaraciones todo sigue funcionando. Las direcciones/rangos estándar y prefijos simples
identifican parte de la semántica; lo demás sigue desconocido. Añade `analysis` al YAML de un Work,
o un mapping de clase `Work.analysis_profile` con la misma estructura:

```yaml
analysis:
  defaults:
    - pattern: val_*
      metadata: {category: validation, split: validation, unit: ratio}
  metrics:
    val_accuracy:
      label: Accuracy
      description: Fracción de ejemplos de validación clasificados correctamente.
      category: validation/quality
      direction: max
      range: [0, 1]
      aggregation: latest
      visibility: primary
      aliases: [accuracy]
      priority: 10
    rejected_fraction:
      category: integrity
      expected_to_vary: true
      unit: ratio
      range: [0, 1]
    transformed_accuracy:
      derived_from: [val_accuracy]
      transformation: Transformación explícita definida por el proyecto.
```

Las etiquetas no renombran claves. Las unidades no convierten valores automáticamente.
`aggregation` describe evidencia real: `latest`, `best`, `selected_epoch`, `terminal`, `mean` o
`unspecified`. Declarar `best` **no** calcula un óptimo inexistente. La selección conserva su
semántica checkpoint/seed y los diagnósticos conservan sus valores reales; la agregación
desconocida se advierte, no se adivina.

También existen `tags`, `role`, `phase`, `scale`, `notes`, `discovery: false`, `practical_scale` y
`visibility: normal|advanced|hidden`. La escala práctica afecta a la salud visual, no al margen de
equivalencia científica del objetivo. Precedencia: inferencia, reglas de patrón por orden,
declaraciones de clase y overrides YAML. Los campos individuales de una métrica se combinan;
questions/defaults YAML sustituyen esas colecciones de la clase. Familias se combinan por nombre:
la definición YAML sustituye la familia homónima; campos individuales de métricas se combinan.
En una composición
`steps`, declara `analysis` en cada Work o clase, no en la raíz. Las métricas personalizadas pueden
pasar `metadata=` al constructor base de `Metric` y reutilizar `.metadata` en el perfil; no se
construye un Work para descubrir métricas transitorias.

Se validan localmente campos, rangos, categorías, direcciones, alias, coordenadas de familias,
referencias y ciclos de derivación. Métricas de test, incluso derivadas, no pueden gobernar el
objetivo ni restricciones, tampoco dentro de una utilidad compuesta. Siguen disponibles para
análisis final a posteriori, pero se excluyen del descubrimiento automático provisional. El
análisis no inventa restricciones, pesos del objetivo ni reglas de poda.

<a id="guide-research-analysis-3-familias-y-preguntas"></a>

### 3. Familias y preguntas

Agrupa mediciones repetidas por su significado:

```yaml
analysis:
  families:
    efficiency:
      label: Calidad por fracción de datos y estrategia
      dimensions:
        strategy: {kind: categorical, values: [uniform, balanced]}
        fraction: {kind: ordered, values: [10, 25, 100]}
      template: quality_{strategy}_{fraction}
  questions:
    - {id: efficiency-view, kind: metric_family, family: efficiency}
    - id: capacity-quality
      kind: parameter_screen
      parameters: [width]
      metrics: [val_accuracy]
    - id: expected-agreement
      label: Acuerdo con la accuracy transformada
      priority: 10
      kind: relationship
      x: val_accuracy
      y: transformed_accuracy
      expected: positive
```

Dimensiones numéricas, ordenadas o categóricas. `members` puede asignar explícitamente claves a
coordenadas en vez de usar template. Las plantillas admiten hasta 4096 miembros únicos. Las
gráficas conservan orden, etiquetas, valores ausentes y soporte. Su SD es dispersión entre
candidatos, no confianza entre seeds; soporte desigual no representa una intervención pareada.
Los miembros de familias no se colapsan automáticamente como redundantes.

Tipos de pregunta: `relationship`, `consistency`, `parameter_screen`, `metric_family`,
`category_summary` y `tradeoff`. Relaciones/consistencia necesitan métricas `x`/`y` y admiten
`expected: positive|negative|equal`; tradeoff enumera al menos dos métricas y category_summary una
categoría. Referencias requeridas desconocidas fallan; `optional: true` admite una métrica ausente
y registra la pregunta no disponible. `label` y `priority` no negativa son opcionales y nombran/
ordenan tarjetas, no ponderan HPO. Las preguntas enfocan la exploración, no planifican Runs ni
crean un protocolo de contraste formal. Resúmenes de familia/categoría y Pareto de recursos ya
existente siguen siendo descriptivos.

<a id="guide-research-analysis-4-cómo-funciona-el-descubrimiento"></a>

### 4. Cómo funciona el descubrimiento

1. Matriz por candidato sobre evidencia **completa comparable**, sin fabricar scores finales de
   candidatos censurados. Screening y confirmación siguen separados; no se mezclan fidelidades.
   Los alias se mapean a claves canónicas de análisis.
2. Perfil de todas las métricas: soporte finito/ausente, valores únicos, rango, media, mediana, SD,
   dispersión robusta y soporte registrado de Runs/seeds. Constancia exige igualdad repetida.
   Las métricas presentes en menos de la mitad de los candidatos muestran `low_coverage`;
   conservan cobertura de candidatos/Runs y soporte de seeds numéricos. El ranking penaliza
   también la cobertura pareada real, sin equiparar unos pocos ejemplos con evidencia completa.
   La clasificación casi constante compara dispersión con rango/escala declarada o magnitud (`1e-6`),
   nunca un epsilon absoluto que destruya señales de unidades pequeñas.
3. Conjunto acotado y determinista, priorizado por la declaración. Grupos de respuestas casi
   redundantes: Spearman absoluto al menos `.995` y seis candidatos pareados, excluyendo familias.
   Es una ayuda de navegación, no prueba de equivalencia científica.
4. Screening acotado parámetro/métrica y métrica/métrica, intercalando ambos tipos y rotando
   parámetros para que el primero no monopolice el presupuesto. Los parámetros numéricos reutilizan
   `ParameterSpace`, incluidas escalas log, categorías y condiciones. Se comparan correlación de
   rangos y respuesta cuadrática de rangos con leave-one-out; categorías usan eta-cuadrado.
   Ninguna de estas relaciones es un nuevo modelo causal multivariante.
5. Hasta 32 relaciones preseleccionadas reciben permutaciones deterministas y bootstrap de signo.
   Se reutilizan matrices pequeñas, sin matrices cuadráticas de candidatos ni productos completos
   de métricas. Tests sin confirmar cuentan como `p=1` en el ajuste Benjamini–Yekutieli. Valores
   por defecto: 64 métricas, 128 relaciones y 256 permutaciones; `analysis.discovery` permite
   ajustar `max_metrics`, `max_pairs`, `resamples` dentro de límites validados o desactivar discovery.
6. Hallazgos priorizados con efecto, soporte y estabilidad separados. La derivación directa,
   transitiva o compartida reduce novedad. También hay invariancia, outliers descriptivos por
   1.5-IQR, cambios de signo en contextos categóricos acotados y hallazgos del análisis existente.
   Persisten componentes del ranking y penalizaciones por redundancia. El inbox muestra hasta
   ocho grupos de evidencia distintos; los equivalentes permanecen en la lista completa.

Efecto, soporte, estabilidad, p ajustado y confianza científica no son lo mismo. **Supported
exploration** exige soporte y p diagnóstico ajustado pequeño; **preliminary** no está confirmado.
La selección adaptativa, comparaciones múltiples, reutilización de datos y ruido de seeds limitan
estos diagnósticos. BY no transforma un Study adaptativo en experimento aleatorizado formal.
El acuerdo del bootstrap no es la probabilidad de que una conclusión sea verdadera. Outliers y
contextos son explícitamente inspection-only, sin confianza inventada. Métodos basados en la
[guía de permutaciones de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
y el [ajuste de FDR con dependencia](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

<a id="guide-research-analysis-5-explorar-y-guardar"></a>

### 5. Explorar y guardar

En **Explore**, elige **Analyze** (métrica), **By** (chips de parámetros), **Compare with** opcional
(chips de métricas) y el conjunto completo/parcial. **Explore observations** usa el renderer
existente: un parámetro produce scatter, dos heatmap y más coordenadas paralelas. Elimina chips
con ×. Métricas y controles avanzados conservan su preview inmediato. **Advanced visualization
options** agrupa tipo/ejes X/Y/Z explícitos, agregación, paleta/inversión y normalización visual;
tipo/paleta/filtros usan el mismo desplegable buscable. Nombre, notas y guardado quedan visibles.

- Varias Y comparten escala solo con unidades declaradas iguales. Unidades distintas/desconocidas
  usan gráficos separados; **Visual 0–1 normalization only** es opcional y no cambia evidencia.
- En **Parameters**, abre **Analyze metric**, busca `mean`, marca una métrica, busca `auroc` y
  marca otra sin cerrar el desplegable. La selección se conserva entre búsquedas; desmarcar elimina
  la curva y ninguna selección muestra un estado vacío explícito. Flechas navegan, Espacio marca,
  Escape/Listo/clic exterior cierra. **Estilo del gráfico** permite líneas/puntos, barras agrupadas,
  distribuciones de candidatos por X o mapas de calor para muchas métricas. No hay sectores:
  métricas independientes no son partes aditivas de un total. **Distribución** ofrece gráfico
  combinado o un panel por métrica. **Escalas Y → Automáticas** agrupa rangos con al menos un 50%
  de solapamiento del intervalo más estrecho (incluye contención), compatible entre todos los pares
  del grupo. Una constante puede compartir un rango que la contenga. Nunca agrupa unidades conocidas
  distintas; unidades desconocidas pueden compartir escala visual, no significado científico.
  Puedes forzar escalas independientes o compartidas; los paneles separados respetan esa elección.
  También hay puntos por Trial sin promediar, curvas escalonadas y áreas, violines con puntos reales,
  barras horizontales, histogramas y CDF empíricas. Histogramas/CDF agrupan todos los resúmenes
  candidatos coincidentes; cajas/violines permiten comparar valores exactos del parámetro.
  Unir puntos es descriptivo, no un ajuste de respuesta. Los ejes automáticos muestran los nombres
  de sus métricas, no números de escala anónimos.
  No compares alturas entre ejes independientes. **Dispersión** permite ocultar desviación típica,
  usar bigotes o banda ±SD en líneas con X numérico; otros estilos usan bigotes. Las distribuciones
  ya muestran observaciones. La SD es entre candidatos, no confianza entre semillas, y falta para
  una observación. El mapa normaliza colores por métrica (color neutro para constantes); el hover
  muestra medias originales y lo no observado queda vacío. Los ajustes persisten en este HTML en
  el navegador; regenerarlo los reinicia. Explore conserva gráficos separados por defecto y
  normalización visual opcional.
  La selección no modifica agregación, objetivo ni respuesta modelada persistida.
- Heatmaps/superficies numéricas observadas dejan vacías las combinaciones no probadas. Las
  superficies del surrogate permanecen separadas y etiquetadas. Scatter 3D admite categorías.
- Coordenadas paralelas usan los parámetros elegidos (o el default de ocho en vistas antiguas)
  y una métrica; excluyen coordenadas ausentes explícitamente y no ajustan modelos nuevos.
- Trials conserva marcas de poda/parcialidad. Busca en el ledger y compara dos candidatos en una
  tabla por categorías con valores/diferencias. Hay filtro por rama condicional raíz cuando aplica.
  La comparación empieza con métricas principales/prioritarias; buscar o **Show all recorded
  metrics** la amplía. Dirección desconocida no significa mejora.

**Interactions** tiene un panel observado para cualquier métrica con los selectores compartidos
y heatmap/3D/superficie. El modelo predictivo del objetivo permanece separado. **Evidence**
(antes Findings & evidence) conserva conclusiones científicas, diagnóstico del surrogate, evidencia
de seeds/poda, metodología y snapshot estructurado en secciones plegables. Los hallazgos exploratorios aparecen solo en
Research; el dashboard individual de Run no cambia y sigue usando epochs.
El snapshot completo de auditoría se formatea al desplegarlo y reutiliza el catálogo de métricas,
serializado una sola vez. Todos los paneles comparten el catálogo leído, sin selectores independientes.

Guarda vistas con nombre y notas. Exporta/importa **research-views.json**, que transporta
configuración visual, no evidencia. Se validan versión, campos, referencias, tamaño y cantidad.
Las preferencias del mismo HTML conservan vistas, paletas, tamaños, favoritos y búsquedas;
regenerarlo crea un ámbito nuevo. No hay servicios cloud ni LLM necesarios.

<a id="guide-research-analysis-pestañas-html-del-proyecto"></a>

#### Pestañas HTML del proyecto

La visualización de dominio pertenece al consumidor. Dentro de `Work.run`, declara un HTML
gestionado y escribe el documento completo de tu aplicación:

```python
viewer = self.outputs.html_section("predictions", section="proteins", title="Proteins")
viewer.write_text(render_predictions_html())
```

La acción HTML interactiva del Study, `lf results report SELECTOR --output report.html` y la
exportación portable leen estos archivos solo bajo petición explícita. Una sección crea una pestaña
junto a Explore/Evidence/Resources. Varios outputs o Runs comparten pestaña con selector buscable
etiquetado por Trial/seed/output. Los títulos de una sección deben coincidir. Overview, HPO,
paneles Analysis y sondeos ordinarios no descargan HTML. La lectura remota se pagina y cachea en
su propio host, fuera del heartbeat del planificador.

El documento es HTML UTF-8 autocontenido: incluye JavaScript/CSS, visualizaciones y datos de
predicciones; no funcionan rutas relativas a recursos ni peticiones externas. LambdaForge no
programa lógica de dominio, lee NPZ ni expone callbacks Python al navegador. Cada documento usa un
iframe de origen opaco `sandbox="allow-scripts allow-downloads"` y CSP restrictiva; sus scripts no
acceden al dashboard padre, archivos locales, otros sitios ni credenciales. Esto aísla presentación,
no el Python confiable del Work. Dentro del iframe el HTML sigue siendo interactivo; preferencias
y navegación del dashboard son independientes. Antes de incrustarlo se verifican ownership,
tamaño exacto y fingerprint persistido. Symlinks, contenido ausente/modificado o exceso de límites
(16 MiB/documento, 64 MiB/informe) dan error explícito, no pestañas vacías. Se usa el lifecycle y
retención del output ordinario, sin otro registro, runner o sistema de plugins.

También se pueden aportar documentos directamente sin ejecutar un Work:

```python
from lambdaforge.analysis.Report import write_html

write_html(analysis, "report.html", sections=[
    {"name": "proteins", "title": "Proteins", "label": "Trial 3 · seed 7", "html": html_text},
])
```

<a id="guide-research-analysis-6-arquitectura-seguridad-y-límites"></a>

### 6. Arquitectura, seguridad y límites

`MetricCatalog` posee significado; `AnalysisProfile` valida declaraciones inmutables;
`ResearchAnalysis`/`ResearchDiagnostics` calculan read models deterministas; assets offline
modulares los muestran. `StudyAnalysis` conserva efectos, cobertura, seeds, ganador/confirmación
y recursos. El documento versión **8** añade `research` sin eliminar campos. Se elimina el viejo
renderer lineal sin uso, no se mantiene una segunda ruta.

Antes de ejecutar, `analysis-semantics.json` guarda declaraciones resueltas, defaults conocidos
del registro, identidad y versiones. Las especificaciones hijas referencian ese fichero, sin
copiar el catálogo por Run. Finalización, recarga, recuperación y export usan evidencia congelada;
cambiar etiquetas no cambia el fingerprint científico previo ni decisiones HPO. La caché de
análisis verifica además la identidad semántica. Studies antiguos reciben metadata inferida;
documentos HTML antiguos obtienen navegación de nombres, no hallazgos retrospectivos inventados.

Límites deliberados: discovery acotado, no exhaustivo; semántica desconocida permanece así;
señales no lineales/contextuales son herramientas aproximadas de inspección, no un modelo
universal de interacciones; no causalidad ni pesos nuevos; no nueva confianza shared-seed;
sin imputación, datos fabricados ni fitting científico en navegador. HPO y análisis multivariante
previos siguen siendo autoridades. Export/import transporta vistas, no un workspace vivo ni
actualización automática del HTML. Las conclusiones acopladas rendimiento/recursos necesitan
evidencia comparable suficiente de recursos y seeds.

<a id="guide-research-analysis-7-informe-de-implementación"></a>

### 7. Informe de implementación

Este informe recoge la implementación y verificación local del 2026-10-03.

1. **Arquitectura previa:** Study Analysis ya normalizaba evidencia y gestionaba ganador/seeds,
   cobertura, efectos multivariantes y recursos. El HTML era principalmente una colección de gráficas.
2. **Causas de usabilidad:** nombres/selectores planos, poca semántica, métricas ausentes/constantes
   compitiendo por atención, navegación monométrica y falta de priorización para investigación.
3. **Estructuras nuevas:** `MetricCatalog`, `AnalysisProfile` inmutable, `ResearchAnalysis` acotado,
   helpers de inspección/ranking y presentación offline modular integrada en el análisis existente.
4. **Catálogo:** `analysis.metrics/defaults` opcional o `Work.analysis_profile`; se reutiliza
   `Metric.metadata` sin construir objetos Work científicos.
5. **Perfil:** declaraciones, familias y preguntas/discovery validados; combina clase/YAML y se
   congela de forma independiente del objetivo y política de ejecución.
6. **Familias:** coordenadas explícitas o plantillas acotadas con placeholders simples; orden,
   categorías y soporte incompleto aparecen juntos sin inventar mediciones.
7. **Salud:** igualdad repetida, casi constancia relativa, ausencias, cobertura de candidatos/Runs/
   seeds, rangos y variación esperada; no borra ni imputa evidencia.
8. **Redundancia:** agrupación determinista por similitud de rangos, representante y miembros
   desplegables; protege familias. Ranking/inbox reducen repetición, no eliminan valores.
9. **Discovery:** screening intercalado parámetro/métrica y métrica/métrica, respuesta de rangos/
   cuadrática, asociaciones categóricas, relaciones declaradas y señales baratas de inspección.
10. **Multiplicidad:** permutaciones deterministas acotadas y ajuste BY sobre hipótesis examinadas;
    tests sin confirmar cuentan como `p=1`. La resolución limitada sigue siendo conservadora.
11. **Fiabilidad:** soporte, efecto, cobertura, estabilidad de signo, p diagnóstico ajustado y
    novedad son componentes separados; ninguno es probabilidad de verdad.
12. **Derivadas:** linaje acíclico validado; relaciones estructurales directas/transitivas/compartidas
    pierden novedad. No se infiere dirección ni equivalencia práctica de una fórmula derivada.
13. **Test:** se rechaza evidencia de test/derivada para objetivo/componentes/restricciones antes
    de ejecutar. Discovery provisional la excluye; análisis terminal conserva sus valores registrados.
14. **Persistencia:** `analysis-semantics.json`, identidad semántica separada y referencias hijas;
    defaults congelados al finalizar, recargar, recuperar y exportar.
15. **HTML:** inbox Research, salud/catálogo, inspectores semánticos, hallazgos diversos y Explore
    progresivo; se retira el renderer lineal viejo sin uso.
16. **Búsqueda:** palabras/subsecuencias sobre claves, etiquetas, alias, descripciones/categorías,
    parámetros, familias, hallazgos, Trials y vistas; teclado, favoritos y recientes.
17. **Parameters multimétrica:** renderer observado compartido; añade métricas, separa escalas
    desconocidas/distintas y normaliza visualmente solo cuando se solicita.
18. **Pares/3D:** scatter/heatmap/superficie observados con categorías y huecos; superficies modeladas
    del objetivo siguen separadas. Coordenadas paralelas acotadas, sin otro modelo científico.
19. **Explore:** preview inmediato, X/Y/Z, varias Y, tipo, agregación, paleta, evidencia parcial y
    normalización explícita; Advanced no domina la experiencia inicial.
20. **Research Views:** especificaciones guardadas con nombre/notas y preferencias por HTML;
    import/export JSON validado/acotado transporta presentación, no datos científicos.
21. **Regresiones añadidas:** precedencia/validación, leakage/linaje, recarga congelada, namespaces,
    censura/ausencias, salud, familias, discovery determinista/no lineal, catálogos grandes y navegador:
    búsqueda, inspectores, unidades, comparaciones y persistencia/import de vistas.
22. **Verificación:** Ruff/mypy correctos; pytest completo: **1118 passed**, cuatro avisos Lightning
    no fatales sobre uso CPU/loaders. Pasan análisis/CUDA/documentación focalizados, smoke de wheel
    instalada con CLI/scaffold/validación/HTML e interacción/capturas Chromium. Matriz sintética de
    500 candidatos, 300 métricas, 15 parámetros y 15 seeds: **9,424 s**, **4.809.115 bytes** y solo
    32 relaciones con remuestreo costoso por defecto. El tiempo depende de la máquina.
23. **Versiones/esquema al implementar:** release era **0.16.0**; documento de análisis pasó a **8**;
    documentos research/semánticos comienzan en **1**. YAML añade declaraciones explícitas opcionales.
24. **Evidencia anterior:** no exige declaraciones; ausencia de semántica se infiere explícitamente.
    Se conservan campos científicos. HTML legado obtiene navegación, no hallazgos inventados.
25. **Límites reales:** discovery acotado/observacional puede seguir preliminar; no se afirman modelos
    genéricos de saturación, fairness/subgrupos ni causalidad de alto orden. Se reutilizan análisis
    previos de top-region/recursos/seeds, no se generalizan a toda métrica. No se añade ingestión de
    baselines externos ni color/tamaño/faceting arbitrario de candidatos: la referencia es el Trial
    seleccionado en comparación de dos Trials. Las vistas guardadas no son un workspace vivo.

No se modificó WISDOM ni se contactó ningún clúster real durante la verificación.

<a id="guide-research-analysis-8-informe-de-la-segunda-simplificación"></a>

### 8. Informe de la segunda simplificación

Este apartado recoge la segunda ronda focalizada del 2026-10-03; el apartado 7 describe la base previa.
La [guía de Studies condicionales](#guide-conditional-studies) contiene gramática y ejemplos de planificación.

1. **Controles retirados:** selects estáticos largos de métricas/parámetros en ranking, Parameters y
   X/Y/Z; segundo catálogo de checkboxes/búsqueda para métricas adicionales; select plano de categorías.
   Los valores ocultos solo adaptan el renderer: no quedan listas gigantes de opciones ocultas.
2. **Capacidades conservadas:** ranking, varias métricas/unidades, heatmaps/3D/superficies observadas,
   coordenadas paralelas, marcas parciales/pruned, familias, comparación de candidatos, modelos
   persistidos, vistas/import/export, notas, favoritos/recientes y dashboard individual de Run.
3. **Pestañas:** Findings & evidence pasa a Evidence. Conserva conclusiones/auditoría de seeds/poda/
   surrogate; los hallazgos exploratorios quedan en Research. Las otras pestañas mantienen su finalidad.
4. **Selector:** un catálogo y diálogo dinámico, resultados acotados, búsqueda semántica/aproximada,
   teclado, favoritos/recientes. Los parámetros muestran tipo/dominio/activación, no solo nombres.
5. **Explore:** Analyze / By / Compare with / subconjunto, chips eliminables y elección de gráfica.
   Ejes X/Y/Z, tipo, agregación, paleta y normalización siguen disponibles en Advanced.
6. **Preguntas:** tarjetas con etiqueta/prioridad, estado legible, soporte/motivo e Inspect antes de
   los hallazgos exploratorios. label/priority afectan presentación, no política científica.
7. **Jerarquía:** árbol de categorías desplegable; seleccionar un padre incluye descendientes.
   Las dimensiones de familias de métricas no se confunden con categorías o condiciones de parámetros.
8. **Gramática:** `when: {parent: scalar}`, `{parent: {eq: scalar}}` o
   `{parent: {in: [scalar, ...]}}`; varios padres significan AND. Listas ambiguas/operadores ajenos fallan.
9. **Representación:** predicados ActivationCondition inmutables dentro de ParameterDescriptor;
   orden canónico de padres/miembros, serialización escalar histórica y persistencia JSON.
10. **Autoridad:** normalización Work, sweep exhaustivo, RandomSearch, geometría ParameterSpace/
    Sobol/adaptativa, codificación/decodificación/validez y activación en análisis comparten predicados.
11. **Pertenencia:** valores finitos, únicos y no vacíos comprobados contra el dominio del padre;
    sin OR implícito, expresiones recursivas, expansión del dominio ni API específica de un consumidor.
12. **Igualdad:** eq explícito se normaliza al escalar antiguo. Igualdad sobre rangos numéricos y
    orden declarado de dominios independientes conservan comportamiento; las regresiones protegen identidad.
13. **Inactivos:** claves ausentes de candidatos/argumentos; se aplican defaults de la firma. Sin
    null, categoría inactiva inventada o serialización de valores irrelevantes de hijos.
14. **Duplicados:** enumeración topológica activa evita multiplicar ramas inactivas; valores repetidos
    del sweep fallan en vez de duplicar identidades de evidencia silenciosamente.
15. **Fixture:** seis ramas genéricas dan 1+2+10+16+9+18 = **56 candidatos únicos** × cuatro seeds
    compartidas = **224 Runs obligatorios**, iguales a la unión pretendida de ramas independientes.
16. **Referencia:** el selector existente debe identificar exactamente un candidato generado; cero/
    varios resultados fallan. No se inventan referencias a valores inactivos.
17. **Preflight:** validate/explain/dry-run muestran diseño/seeds/Runs/ramas/referencia, paralelismo,
    GPUs y presupuestos. Dry-run humano no enumera cientos de Runs; JSON conserva el plan completo
    y la observación de capacidad del destino.
18. **Tiempo:** wall-time del scheduler y presupuesto lógico de dispatch siguen separados. Los
    niveles secuenciales suman; los paralelos usan el máximo. Pruebas de 1008 h/42 días y composición mixta.
19. **Capacidad:** sondeo directo UUID/visibilidad acotado rechaza peticiones GPU imposibles conocidas
    antes de empaquetar/enviar. Fallo/ambigüedad permanecen desconocidos; ocupación no es capacidad.
    Scheduler/site-command mantienen su concesión existente; preflight nunca reserva ni amplía GPUs.
20. **Steps:** with/seeds/replicates/search/sweep/execution/objective/analysis raíz fallan explícitamente
    en vez de ignorarse. resources raíz mantiene su herencia documentada.
21. **Hook consumidor:** no se añade. Se reutiliza validación de clase/firma/tipos/marcadores sin
    construir Work. Semántica arbitraria necesita validación del consumidor; no era necesario crear
    otro contrato de callbacks para este cambio focalizado.
22. **Esquema/versión:** condiciones y label/priority opcionales añadidos; se rechaza política raíz
    sin efecto. Implementado sobre 0.16.0; análisis v8 y vistas v1 compatibles. Sin otra versión YAML,
    runner, política científica o DSL paralela de condiciones.
23. **Compatibilidad:** no se reescribe evidencia histórica congelada. Identidad escalar, seeds,
    objetivo/HPO/recursos, retry/export y HTML/vistas antiguos se conservan. Las condiciones viajan
    por StudyDesign, inicialización e identidad de recuperación existentes.
24. **Pruebas:** gramática/AND/dominio/ciclos/orden, inmutabilidad/pickle/esquema/identidad, sweep/
    referencia, geometría Sobol/random/análisis, steps/raíz/tiempos y capacidad previa al submit sintética.
    Chromium cubre 300 métricas/15 parámetros, jerarquía, preguntas, selector/chips, gráficas nuevas y
    previas, comparación y recarga de vistas. Ruff/mypy pasan; pytest completo: **1136 passed**,
    cuatro avisos no fatales de Lightning. Revisión final análisis/navegador/config/docs: **63 passed**;
    política GPU/config/docs/smoke de entrenamiento: **34 passed**. Wheel, instalación aislada y
    scaffold/validate/dry-run también pasan reutilizando dependencias instaladas del entorno local.
    Integración CUDA local explícita: **1 passed**. La regresión de padre numérico comprueba que
    las observaciones no transforman un rango declarado en dominio finito.
25. **Límites:** sin DSL de expresiones, nuevo fitting, fusión automática de pasos, lint heurístico de
    particiones o reglas de consumidor. Capacidad incierta permanece desconocida; in necesita dominio
    finito. Conteos adaptativos describen una ventana, no garantizan ejecución. HTML sigue offline;
    los controles no programan/podan Runs ni modifican evidencia.

Verificación local/sintética. No se editó ningún proyecto consumidor ni se envió ningún Work remoto.


<a id="guide-coordinated-studies"></a>

## 29. Studies coordinados: estado y contratos

[English](MANUAL.md#guide-coordinated-studies)

<a id="guide-coordinated-studies-estado-lanzamiento-fleet-adaptativo-nuevo-y-fijo-recoverytransferencia-pendientes"></a>

### Estado: lanzamiento Fleet adaptativo nuevo y fijo; recovery/transferencia pendientes

`lf run CONFIG --on-fleet FLEET` encola un coordinador local durable para Runs repetidas,
sweeps fijos, sweeps pareados automáticos y HPO adaptativo nuevo sin continuación de fidelity.
El selector de destino de Run Work en la Consola ofrece
la misma ruta `fleet:NAME`. El YAML científico no cambia. Es una **capacidad pública limitada**,
no la finalización de la petición completa de HPO adaptativo coordinado.

La ruta de un clúster y `MultiClusterSubmissionService` independiente se conservan.
Los envíos independientes no son HPO coordinado.

<a id="guide-coordinated-studies-implementado"></a>

#### Implementado

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

<a id="guide-coordinated-studies-comandos-disponibles"></a>

#### Comandos disponibles

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

<a id="guide-coordinated-studies-pendientes"></a>

#### Pendientes

Stream de checkpoints/continuación y recovery distribuido; predictive/lookahead;
co-location; pause/resume/reconcile públicos y adopción; recovery tras reinicio del coordinador;
expansión/drain de miembros en vivo; equivalencia de datasets; transferencia de checkpoints/
artifacts, compresión/reanudación y export distribuido; TUI/HTML completos de placement/utilización;
StudyDecision y dependencias. Los controles internos de lifecycle/catálogo no sustituyen esas
operaciones públicas. Retry Fleet rechaza reiniciar como un solo clúster; export rechaza generar
un paquete del coordinador que omita silenciosamente la evidencia de los miembros.

La petición **no está terminada**. Esta ruta limitada no es HPO adaptativo distribuido listo para
producción completo; las pruebas loopback/fake no acreditan todos los proveedores.

<a id="guide-coordinated-studies-validación-incremental-2026-10-05"></a>

#### Validación incremental (2026-10-05)

Regresiones del stream cubren lecturas acotadas de varios chunks, compresión, replay, registros
incompletos, interrupción del receptor, leases ajenas y symlinks. Regresiones del miembro cubren
poda dirigida, idempotencia, órdenes inmutables y evidencia protegida. La integración pública CPU
loopback ejecuta el planner adaptativo nativo sobre dos allocations con un único estado HPO/análisis.
La poda histórica se prueba con espejos verificados y rutas del propietario deliberadamente ausentes.
No acredita aceptación adaptativa remota/GPU real ni recovery tras reinicio del coordinador.

<a id="guide-coordinated-studies-validación-de-esta-implementación-2026-10-04"></a>

#### Validación de esta implementación (2026-10-04)

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

<a id="guide-coordinated-studies-catálogo-de-fleet"></a>

### Catálogo de Fleet

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

<a id="guide-coordinated-studies-propiedad-y-persistencia"></a>

### Propiedad y persistencia

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

<a id="guide-coordinated-studies-pruebas-reproducibles-de-esta-base"></a>

### Pruebas reproducibles de esta base

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


<a id="guide-operational-reliability"></a>

## 30. Salud operativa, monitorización y recuperación deliberada

**Locks de preparación huérfanos.** Un controlador detenido puede dejar el marcador de construcción
de entorno/runtime aunque no haya estudios activos. La preparación actualizada verifica host y
vida del proceso controlador, y comprueba que no quede un instalador del prefijo exacto en destino.
Solo retira el marcador huérfano sin cambios bajo el lock GC nativo, nunca entornos ni ficheros
científicos. La antigüedad no basta: propietarios vivos, ajenos, ilegibles o cambiantes siguen
protegidos. Requiere el controlador actualizado; no relanza Jobs por sí mismo. Si falló totalmente
en preparación, vuelve a enviar el YAML: no hay Execution científica que reanudar. Para recuperaciones
de entrenos ya iniciados sigue existiendo `lf retry`.

Español · [English](MANUAL.md#guide-operational-reliability)

Estado lógico, evidencia científica y salud operativa son conceptos distintos. Un registro antiguo
`running` no demuestra que hoy existan sus procesos. `UNKNOWN` expresa propiedad u observación
sin resolver: no demuestra fallo científico ni autoriza duplicar el cálculo. Se reutilizan el
JobStore del proyecto, los proveedores y el runner nativo. El
[registro de aceptación](ai/IMPLEMENTATION.md#history-operational-reliability-progress) detalla la reforma aún pendiente.
La [guía de recursos](#guide-resource-safety) documenta observación compartida, coordinación
GPU/MIG, RAM cgroup y los límites conservadores de propiedad de reservas externas.

<a id="guide-operational-reliability-inspección-sin-cómputo"></a>

### Inspección sin cómputo

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

<a id="guide-operational-reliability-monitor-independiente-del-proyecto"></a>

### Monitor independiente del proyecto

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

<a id="guide-operational-reliability-correo-opcional-e-incidentes-sin-spam"></a>

### Correo opcional e incidentes sin spam

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

<a id="guide-operational-reliability-consola"></a>

### Consola

Bare `lf` → **Operational health…**, o Ctrl+P → Operational Health System: salud guardada,
detalle de incidentes y configuración. Refresh providers es explícito; Start/Stop sólo controlan
el observador. Save settings comparte validador CLI. Send test email muestra aceptación o fallo.
Durante una petición se impide repetirla. Cerrar el panel no cancela ciencia. Overview/Work/Studies
separan salud y estado; los imports son históricos de sólo lectura, sin proveedor activo esperado.

Monitor y consola coordinan sondas por proyecto/proveedor sin espera larga. Una consulta concurrente
conserva registros nativos y relojes originales; receipts no prueban frescura del proveedor. Jobs
SLURM distintos mantienen carriles independientes. Morir el observador libera el lock del SO y no
deja una barrera permanente para consultar.

<a id="guide-operational-reliability-recuperación-y-yaml-trasladado"></a>

### Recuperación y YAML trasladado

#### Recuperación de Works secuenciales

`lf retry JOB --config pipeline.yaml --dry-run --json` admite `steps` con un Work ejecutable sin
semillas por nivel secuencial. La acción **Retry / Recover Work** de la consola usa el mismo plan.
`strategy: sequential` muestra cada paso como `reuse`, `retry` o `pending`, Run original, siguiente
Attempt e inventario registrado de checkpoints. No se inventa estado de HPO.

Se conservan Execution y checkpoints. Los predecesores completados se reconstruyen desde valores,
artifacts y datasets publicados, sin volver a ejecutar su ciencia ni depender de un retorno Python.
El worker comprueba hashes de outputs e identidades exactas antes de reutilizarlos; el preview solo
inspecciona metadata, sin descargar assets pesados. El paso fallido/interrumpido crea un nuevo
Attempt del mismo Run; los posteriores esperan a su éxito. `resume_map` conserva registros válidos
y comprueba dependencias/validadores, recalculando únicamente lo ausente o inválido. No garantiza
continuación por epoch ni compatibilidad universal de checkpoints del consumidor.

Se preservan identidad científica, recursos raíz/por paso y presupuestos consumidos.
`--accept-code-change` reconoce una corrección de código, nunca cambios científicos. Siguen vigentes
propiedad del proveedor, locks y protección frente a limpieza. Niveles paralelos y composiciones
con Studies se rechazan como no soportados, no como corruptos. Un Attempt antiguo interrumpido sin
evidencia suficiente de propiedad se bloquea, nunca se reinicia silenciosamente. Despliega LF
corregido mediante preparación gestionada, sin parchear el entorno original ni mover checkpoints.
Tras revisar el plan: `lf retry JOB --config pipeline.yaml --accept-code-change`.

#### Work único con productores en checkpoints de archivo

**Nombres históricos del staging acotado.** Retry usa el contrato versionado `bounded-role-v1`
cuando el recibo del bundle original y los inputs del Attempt demuestran el renombrado ligado al
rol (`train.txt` → `inputs/0000-train.txt`). Comprueba bytes originales contra SHA-256 raw del bundle,
rol/configuración y fingerprint histórico. El YAML local o el nuevo staging deben aportar esos
mismos bytes; se reconstruye la identidad antigua sin renombrar datos, reescribir la Execution ni
cambiar globalmente los hashes legados. Dos roles con basename igual no se emparejan por nombre.
Los nombres relativos dentro de directorios siguen siendo contenido científico. El preview muestra
rol, nombre histórico y digest raw; un cambio informa hashes esperado/observado. Sin procedencia
original válida no se autoriza esta adaptación. Preview, dependencias y worker comparten autoridad;
aceptar código no evita verificar contenido/diseño. Los checkpoints mantienen su propietario y se
comprueban mediante acceso nativo y validadores científicos del consumidor.

La recuperación repetida conserva el binding autorizado del **Attempt 1** y su bundle original.
Los Attempts posteriores validan sus propios directorios, bundle, roles y bytes; no deben apuntar
físicamente al Job original. Deben acreditar un request del mismo emplazamiento dirigido a la
Execution original y una entrada aceptada en el historial nativo. No se omite un Attempt corrupto
porque otro resulte válido. Se admiten recibos legados v3/v4 y fingerprints con basename histórico
cuando se acreditan propiedad y contenido. Evidencia original ausente, linaje ajeno o bytes cambiados
bloquean con Attempt/propietario/rol concretos, sin sugerir cambiar ciencia ni borrar Attempts.

El preview envía el helper actual para lectura; **no actualiza** el entorno remoto. Antes de enviar
ciencia, el intérprete preparado debe acreditar `bounded-role-chain-v2`. La preparación gestionada
normal instala el wheel exacto actual en otro entorno inmutable; un intérprete antiguo/existente se
rechaza con indicación de actualizar/bootstrap, sin parchear el entorno histórico ni lanzar el Job.

`lf retry JOB --dry-run --json` admite un Work ejecutable sin seeds/search/sweep. El diseño se
verifica en la Execution/configuración resuelta y sus registros Run/Attempt, sin HPO ni metadata
Study inventada. La Run fallida/interrumpida crea otro Attempt de **la misma Run lógica y
Execution original**. Se vuelve a entrar en `run(...)` con su colección pública `self.checkpoints`.
El Work restaura productores completos mediante `checkpoints.file(..., validate=...)` y construye
los que falten/fallaran. No implica continuación de epochs ni que LF conozca el grafo de
productores de la aplicación. Un límite preventivo de asignación no demuestra OOM del sistema.

El plan de metadata informa `strategy: single`, `work_status`, propietario Job/Execution/Run,
`next_attempt`, número/bytes registrados, archivos ausentes/tamaño inválido, confirmación de código,
`effective_resources` y `resource_changes`. **No verifica hashes de payload** en preview: el acceso
gestionado comprueba hashes y la validación del Work verifica física/schema. No infiere cuántas
proteínas/productores faltan a partir de claves opacas. Recibos corruptos/ajenos y rutas inseguras
bloquean recuperación; corrupción de igual tamaño se detecta al acceder. Solo tras validar todos
los productores debe publicarse el dataset; para assets en checkpoints se usa la API nativa
`outputs.dataset(..., source_checkpoint="producers")`, sin copiar rutas privadas ni publicar
resultados parciales como completos.

En esta ruta de Work único se pueden cambiar explícitamente `resources.cpu`, `resources.memory`
y `resources.time` en el YAML revisado. Los usan tanto el nuevo Job como el Work y quedan en la
auditoría; el origen inmutable conserva su solicitud antigua. No habilita cambios de topología
GPU/procesos/storage. Parámetros físicos, bytes de entradas tipadas, identidad del dataset,
objetivo, diseño y contratos de publicación siguen intactos, incluso con `--accept-code-change`.
El nombre implícito de un YAML que staging renombró a `config.yaml` se normaliza al namespace
histórico, sin modificar cómo se resuelven entradas ni omitir comprobaciones científicas.
Cambios compatibles de implementación/defaults de guardas requieren confirmación explícita de
código; LF no decide que un parámetro físico cambiado es meramente operativo.
Los presupuestos originales `execution.max_runs`/`execution.max_time` no se reinician: si están
agotados bloquean recuperación. Cambiar `resources.time` por Attempt no devuelve cómputo gastado.

```sh
# Solo inspección: no crea Jobs, prepara entornos ni descarga checkpoints.
lf retry JOB --config experiments/preprocess/enrich_structures.yaml --dry-run --json
# Solo después de revisar compatibilidad/recursos y decidir lanzar:
lf retry JOB --config experiments/preprocess/enrich_structures.yaml --accept-code-change
```

Un solve de entorno exacto sin preparar es un requisito posterior, no pérdida de checkpoints.
Apply prepara el runtime revisado y vuelve a verificar propietario, lock nativo del controlador,
hash del YAML y duplicados bajo el lock existente. Los Jobs propietarios/dependencias siguen
protegidos de limpieza. La consola ofrece **Recover Work…** con el mismo plan, selector de YAML
y checkbox de código; el primer click solo inspecciona, y rechazo/preparación aceptada tienen
feedback visible.

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

Para Studies de un solo host lanzados mediante `gpu exec`, recovery usa la concesión nueva de
`CUDA_VISIBLE_DEVICES`, no la lista del Job anterior. Si el mismo YAML pide tres GPU pero
el proveedor concede una, los estudios adaptativos, seeds repetidas y sweeps fijos trabajan
solo en esa GPU: `runs_per_gpu` se aplica al número actual y `max_parallel` sigue siendo un
techo. Se conservan Runs completados, checkpoints, decisiones, costes y registro original de
recursos pedidos. No necesitas editar YAML solo porque ahora tengas menos GPU. Una visibilidad
CUDA explícitamente vacía/deshabilitada nunca se convierte en índices físicos: un Study GPU
falla de forma segura con un diagnóstico de concesión. Esto no reescribe una topología DDP/
multi-GPU programada por el consumidor. SLURM/asignaciones exclusivas mantienen el contrato
exacto: la capacidad insuficiente espera o se rechaza de forma segura, nunca se usan GPU ajenas.

La modal conserva la casilla de compatibilidad, el motivo de bloqueo y la confirmación fuera
del plan desplazable. Pulsar Resume sin la aprobación exigida o un YAML validado explica por
qué no se envió nada; no implica aprobación. Validar deshabilita el envío hasta recibir respuesta,
aunque se cambie la casilla mientras tanto. Tras confirmar explícitamente se envía como máximo
una recuperación y el resultado aparece en el workspace o en un diálogo persistente de fallo.

Se fija hash entre preview y preparación; editar el archivo invalida el plan. Preview no crea
Jobs, consume seeds/presupuesto, reserva GPUs ni descarga checkpoints. Apply verifica de nuevo
propiedad dentro del lock. Propiedad inaccesible o procesos potencialmente vivos bloquean
duplicación. Pérdida verificada reciente permite recuperar sin inventar un resultado failed.

La verificación directa acota los procesos inaccesibles al Job exacto: una denegación de acceso
del mismo usuario no provoca incertidumbre global en el host. La fecha de creación permite
descartar procesos anteriores al Job; en POSIX, evidencia de sesión/UID/vida puede demostrar
una sesión de acceso independiente de propietario distinto. El nombre del proceso no basta.
Supervisor, proceso científico o huérfanos marcados vivos siguen bloqueando recovery, al igual
que propietarios inaccesibles y procesos cuya independencia no se puede demostrar. Inventario
sondea una vez y evalúa esa evidencia para cada Job, sin señales ni cambios en registros
científicos. Reinicia una consola ya abierta tras actualizar LF local. La sonda de lectura
también aplica a entornos remotos inmutables antiguos sin instalar nada dentro de ellos.

Tras reiniciar el host, la sonda compara las fechas de creación de los propietarios con
el arranque del SO. Un propietario verificado anterior al arranque no puede sobrevivirlo;
procesos de acceso fechados en el nuevo arranque no son descendientes suyos. No se cambia
el resultado cancelado/fallido ni se autoriza recovery por un `htop` vacío: una marca viva
del Job, fechas/arranque desconocidos o un propietario posterior inaccesible siguen
bloqueándolo. CLI y consola usan la misma sonda nativa. Reinicia una consola ya abierta
para cargar la sonda actualizada.

La aceptación genera `recovery_request_id`, enlazado desde eventos del original y metadata/eventos
del nuevo Job. `lf jobs show NEW_JOB --json` expone fase: aceptada, preparación, admisión, runtime
observado, progreso científico observado o resultado terminal. Admisión/epoch cero no son progreso.
Fallos conservan fase/categoría/log. Consola muestra ID y fase observada.

<a id="guide-operational-reliability-historial-guardado-incluidos-rechazos"></a>

#### Historial guardado, incluidos rechazos

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

<a id="guide-operational-reliability-política-y-evidencia-de-checkpoints-gestionados"></a>

### Política y evidencia de checkpoints gestionados

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

<a id="guide-operational-reliability-emergencia"></a>

### Emergencia

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

<a id="guide-operational-reliability-ampliación-deliberada-frente-a-recuperación"></a>

### Ampliación deliberada frente a recuperación

Añadir valores/seeds a un sweep fijo utiliza **Extend Study**, no Resume. El delta tiene su propia
Execution/Job; revisiones anteriores y gasto físico de Attempts/tiempo permanecen intactos.
Si falla, su Job admite `lf retry` / **Resume Study…**, conservando Execution y Runs correctos.
La serie sigue la genealogía exacta sin duplicar evidencia. Job ausente con preparación/estado
del proveedor existente es incertidumbre, no permiso para reenviar. Solo un hueco acreditado
anterior al Job puede recuperar el ID preasignado. Continuar una epoch requiere soporte explícito
de checkpoints de la Work. Consulta [ampliaciones](#guide-study-extensions).

El preview utiliza por defecto `--code-policy revisioned`: código modificado/dirty puede añadir
celdas mediante el despliegue nativo gestionado, con procedencia de código/runtime e inferencia
separadas por revisión. No atribuye resultados anteriores al código nuevo. `--code-policy strict`
mantiene la equivalencia obligatoria. El código revisado queda fijado durante la preparación, y se
verifica el contrato semántico y la identidad Execution del host antes de enviar al proveedor.


<a id="guide-resource-safety"></a>

## 31. Seguridad de recursos compartidos — implementación en curso

Español · [English](MANUAL.md#guide-resource-safety)

Trabajo sin publicar sobre 0.19.0: no constituye aceptación de la reforma completa ni explica
el reinicio de un servidor. No se ha conectado a CiTIUS, cambiado reservas, ejecutado estrés
ni lanzado cómputo científico para esta implementación. Véanse la
[auditoría y tareas pendientes](ai/IMPLEMENTATION.md#history-resource-robustness-audit).

<a id="guide-resource-safety-observar-no-concede-gpu"></a>

### Observar no concede GPU

Runner, supervisor y servicio directo de recursos reutilizan una caché del mismo usuario/host.
La medición ordinaria usa el binding oficial opcional `nvidia-ml-py` (importa `pynvml`), sin
PyTorch, tensores, asignaciones ni contextos CUDA de cómputo. Los proyectos CPU no adquieren
dependencias GPU obligatorias. Sin el binding se agrupan consultas acotadas a `nvidia-smi`
dentro del mismo observador aislado; atribución ausente no significa GPU sin procesos. MIG
requiere la capacidad medida de la instancia: nunca se sustituye por la de su tarjeta padre.

Por defecto: TTL informativo de 5 s, de admisión de 0,5 s, intervalo mínimo de 0,25 s, timeout
del helper de 5 s y backoff de 2 a 60 s. Solo un proceso refresca ante misses concurrentes;
los hits no crean helpers. Se registran consultas, hits, fallos, timeouts y latencia. El timeout
termina exclusivamente el grupo privado del helper y verifica su salida; propiedad incierta
impide crear otro. La caché está limitada a 1 MiB. Fallos de escritura/propiedad no autorizan
arranques. Los valores antiguos son informativos, no evidencia fresca para admitir.

Raíz privada del UID, por host/arranque e independiente del Project y entorno virtual:
`$HOME/.cache/lambdaforge/hosts/<hash-host>/`. El operador puede elegir la base con
`LAMBDAFORGE_HOST_COORDINATION_ROOT`; los proyectos deben usar la misma. Se rechazan enlaces,
propietarios ajenos y permisos inseguros. Es metadata operativa, no identidad científica ni
un segundo almacén de resultados. No debe editarse manualmente durante ejecución.

Una concesión UUID GPU/MIG completa no necesita CUDA. Los tokens numéricos antiguos requieren
**una enumeración inicial acotada de identidad mediante el driver** (`cuInit` y
`cuDeviceGetUuid_v2`, sin crear contextos de cómputo). Inicializa el driver, pero no es el
sondeo periódico de memoria. Se fija por vida del controlador, arranque, visibilidad original
y orden CUDA. No se equiparan índices CUDA con NVML ni se sustituye una GPU desaparecida.
Tras establecer el baseline, un dispositivo sin medición no bloquea los arranques de otro
concedido y sano; sus valores antiguos se excluyen de admisión/actualización predictiva y
se marcan no disponibles.

<a id="guide-resource-safety-startup-entre-studies"></a>

### Startup entre Studies

**Inferencia en vivo e inicialización lenta.** Una Run nativa puede tardar mucho en cargar o
validar científicamente datos del consumidor antes de su primera asignación CUDA. El consumo
cero exacto en esa fase no es una muestra de demanda GPU nula: sigue siendo una Run viva, con
propietario y una asignación futura sin validar. No permite liberar su slot ni lanzar workers
sin límite. El diagnóstico distingue `materialized_runs` y `unmaterialized_runs`, incluye
`elapsed_seconds` de cada residente y expone `capacity_wait_reason` del planner real (por ejemplo
`UNVALIDATED_LADDER_STEP` o `PROTECTED_LANE`). El log de espera, limitado en frecuencia, también
muestra workers sin asignación materializada.

Las trayectorias físicas de cada Run son la autoridad de inferencia en vivo. Su pronóstico
anterior **no** es evidencia histórica independiente: solo resultados compatibles registrados
aportan el prior de residuos históricos, una vez. Sondeos con idéntica asignación/progreso no
pueden modificar recursivamente su distribución de memoria. El predictor de placement sigue
compartiendo prefijos medidos y censurados entre candidatos/GPU compatibles; su predicción
puntual es la mediana ponderada, no un pico exacto ni un límite OOM obligatorio para vecinos.
Los workers sin asignación no contaminan esa predicción. Las colas de fases futuras desconocidas
se conservan: una envolvente superior de toda la GPU no equivale a una asignación física.
Siguen vigentes memoria libre real, mínimos declarados, packings fallidos exactos, exploración
protegida y comprobaciones frescas antes del lanzamiento.

Esto no acelera una validación científica dentro de `Work.run` del consumidor. Ese código debe
reportar progreso nativo y evitar decodificar/revalidar repetidamente los mismos assets inmutables
cuando su contrato científico lo permita; LF no omite sus comprobaciones implícitamente. Los Jobs
existentes usan runtimes gestionados inmutables y no recargan correcciones del checkout. Hay que
desplegar el wheel corregido mediante preparación gestionada normal antes de una recuperación
solicitada explícitamente; no editar el entorno activo, registros científicos ni datos para
forzar admisión.

`shared` y `command` coordinan arranques mediante locks cortos por UUID GPU/MIG. La autoridad
sigue siendo `GPUPlacementPlanner`, que comprueba memoria física fresca bajo el lock. El
compromiso temporal se persiste **antes** de crear el worker e incluye las identidades nativas
disponibles de controlador, worker, Job, Execution, Run y Attempt. Un error de escritura no
puede ocultar un Future ya arrancado. Otras GPU conservan admisión independiente.

Demanda pendiente de otro Study: `max(compromiso de startup - consumo observado, 0)`. La VRAM
libre ya excluye el consumo residente; no se resta otra vez. El controlador ya contempla sus
propias Runs con sus envelopes. Se reconcilian vidas de worker verificadas como ausentes;
propietarios inciertos siguen protegidos y un fichero de PID ausente no demuestra muerte.
Para huérfanos se puede reutilizar la observación nativa de propiedad del Job; la edad no es
prueba. Si se eliminó todo el directorio del Job antiguo, solo se retira su compromiso operativo
tras confirmar la muerte/reutilización del controlador y comprobar mediante la inspección nativa
que no quedan descendientes con su marcador de Job ni procesos relevantes de propiedad incierta.
Esto no recrea registros, autoriza recovery ni evita comprobar reservas y capacidad física actual.
Si el directorio existe pero faltan registros individuales, la evidencia sigue siendo incierta o
corrupta, no una prueba de ausencia.
La ventana de startup de un worker vivo atestiguado es por defecto 60 s, finita, no
una reserva permanente de `gpu_memory` por residentes. Evidencia compatible de tiempo hasta
el envelope puede extenderla, con límite de una hora. Baseline y exploración no pueden ignorar
startup pendiente de otro Study. No cambian techos ni decisiones científicas. Waiters acotados
por dispositivo incluyen vida nativa, prioridad, antigüedad y arranques servidos: un Study
antiguo factible tiene oportunidad, sin alternancia obligatoria ni bloquear trabajo ligero
tras otro que no cabe. Los waiters caducan; los compromisos requieren reconciliación más fuerte.
No se comparan objetivos de Studies diferentes. La equidad entre usuarios pertenece al proveedor.

<a id="guide-resource-safety-ram-y-cpu"></a>

### RAM y CPU

Se observa RAM física y límites legibles de ancestros cgroup v2 mediante `memory.max` y
`memory.current`, junto con presión y eventos OOM disponibles. No se escriben controles
cgroup. Evidencia inválida queda no disponible, no consumo cero. Se usa capacidad efectiva
para admitir Jobs y se rechazan peticiones imposibles antes de calcular. Antes de iniciar
una Run GPU se comprueba otra vez disponibilidad frente a su share RAM nativo de dispatch.

La atribución verifica nacimiento del PID y UID. RSS, PSS y USS son distintos: sumar RSS
puede duplicar páginas compartidas. Solo PSS cualificado reduce el compromiso RAM todavía
no materializado; si falta, se conserva conservadoramente. `ram_bytes=0` se señala como
demanda desconocida. Propietarios inaccesibles o con argv cambiado retienen protección CPU/RAM;
un exec no demuestra muerte. Se mantienen afinidad y límites nativos de threads existentes.

Los leases directos guardan host/boot bajo `hosts/<host-id>` en la raíz de leases configurada.
Las claves GPU y la visibilidad del hijo usan UUID GPU/MIG exactos, no ordinales CUDA/NVML.
Un lease exclusivo del dispositivo padre bloquea sus instancias MIG; las instancias independientes
siguen siendo independientes. El descubrimiento queda fuera del lock y la liberación usa
propiedad retenida, no una topología que puede haber cambiado. Se conserva el namespace del lock
exterior para coordinar supervisores antiguos y nuevos. Los registros antiguos sin host/boot o
con índices ambiguos quedan protegidos aunque un PID local parezca muerto: el archivo puede ser
de otro host con home compartido. El Job en cola explica esa espera. Un supervisor antiguo activo
normalmente libera su lease al salir; un resto tras un crash sin evidencia de host/boot necesita
reconciliación explícita, no borrado ciego.

No cambian DataLoader, fidelidad, seeds, checkpoints ni resultados. `cpu.max` y
`cpuset.cpus.effective` legibles estrechan capacidad, sin ampliar afinidad. Las cuotas fraccionarias
se explicitan y se acotan por su techo. Se descuenta overhead del controlador antes de repartir
RAM agregada. Predictor RAM online y backpressure por presión de procesos siguen pendientes.

<a id="guide-resource-safety-perfiles-circuitos-e-informes"></a>

### Perfiles, circuitos e informes

Errores de memoria/desaparición de UUID abren circuitos de observación por dispositivo que
caducan (5–120 s por defecto). Se suprimen consultas NVML de memoria/actividad hasta recuperación;
sigue siendo necesario enumerar identidades. Una lectura cualificada nueva del mismo UUID cierra
el circuito. La dominancia OOM por candidato/placement sigue separada: un OOM no impone techo
permanente. CUDA genérico no demuestra ECC, Xid, reset ni autorización para reintentar.

Ajustes operativos del cluster, no YAML científico:

```sh
lf clusters set CLUSTER resource_safety shared-host
lf clusters set CLUSTER resource_safety.priority 2
lf clusters set CLUSTER resource_safety.probe_timeout 3
lf resources --report --on CLUSTER --since 2026-10-13T00:00:00Z --until 2026-10-14T00:00:00Z --json
```

`balanced` conserva defaults; `throughput` usa TTL informativo 3 s/espera fairness 20 s;
`shared-host`, TTL 10 s/espera 15 s; `conservative`, startup 120 s y mayor backoff inicial.
Ninguno modifica parámetros científicos ni techos. Overrides se validan y se propagan por submit
directo/remoto nativo; `resource_safety` se edita en el YAML Advanced de la consola. Los perfiles
existentes usan balanced sin migrar el paralelismo.

Los heartbeats CPU no crean helpers GPU. Nuevos samples `usage.jsonl` rotan a partir de 2 MiB
a cuatro ventanas comprimidas, con muestras recientes y picos acumulados en `usage-summary.json`.
No se compactan métricas, logs científicos ni checkpoints. Un historial antiguo excesivo no se
lee sin límite ni se borra: el estado actual permanece disponible y la limpieza es deliberada.
Los temporales de compresión interrumpidos no bloquean la compactación posterior y no se
adoptan ni eliminan como si fueran evidencia propia de esta operación.

El informe sanitizado de evidencia guardada aparece en cluster **Resources → Saved resource report**
y en `get_resource_report` (`clusters.resource_report`) para sesiones de agentes fijadas al proyecto.
Lee metadata propia acotada, sin consultar proveedor, assets ni logs completos. Excluye comandos,
rutas, errores arbitrarios, credenciales y PID ajenos. Funciona offline/tras reinicio si sobrevivió
metadata. Lo desconocido no se convierte en cero. Los picos son máximos muestreados de toda la
vida del Job, no máximos exactos del intervalo. No fabrica timeline completo, diagnóstico de
driver ni causa de reinicio.

<a id="guide-resource-safety-reservas-del-centro"></a>

### Reservas del centro

En CiTIUS se usa `command_prefix: [gpu, exec]`, con claim persistente manual cuando corresponda.
Dentro del wrapper se pasa Python gestionado absoluto, sin depender de activación Conda.
LambdaForge no ejecuta `gpu release` tras éxito, fallo o cancelación. El proveedor gestiona
las reservas propias del wrapper; las manuales pertenecen al usuario.

Los pares genéricos `claim_command`/`release_command` no aportan un recibo acotado de propiedad.
Su ejecución automática se rechaza **antes del claim**, también en preview. Los perfiles
guardados se leen pero deben retirar el par. Es una limitación conservadora explícita, no
un protocolo inventado de gpuctl. Una integración por reserva necesita verificación real
autorizada; actividad o UUID coincidente no prueban propiedad.

Exactamente `[gpu, exec]` deriva `[gpu, env, --shell]`; comandos personalizados explícitos
no se cambian. El parser admite una línea CSV o asignación explícita `CUDA_VISIBLE_DEVICES`,
nunca ejecuta su contenido y rechaza ambigüedad/salida excesiva. Una concesión vacía confirmada
es distinta de sondeo fallido. Comparar formatos requiere identidades fijadas; no se revoca
por confusión numérico/UUID. Solo se retiran/restauran tokens originales. Los entrenos sanos
no se cancelan por un fallo del observador.

El resume del Study establece una concesión nueva para el nuevo Job. En modo `command`
(`gpu exec`), un Study cuyo YAML pide tres GPU puede continuar con una sola concedida ahora,
sin editar YAML, reutilizar la lista antigua ni repetir evidencia completada. `runs_per_gpu`
numérico se aplica al número actual; siguen activos los controles de memoria física/recursos.
Hay regresiones nativas de recovery adaptativo, seeds repetidas y sweep fijo con hardware
simulado. `CUDA_VISIBLE_DEVICES` explícitamente vacío/deshabilitado falla de forma segura en
cualquier modo, sin inventar índices físicos. Una ejecución CPU sigue siendo válida con GPU
deshabilitada. Las asignaciones del scheduler son contratos exactos; no se reescribe de forma
automática la topología distribuida/DDP de una aplicación.

<a id="guide-resource-safety-aceptación-pendiente"></a>

### Aceptación pendiente

Faltan reconciliación por fases más allá de la ventana startup aprendida, predictor RAM sin
presupuesto declarado, backpressure por presión de procesos, informes históricos completos/
eventos hardware confirmados, reconciliación nativa de leases antiguos ambiguos y aceptación
extendida de proveedores. Los leases exteriores ya tienen scoping host/UUID con regresiones
locales; eso no certifica un despliegue real en el centro/home compartido. Se mantiene
ARI/throughput, no FIFO ni una Run por GPU. Establecer el
baseline inicial todavía requiere todas las identidades solicitadas. NVML/MIG real y la
preempción/limpieza del centro requieren comprobación explícita.
Las pruebas CPU comparan packing estático/dinámico y cold start terminal/en vivo. El microbenchmark
de caché mide overhead local con backend simulado, no latencia NVML ni throughput científico real.
Las pruebas de competencia usan procesos nativos independientes.

Fuentes primarias: [binding NVIDIA](https://pypi.org/project/nvidia-ml-py/),
[orden/visibilidad CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html),
[NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html),
[manual CiTIUS](https://wiki.citius.gal/en:centro:servizos:servidores_de_computacion_gpgpu).


<a id="guide-developer-experience"></a>

## 32. Experiencia de desarrollo — implementación incremental

[English](MANUAL.md#guide-developer-experience)

Este bloque mejora ergonomía nativa sin cambiar identidad científica, YAML, recovery, scheduling
GPU ni las ampliaciones de Studies sin commitear. **La reforma completa no está terminada**.
No se aumenta versión. El registro de aceptación distingue lo probado de lo pendiente.

<a id="guide-developer-experience-empezar-y-descubrir"></a>

### Empezar y descubrir

```bash
lf init my-research --template minimal
cd my-research
python -m pip install -e .
lf validate experiments/work.yaml
lf explain experiments/work.yaml
lf run experiments/work.yaml --dry-run
lf help --search "recuperar estudio" --json
```

Init deriva distribución, paquete/imports, YAML, README y test ligero del nombre de carpeta.
Translitera Unicode cuando es posible, sanea caracteres, prefija números y evita palabras
reservadas y colisiones con stdlib/framework. Los proyectos antiguos no se migran. Sobrescribir
requiere el `--force` existente; symlinks se rechazan. Agentes, memoria e instalaciones siguen
opt-in. Dry-run no ejecuta Work.run. Las instrucciones usan los archivos reales, sin inventar
una ruta al checkout. Pytest solo hace falta para ejecutar los tests generados.

La ayuda por intención se deriva del parser instalado y reutiliza vocabulario bilingüe nativo.
Los resultados son comandos/usage reales, no servicios inventados. `lf help`, `lf --help` y
ayuda anidada siguen funcionando; las erratas sugieren opciones existentes.

```bash
source <(lf completion bash)
source <(lf completion zsh)  # tras compinit habitual del shell
lf completion fish | source
```

Usa solo la línea de tu shell. No se editan archivos de inicio. Completions cubren comandos,
subcomandos, flags y enums del mismo parser, sin proyectos, secretos, Torch ni consultas remotas.
Nombres dinámicos de clusters/datasets no se expanden; el shell conserva fallback de ficheros.
Sintaxis Bash está probada; interacción real Zsh/Fish queda pendiente.

<a id="guide-developer-experience-entender-yaml"></a>

### Entender YAML

```bash
lf config schema
lf config schema --output work.schema.json
lf config explain experiments/work.yaml --field execution.max_parallel
lf config diff experiments/previous.yaml experiments/work.yaml --json
lf config resolve experiments/work.yaml --format json
```

Schema es el instalado, no otra gramática. `--output` crea un fichero nuevo, sin sobrescribir.
Asócialo al YAML manualmente en el editor; no se cambia VS Code. Explicar un campo usa YAML/schema
acotado y ExecutionPolicy nativo para límites, sin importar consumidor. Los parámetros `with`
siguen perteneciendo a Work.run; para ellos usa validate/explain con imports de código confiable.
Schema documenta concurrencia/presupuesto y admite null/auto sin distinguir mayúsculas como el
normalizador. No se afirma paridad exhaustiva de cada alias legado de recursos.

Diff usa WorkConfig y JSON canónico compartido del comparador puro existente, sin modificar su
servicio de ampliación. Muestra cambios de parámetros, seeds, políticas, recursos, análisis y
productos. **No** certifica compatibilidad de recovery/extension, dependencias o bytes de dos
rutas. Importa código confiable pero no ejecuta Work.run. Explain distingue totales, asignación
y techos de admisión; pedir concurrencia no garantiza disponibilidad física futura.

<a id="guide-developer-experience-inspección-diagnóstico-y-seguimiento"></a>

### Inspección, diagnóstico y seguimiento

```bash
lf inspect job-EXACT --json
lf inspect execution-EXACT --kind execution
lf inspect NAME --kind work
lf diagnose job-EXACT --json
lf watch job-EXACT --refresh --interval 2 --count 10 --json
```

Prefijos exactos seleccionan identidades nativas. Nombres ambiguos entre Work/Study/Execution/Job
exigen `--kind` o ID exacto. No se incluyen selectores arbitrarios Run/product/dataset todavía.
Se reutilizan catálogo compacto ResultStore, agrupación Work y JobService/OperationalHealth,
sin abrir modelos/checkpoints/artifacts. `--refresh` pide metadata del proveedor a través del
gate existente; sin él se muestra observación persistida y su edad, no frescura inventada.

Diagnose reúne diagnóstico, fallo registrado confirmado e historial reciente nativo. No inventa
causas posibles ni certifica recovery por FAILED/UNKNOWN. El comando sugerido
`lf retry JOB --dry-run --json` comprueba elegibilidad/propiedad/checkpoint real; nunca se lanza
automáticamente. Tracebacks y análisis profundo de Runs siguen siendo consultas explícitas.

Watch observa en primer plano, no crea monitor permanente. JSON es **NDJSON**, un documento por
snapshot cambiado. Si no cambia, backoff hasta 60s; la edad por sí sola no es evento. Ctrl-C sale
solo del observador, sin cancelar Jobs. No transmite cuerpos de logs ni recursos detallados de
cada Run: conserva `lf logs --follow` e informes existentes para ello.

<a id="guide-developer-experience-tablas-pequeñas-sin-exportar-todo-el-experimento"></a>

### Tablas pequeñas sin exportar todo el experimento

```bash
lf results table execution-EXACT --metric val_accuracy --metric val_loss --json
lf results table execution-EXACT --trial 2 --columns seed,value,parameters --format csv --output scalars.csv
lf results table execution-EXACT --offset 100 --limit 100 --fingerprint sha256:PAGINA_ANTERIOR
lf results table execution-EXACT --attempts all --json
```

`ResultStore.table` proyecta metadata local retenida o importada. Por defecto incluye solo el último
Attempt de cada Run lógico; `--attempts all` conserva también el historial físico. Métricas, IDs de
Run y números de Trial son selectores exactos repetibles. Las columnas opcionales permiten acotar
la tabla, pero las identidades, fingerprint, estado y tipo de evidencia permanecen obligatorios.
Los valores ausentes siguen vacíos; un podado es `censored_snapshot`, no un score final inventado.
Los valores son **snapshots escalares persistidos**, no se infiere mejor epoch ni agregación de
selección. Se lee el agregado de metadata seleccionado, no series, logs, modelos, checkpoints ni
bytes de artefactos. No se recalcula StudyAnalysis ni se descargan artefactos remotos.

JSON contiene aviso descriptivo, número de filas, siguiente offset y fingerprint de evidencia.
Pasarlo en la página siguiente rechaza cambios de evidencia en vez de mezclar snapshots. CSV
conserva procedencia en cada fila y muestra paginación/aviso en stderr; escapa fórmulas de hoja de
cálculo y aplica redacción nativa de secretos. Máximo 1000 filas por página y tamaño acotado: omite
columnas voluminosas si una fila sola supera el límite. El destino explícito debe ser nuevo, sin
symlinks y fuera de la raíz de evidencia Run; nunca sobrescribe. Son tablas derivadas, no autoridad
científica importable. El export/import completo existente no cambia.

<a id="guide-developer-experience-agentes-registro-no-significa-conexión"></a>

### Agentes: registro no significa conexión

```bash
lf ai doctor
lf ai context
lf ai connect codex --json
lf ai connect codex --apply --plan-hash sha256:PLAN_REVISADO
lf ai launch codex
```

Context pagina YAML/Work AST local, clusters, estado de conocimiento/integración/SDK y permisos
READ/PLAN reales. No importa consumidor, lee notas ni sondea clusters. Doctor añade versión/help/
registro Codex instalado con consultas acotadas; ausencias/incompatibilidades son explícitas,
sin instalar nada.

Connect sigue la [CLI MCP oficial de Codex](https://developers.openai.com/codex/mcp/), comprueba
gramática add/get/list y usa nombre propio del proyecto e intérprete invocante. Preview no escribe.
Apply explícito requiere hash actual, SDK compatible y lock nativo; rechaza conflictos y es
idempotente si coincide. Codex gestiona su configuración: LambdaForge no edita TOML directamente
ni reemplaza servidores ajenos. Ediciones independientes de otros clientes no están protegidas
por este lock de proyecto; aceptación real de configuración/comentarios sigue pendiente.

**Verificar registro no es handshake MCP ni conversación con agente.** Se indica en la respuesta.
Conserva `lf ai mcp test` para SDK y `/mcp` de Codex para descubrimiento real. Instalación,
registro y launch son acciones separadas. Launch exige TTY/registro exacto, usa cwd del proyecto
sin elevar sandbox/permisos ni lanzar ciencia. No se modificó configuración global Codex durante
la implementación: registro se prueba con estado simulado desechable.

<a id="guide-developer-experience-consola-y-skills"></a>

### Consola y Skills

Tablas principales Work/Study tienen filtros locales por nombre/estado/cluster/ID. Escribir no
consulta proveedor. Filtrar, abrir y refrescar usan las mismas filas e identidad exacta. Un
snapshot igual actualiza frescura sin vaciar tabla; al cambiar se conserva scroll y selección
existente. Widgets desmontados descartan respuestas tardías sin parar ciencia. Otras pantallas
mantienen sus layouts y controles propios.

Skills empaquetadas Work/Study usan comandos reales, ejemplos completos y decisiones sobre
diseño, evidencia ausente/censurada, imports e inputs explícitos. No autorizan submission ni
prometen compatibilidad universal. La guía nativa define diseños fijos, GPU gestionada en el mismo
destino y revisiones/análisis acumulados.

<a id="guide-developer-experience-pendiente"></a>

### Pendiente

Búsqueda global paginada, sorting/paginación más amplia, elegibilidad recovery consolidada,
comparaciones compatibles/emparejadas entre Executions, fusión guiada del editor,
copiado de recetas, refinamientos HTML, extracción
grande de Workspace, contratos MCP adicionales y resto de Skills prioritarias. No se afirma
GPU/SLURM/SSH real, conversación externa o suite backend completa. Comandos/JSON públicos,
autoridades de dominio y ampliaciones incrementales pendientes siguen intactos.


<a id="guide-integration"></a>

## 33. Integración de agentes fijada al proyecto

[English](MANUAL.md#guide-integration). **La reforma integral todavía no está terminada**:
consulta el [registro de aceptación](ai/IMPLEMENTATION.md). No cambia ciencia, identidades, HPO o GPU.

<a id="guide-integration-inicio-rápido"></a>

### Inicio rápido

```bash
lf init mi-investigacion --agents portable --knowledge markdown
lf ai --project /ruta/absoluta init --agents codex --knowledge obsidian
lf ai --project /ruta/absoluta init --agents codex --knowledge obsidian --apply
lf ai --project /ruta/absoluta doctor
lf ai skills list
lf ai capabilities --json
```

Sin flags, init sigue creando únicamente el proyecto científico. `--interactive` pregunta en una
TTY; no instala dependencias. **Agents / knowledge…** muestra estado, Skills, búsqueda y preparación
con confirmación mediante el mismo backend. La guía consumidora enseña APIs públicas, no internals.
AGENTS raíz inglés es canónico; los contratos detallados siguen en FRAMEWORK_CONTRACTS.md.

<a id="guide-integration-mcp-clientes-y-permisos"></a>

### MCP, clientes y permisos

Instala expresamente `python -m pip install "lambdaforge[agents]"` en el entorno deseado.
Arranca `python -m lambdaforge ai --project /ruta/absoluta mcp serve`. SDK oficial 2.3, stdio;
stdout pertenece al protocolo. `mcp test` usa un cliente real en memoria; las pruebas además usan
un proceso stdio desde otro directorio. Codex 0.162.0-alpha.17.2 ya ha descubierto herramientas y
realizado llamadas reales con configuración temporal aislada y sin credenciales de modelo.
Eso no acredita una conversación LLM ni un segundo cliente de proveedor, todavía pendientes.

Codex genera `.lambdaforge/agents/codex-config.example.toml`; revisa y combina la entrada con
la configuración existente, usando el Python absoluto correcto. No modifica archivos globales.
Toolsets core/work/study/results/products/datasets/clusters/operations/knowledge, por defecto
core/work/knowledge. El catálogo distingue disponible, habilitado en sesión y no soportado.
Los detalles remotos se piden expresamente; listar no descarga todos los artifacts/epochs.

READ/PLAN no autoriza enviar, cancelar, borrar ni escribir como agente. Envío/recovery/borrado
siguen deshabilitados como acciones de agente. Cancelar un Job exacto exige OPERATE, toolset
operations y aprobación humana separada. Notas aprobadas y HTTP local emparejado son opt-in.
Validar importa Python del consumidor y requiere `--allow-project-code` del operador. MCP aísla
planificación en un hijo con timeout/cancelación y sin prints en stdout del protocolo. **No es un
sandbox** contra código confiado o agentes con shell propio. Texto recuperado no concede permisos.
Las respuestas son tipadas, con origen/fecha de consulta/error; consultar ahora no demuestra
observación fresca del proveedor. No ofrece shell ni lectura arbitraria de archivos.

<a id="guide-integration-aprobación-humana-y-dependencias"></a>

### Aprobación humana y dependencias

Cancelación exacta: `prepare_job_cancellation(job_id)` inspecciona metadata sin consultar al
proveedor; `request_job_cancellation(job_id)` crea una propuesta sin detener procesos.
El operador stdio habilita `--toolsets core,operations --permissions READ,PLAN,OPERATE`.
La persona revisa el Job, proyecto, cluster, binding de proveedor y alcance mediante el mismo
CLI/Console de aprobaciones. Después `apply_job_cancellation(request_id, plan_hash)` revalida
e invoca `JobService.cancel`. Solo afecta a ese Job, no a todos los Jobs del Work; preserva
evidencia. Estado incierto o plan cambiado bloquean apply. Fleet se rechaza explícitamente.
`job_cancellation_status(request_id)` permite consultar el resultado guardado. Fallo tras claim
no permite replay; no hay herramienta MCP de aprobar. Permiso/revisión de notas no autoriza cancelar.

Las respuestas mantienen el envelope v1, pero los payloads declarados tienen esquemas específicos
que también anuncia el SDK. Errores de herramienta llevan `isError=true` y códigos distintos:
`APPROVAL_REQUIRED`, `APPROVAL_EXPIRED`, `PLAN_STALE`, `OPERATION_CONFLICT`, incertidumbre/fallo
de proveedor y error interno. `retryable` nunca concede permiso para repetir una mutación.
`lf ai capabilities --search "recuperar estudio" --limit 20 --json` admite sinónimos técnicos
deterministas EN/ES; relevancia de búsqueda no es confianza científica.

El operador stdio puede habilitar `--permissions READ,PLAN,WRITE_PROJECT`. `agent_actions` descubre
`knowledge.create/update/link/export`: export crea un ZIP de notas, nunca exportación científica.
`preview_agent_action` y `request_agent_action` no editan notas ni crean archivos ZIP.
Una persona revisa valores exactos en **Agents / knowledge… → Review agent requests…**, o usa:

```bash
lf ai approvals list
lf ai approvals show approval-ID_EXACTO
lf ai approvals approve approval-ID_EXACTO --plan-hash HASH_EXACTO
lf ai dependencies check
lf ai dependencies install                          # solo preview
lf ai dependencies install --apply                  # virtualenv/Conda actual
lf ai dependencies install --wheelhouse wheels --apply # sin red
lf ai init --agents portable --apply --install-deps
```

CLI exige TTY local y escribir el ID completo; Console muestra todos los argumentos y exige
confirmación equivalente. Aprobar no ejecuta: el agente debe aplicar el plan por separado antes
de caducar (máximo 15 minutos). Proyecto, argumentos, revisiones y hash se revalidan; registros
firmados, claim bajo lock y auditoría rechazan falsificación, cambios y replay/concurrencia.
Un crash tras el claim no permite repetir. MCP no ofrece aprobar, ni admite `approved=true`.
Esto no aísla agentes que dispongan de shell independiente bajo el mismo usuario.

Solo se instala el extra declarado `agents`, con consentimiento y en el intérprete del operador;
nunca Python de sistema, paquetes arbitrarios, aplicaciones de escritorio ni clusters.
Fallos conservan archivos/notas, muestran diagnóstico acotado y exit code de error.

<a id="guide-integration-http-local-opcional"></a>

### HTTP local opcional

`lf ai mcp token --apply` prepara un token privado y no muestra su contenido.
`lf ai mcp serve --transport http --port 8765` usa el mismo SDK/backend, solo READ/PLAN y
`127.0.0.1`. El cliente local autorizado lee `.lambdaforge/agents/http/token` y usa el header
`Authorization: Bearer TOKEN`, nunca una URL. Navegadores requieren además un `--origin` exacto.
Se verifica token/proyecto en cada petición, Host exacto, peer loopback, límites y Origin;
no se confía en headers de proxy. Es emparejamiento local, **no un servidor OAuth** ni exposición
de red. Stdio sigue siendo el transporte recomendado. Rotar el token invalida el anterior.

<a id="guide-integration-memoria-y-sincronización"></a>

### Memoria y sincronización

```bash
lf ai knowledge init --apply
lf ai knowledge create hypothesis "¿Mejora width el score?" --author-type ai --apply
lf ai knowledge search width --kind hypothesis
lf ai knowledge read ID_EXACTO
lf ai knowledge reindex --force
lf ai knowledge reindex --rebuild
lf ai knowledge audit --limit 20  # --cursor usa el next_cursor exacto, en bytes
lf ai knowledge export notas.zip          # preview sin crear el ZIP
lf ai knowledge export notas.zip --apply  # acción explícita del operador local
lf ai knowledge obsidian                  # solo indicios, no arranca la app
lf ai knowledge obsidian --probe-cli --apply # prueba explícita, puede arrancar Obsidian
lf ai knowledge open --apply              # solicita abrir el README propio, sin cambiar ajustes
lf ai sync
lf ai sync --apply
```

Markdown canónico con once tipos de nota, autor humano/IA, fecha, tags, estado, enlaces y proyecto.
Hipótesis IA son propuestas, no ciencia verificada. Enlaza Execution exacta o contenido SHA-256;
un nombre ambiguo no sirve. Comprobar referencia no verifica interpretación. SQLite FTS es un
índice reconstruible que reconoce ediciones externas; revisiones hash rechazan conflictos nativos.
El editor externo no participa en el lock. `knowledge link` enlaza notas existentes con revisión
actual. Un índice corrupto permite reconstrucción explícita y conserva su copia antigua sin tocar
Markdown. IDs duplicados/corrupción de notas generan avisos sin borrarlas.
Las consultas de notas relacionadas usan enlaces/tags/evidencia exacta, no confianza científica.
Hipótesis, decisiones y preguntas abiertas tienen vistas acotadas; verificar referencias consulta
registros nativos locales, sin descargar artifacts ni verificar la interpretación.
Obsidian es opcional, usa los mismos Markdown y preserva preferencias: no necesita escritorio/plugins.
La auditoría nativa guarda intención/finalización, ID y revisiones, no títulos/cuerpos. Intención
interrumpida no significa éxito; la paginación por cursor no lee todo el historial. Ediciones de
Obsidian no se atribuyen automáticamente a una persona ni se convierten en ciencia verificada.
El journal `.lambdaforge/agents/knowledge-audit.jsonl` es independiente del índice FTS desechable:
reindex no borra historia. Un journal corrupto bloquea nuevas ediciones hasta inspección explícita,
sin eliminarlo ni fingir que su historial sigue siendo fiable.
Export comprime solo notas canónicas y un manifiesto con checksums exactos: rechaza corrupción,
duplicados, symlinks, planes obsoletos y sobrescrituras. Conserva los bytes originales; no incluye
datos científicos, índices, plugins ni archivos ajenos. La ruta debe estar dentro del proyecto,
fuera del vault y tener un directorio padre existente. No es `lf export STUDY` ni recovery.
MCP `knowledge_export` solo revisa; aplicar usa `knowledge.export` con aprobación humana separada
y hash del inventario completo. Vaults externos y búsqueda semántica siguen pendientes.

Diagnóstico Obsidian consulta PATH/configuración/indicios de escritorio sin asumir instalación,
versión ni aplicación abierta. La prueba explícita de la [CLI oficial](https://obsidian.md/help/cli)
puede arrancar la aplicación, tiene timeout y salida acotada. Open utiliza la
[URI oficial codificada](https://obsidian.md/help/uri) del README propio: antes hay que registrar
manualmente esa carpeta como vault. Éxito significa `open_requested`, no ventana verificada;
entorno headless, CLI ausente/deshabilitada y fallos dan indicaciones. No instala aplicaciones/
plugins ni cambia ajustes, y MCP no expone abrir escritorio. Se probaron fixtures, no una sesión
real de Obsidian instalada.

MCP ofrece recursos acotados de contratos Work/Study instalados, estructura, recetas, resultados/
productos exactos e índice de notas. Se registran según toolset, nunca para leer rutas arbitrarias.
Trece prompts cortos orientan creación, modelos, checkpointing, comparación, HTML, publicación,
diagnóstico, recovery, revisión de evidencia, decisiones y próximos experimentos, sin autorizarlos.

Las 24 Skills proceden del paquete instalado. Sync preserva AGENTS/Skills personalizados y notas,
usando manifiesto de hashes; rechaza planes modificados. Aplicar desde CLI como operador no sustituye
el futuro protocolo de aprobación del agente. No guarda conversaciones automáticamente.

<a id="guide-integration-python-y-gui"></a>

### Python y GUI

`lambdaforge.agents.AgentServices` da respuestas tipadas fijadas al proyecto; IntegrationService
es el mismo backend de CLI/Console. `examples/agent_gui.py` muestra un cliente Tk opcional sin LLM.
Se prueban protocolo real, aislamiento, sesiones/ediciones externas, conflictos, assets y Work CPU
determinista con métricas, checkpoint, producto y HTML. La explicación inglesa detalla contratos,
fuentes oficiales y límites. WISDOM, clusters y evidencia científica existente permanecen intactos.
Los modelos exponen declaraciones y docstrings reales sin importar Torch; no se inventan shapes.
Contratos/miembros/provenance/consumidores de productos son metadata. Runs/curvas/logs seleccionados
y recursos de un cluster son consultas explícitas nativas; preview de texto verifica ownership e
integridad con límite de 64 KiB. Resto de acciones científicas, semántica, vault externo, cobertura total
de APIs y conversaciones completas siguen en el registro de pendientes.

<a id="guide-graph-science"></a>

## 34. Topología de grafos y calibración independiente

Estos algoritmos genéricos de NumPy se usan directamente desde Python o dentro de un Work
nativo. No conocen proteínas, superficies concretas ni Jobs. Los exports diferidos de `topology`
y `stats` permiten usarlos sin cargar Torch, scikit-learn ni POT. No hay otro motor de ejecución
ni un sistema nuevo de almacenamiento.

### Disimilitudes precalculadas

DBSCAN, HDBSCAN y Agglomerative con `single`, `average` o `complete` admiten
`distance="precomputed"`. Ward, KMeans y MiniBatchKMeans lo rechazan. Se conserva el uso de
características `[N,F]` y de las instancias neuronales `Distance` existentes.

```python
from lambdaforge.clustering import HDBSCAN, silhouette_score
from lambdaforge.distances import validate_dissimilarity

# D puede proceder de fgw.pairwise(objects, symmetric=True).
D = validate_dissimilarity(D, max_pairwise_bytes=512 * 1024**2)
clusters = HDBSCAN(min_cluster_size=10, distance="precomputed").cluster(D)
score = silhouette_score(D, clusters, distance="precomputed")
```

Se exige una matriz densa cuadrada no vacía, numérica real, finita, no negativa y de diagonal
**exactamente cero**. La simetría se comprueba con `symmetry_atol=1e-10`, `symmetry_rtol=1e-8`,
sin cambiar valores ni asumir desigualdad triangular. Para tratar asimetría numérica de forma
explícita puede usarse `validate_dissimilarity(D, symmetry="average")`: sustituye cada par por
su media aritmética, cambiando la disimilitud científica. No mejora la precisión del solver
FGW local. Nunca se recortan negativos, se rellena NaN ni se anula una diagonal automáticamente.

El límite `max_pairwise_bytes`, por defecto 512 MiB, comprueba `N*N*8` **antes** de convertir
a float64 o transferir un tensor a CPU; no limita todo el workspace del backend. HDBSCAN copia
explícitamente su entrada para no modificar la del consumidor. Silhouette conserva `-1` como
etiqueta: para excluir ruido hay que filtrar explícitamente filas, columnas y etiquetas. Sigue
necesitando entre 2 y N-1 etiquetas diferentes.

### Árboles H0 de superniveles

```python
import numpy as np
from lambdaforge.topology import SuperlevelMergeTree

tree = SuperlevelMergeTree().fit(
    values=np.array([5., 2., 4., 1., 3.]),
    edges=np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
    vertex_weights=np.array([1., 2., 3., 4., 5.]),
)
tree.persistence_diagram()  # ((5., None), (4., 2.), (3., 1.))
tree.components_at(2.)     # componentes conexas de values >= 2
tree.region_at(1, 3.)      # rama 1 antes de morir/fusionarse
tree.mass_curve(0)         # nivel descendente y masa después de cada evento
```

El contrato de aristas es `[E,2]` entero y no dirigido; para cero aristas se usa
`np.empty((0,2), dtype=int)`. Duplicados y bucles no alteran la topología. Valores finitos
reales y pesos finitos no negativos con masa total positiva. Se rechazan rango del campo o
masa total que excedan float64, sin reescalado silencioso.

Union-find procesa las mesetas completas. En una fusión sobrevive la rama de mayor nacimiento;
en empate, la de menor identificador de vértice de su meseta de nacimiento. Reordenar vértices
puede renombrar ramas empatadas, pero no las particiones ni el multiconjunto de persistencia.
Una transformación estrictamente creciente conserva la topología, no la persistencia numérica.
Cada componente desconectada conserva una rama esencial con `death=None`, `persistence=None`
y `essential=True`, no una persistencia infinita o cero inventada. Una rama existe en
`death < threshold <= birth`: consultar su muerte o por debajo falla. La rama superviviente
continúa con la región combinada tras absorber otra rama.

`branches` contiene mesetas de origen, padres, hijos y evento de fusión; `events` conserva la
jerarquía de crecimiento/fusión. Cada vértice se almacena una vez en eventos y otra en un índice
contiguo de subárboles. Construcción aproximadamente O(N log N + E alpha(N)), con ordenación
determinista adicional, y memoria O(N+E). Consultar regiones no reconstruye el grafo. La prominencia
topológica **no** es confianza estadística ni la jerarquía de densidad de HDBSCAN.

### Regiones estables y pertenencia por umbrales

```python
from lambdaforge.topology import StableRegionSelector, uniform_membership

selection = StableRegionSelector(
    min_persistence=1., min_mass=1., min_width=1., min_samples=2,
    min_iou=.8, overlap="disjoint",
).select(tree)
for region in selection.regions:
    print(region.branch_id, region.interval, region.mass, region.membership)
print(selection.rejected)
phi = uniform_membership(tree, branch_id=1, lower=2., upper=4.)
```

Se busca el intervalo **más ancho** que cumpla los límites, con extremos en eventos de la rama
y el suelo observado/muerte; no el intervalo diminuto que maximiza IoU. En empate de anchura
se prefiere mayor estabilidad y luego mayor extremo superior. El núcleo en el extremo alto debe
cumplir `min_mass`. La estabilidad es masa del núcleo/masa del soporte, la IoU ponderada entre
regiones anidadas. El extremo bajo es abierto: el soporte corresponde al límite por encima,
no a la región fusionada. `min_samples` cuenta extremos de umbrales, **no observaciones
independientes**. Las ramas esenciales usan el suelo/rango observado sin asignarles persistencia.

En `(lower,upper]`, `membership[i]` integra exactamente y normaliza la pertenencia del vértice
para umbrales uniformes. El momento en que un subárbol se incorpora importa; no basta su
intensidad original. La muerte puede ser un extremo de medida cero, nunca atravesarse.
`discrete_membership(tree, branch_id, thresholds, weights)` permite una distribución discreta
de umbrales normalizada explícitamente; todos sus niveles deben estar en la vida de la rama.
Son pertenencias espaciales, **no probabilidades de un evento científico**.

`overlap="allow"` conserva soportes posiblemente anidados. Por defecto `disjoint` acepta
primero los intervalos más anchos y rechaza soportes posteriores solapados; es una política
greedy explícita, no un óptimo global. Nunca recorta ni renormaliza máscaras. El límite
`max_membership_bytes` (256 MiB) acota los vectores N de pertenencia retenidos, no todo el proceso.

`weighted_iou(a,b,weights=w)` calcula `sum(w*min(a,b))/sum(w*max(a,b))`;
`weighted_dice` calcula `2*sum(w*min(a,b))/sum(w*(a+b))`. Con indicadores son medidas de
conjuntos; con fracciones, solapamiento espacial difuso. Unión/masa vacía usa `empty_value`
explícito, por defecto 1, no evidencia científica inferida. Las entradas deben pertenecer a
`[0,1]` y estar alineadas por vértice.

### Controles independientes y remuestreo

```python
from lambdaforge.stats import EmpiricalNullCalibrator, bootstrap_quantile

null = EmpiricalNullCalibrator(tail="upper", alpha=.05).fit(reference_scores)
result = null.evaluate(new_score)
null.fit_groups(candidate_scores, object_ids, reducer="max")
result = null.evaluate_group(new_object_scores)
interval = bootstrap_quantile(reference_object_maxima, .95, seed=42, n_resamples=1000)
```

Cola superior: `p=(1+count(reference >= score))/(n+1)`; cola inferior: `<=`. Los empates son
inclusivos y se reportan. La decisión es `p <= alpha`. La resolución/mínimo p es `1/(n+1)`;
si alpha impide rechazar se devuelve una advertencia. `reference_level` es un cuantil empírico
descriptivo, **no otro umbral de decisión**. Se conserva cola, alpha, empates, unidad y tamaños
de grupos. `fit_groups` permite `max` o `min`; `evaluate_group` aplica el mismo estadístico.

La definición del score, generación y selección de candidatos deben fijarse antes de calibrar.
Se requieren controles negativos independientes e intercambiables por objeto. Calibrar máximos
controla alguna falsa detección en un objeto negativo bajo esas hipótesis, no la proporción de
regiones falsas en objetos positivos, probabilidad posterior, causalidad o garantías condicionales
bajo cambios arbitrarios de distribución. No se reutiliza `ConformalPredictionInterval` de regresión.

`bootstrap_statistic(values, statistic, group_ids=..., strata=..., seed=...)` remuestrea grupos
independientes enteros con reemplazo, conservando filas y multiplicidad. Un estadístico sobre
filas agrupadas depende de sus tamaños: para peso idéntico por objeto hay que reducir previamente
a un escalar por objeto. Cada grupo debe pertenecer a un estrato. La salida contiene estimación,
intervalo percentil aproximado, confianza, remuestreos, seed, unidad/número de unidades y
degeneración. No es una garantía exacta ni corrige múltiples comparaciones. `bootstrap_quantile`
comparte esa política y usa cuantiles lineales de NumPy. Pocos controles o controles degenerados
siguen siendo evidencia débil.

Ejemplos ejecutables: [grafos y controles](../examples/graph_regions_calibration.py),
[benchmark de escala](../examples/benchmark_merge_tree.py). Referencias:
[GUDHI](https://gudhi.inria.fr/doc/latest/group__persistent__cohomology.html),
[Phipson y Smyth](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf),
[HDBSCAN de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).
Las ampliaciones restantes de clustering/asignación de prototipos siguen en el registro de pendientes.

<a id="fgw-blocks"></a>

### Matrices FGW acotadas y reanudables

`FGWMatrixBuilder` reutiliza el solver FGW existente. No es otro runner ni otro cache. Acepta
colecciones cuadradas o rectangulares, enumera pares deterministas por filas y acota bloques
y threads numéricos explícitamente: por defecto 32 comparaciones y un thread.

```python
from lambdaforge.distances import FGWMatrixBuilder, FusedGromovWassersteinDistance

builder = FGWMatrixBuilder(FusedGromovWassersteinDistance(alpha=.5), block_size=32)
square = builder.build(objects, symmetric=True)
cross = builder.build(new_objects, prototypes)
for block in builder.blocks(objects, symmetric=True):
    print(block.pairs, block.values, block.complete, block.diagnostics)
```

`symmetric=True` declara simetría de la métrica de atributos: resuelve el triángulo superior,
incluidas comparaciones propias, y refleja resultados. No afirma que dos orientaciones del
solver local den exactamente lo mismo. Dos colecciones explícitas son rectangulares y no pueden
pedir reflexión aunque tengan igual longitud. Con una colección sin simetría explícita se resuelven
ambas orientaciones. La firma/comportamiento anterior de `distance.pairwise(...)` se conserva.

`blocks()` no asigna la matriz completa. `build()` devuelve `FGWMatrixResult` con `values` de
solo lectura, `identity`, `symmetric` y diagnósticos por bloque: rango/número de pares, cantidad
convergida, iteraciones totales, máximo error marginal y fallos con hasta ocho mensajes abreviados.
No retiene todos los acoplamientos densos; para uno concreto se usa `distance.solve`.

Por defecto `error_policy="raise"` propaga el primer error. `"record"` continúa con **NaN**, no una
distancia inventada, donde falló una comparación. `complete` significa valores finitos, **no**
convergencia de todos los solves ni óptimo global. Errores de memoria o dependencias siempre se
propagan. Hay que comprobar integridad antes de clustering/publicación/JSON estricto. La validación
de disimilitudes rechaza matrices parciales con NaN.
En checkpoints JSON el fallo se guarda como `null` y su error; NaN solo se reconstruye en memoria.

`max_memory_bytes` vale 512 MiB por defecto. La estimación incluye workspace denso, conversión de
atributos numéricos, buffers de bloque, metadatos de identidad, resúmenes acumulados y dos buffers
de salida en `build()`. No es un límite duro de RSS ni acota callbacks/encoders arbitrarios. Los
objetos de entrada ya están residentes. Para colecciones grandes conviene consumir `blocks()` en
outputs gestionados nativos en vez de aumentar sin más el presupuesto denso.

Dentro de un Work en ejecución se puede usar la misma API:

```python
result = builder.build(objects, symmetric=True, work=self)
```

Cada bloque usa `self.resume_map(..., workers=1, retries=0)` y checkpoints JSON nativos. No hay
rutas de máquina serializadas, pickle, locks paralelos ni otro sistema de ficheros cacheados. Los
bloques acabados sobreviven a interrupciones; se repite el bloque interrumpido. Se recalculan bloques
ausentes, mal formados, incompletos, con checksum inválido o identidad distinta; los fallidos
registrados no se reutilizan como éxitos.

La identidad incluye esquema versionado, estructuras/atributos/pesos exactos, masas originales,
modo relacional, orden de colecciones, simetría, parámetros/clase del solver, threads y versiones
de LambdaForge/POT/NumPy/SciPy. Tamaño de bloque y límite de memoria no son identidad científica.
Los atributos numéricos usan dtype/shape/bytes de endian canónico. Los opacos requieren
`feature_encoder(features) -> bytes`, determinista y exacto. Una métrica callable requiere
`metric_identity="mi-metrica/revision-implementacion"`; debe actualizarse si cambia su ciencia.
No se adivina equivalencia de closures ni se serializan objetos Python arbitrarios. El checksum
detecta corrupción, no una reescritura maliciosa acompañada de otro hash. Identidad/ownership y
aceptación de cambio de código del Work siguen comprobándose independientemente. Se reanudan
comparaciones, no las iteraciones internas de un solve POT interrumpido.

<a id="fgw-barycenters"></a>

### Baricentros FGW reales y estructuras relacionales

`FGWBarycenter` llama al solver opcional real `ot.gromov.fgw_barycenters`. No promedia nodos sin
correspondencias. Optimiza un **prototipo relacional sintético local**, con soporte elegido por
el investigador; un medoide sería una observación real. No garantiza una realización física ni
que el tamaño del soporte sea óptimo.

```python
from lambdaforge.distances import FGWBarycenter

result = FGWBarycenter(n_support=16, alpha=.5, seed=42, n_init=3).fit(
    objects, object_weights=None, support_weights=None,
    init_structure=None, init_features=None,
)
prototype = result.prototype
print(result.objective, result.converged, result.warnings)
```

Se requieren atributos numéricos finitos `[N,F]` en una dimensión común. Se admite
`0 < alpha < 1`, masas balanceadas de probabilidad, pérdida estructural cuadrática y coste
de atributos euclídeo cuadrado. Esta formulación no admite atributos opacos/métricas personalizadas
ni promete baricentros de alpha extremo. Los pesos de objetos se normalizan explícitamente con
suma positiva. Los de soporte deben ser estrictamente positivos: POT divide por ellos. Los objetos
pueden tener masas no uniformes. No se reescalan unidades; alpha mezcla componentes cuya escala
debe elegirse conscientemente. Para un prototipo de cluster se pasan solo sus objetos y pesos.

`StructuredMeasure` mantiene por defecto sus validaciones de simetría/no negatividad/diagonal cero.
`StructuredMeasure(..., relational=True)` permite explícitamente relaciones de diagonal no nula.
La salida POT se conserva en ese modo, **sin anular su diagonal**. No se afirma que sea una métrica;
puede compararse después con la distancia o el builder FGW existente. Geometría/masas complejas se
rechazan, sin descartar silenciosamente la parte imaginaria. La distancia ordinaria mantiene sus
atributos arbitrarios, no este algoritmo numérico de baricentros.

Controles: `n_support`, `alpha`, `max_iter=100`, `tol=1e-7`, `seed`, `n_init=1`, `threads=1`,
`max_memory_bytes=512 MiB`, `transport_max_iter=200`, `transport_tol=1e-9`. Los dos últimos
son para verificaciones de transporte **posteriores**; los solves internos POT también usan
`max_iter`. C inicial `[support,support]` y X inicial `[support,F]` se aplican a cada reinicio.
Si se dan ambas, los reinicios pueden ser idénticos: no se añade ruido no solicitado. Si falta C,
se usa la inicialización estructural POT con seed; se registra cada seed generada. Multistart es
secuencial y elige el menor objetivo ponderado posterior válido, no un óptimo global ni necesariamente
un reinicio convergido. Los inválidos conservan diagnóstico; si todos lo son se produce un error.

El resultado contiene `prototype`, pesos, seed elegida, reinicios, historial de cambios de parámetros,
iteraciones, avisos y comparaciones finales. `objective` se verifica posteriormente mediante
`sum(lambda[k] * FGW(final_prototype, object[k]))`. `comparisons` contiene nuevos acoplamientos
factibles en el prototipo **final**, no los acoplamientos previos al último update POT. El modo
de logging usado no da un historial ponderado del objetivo final: `backend_objective=None` lo indica.

En POT 0.9.6 y 0.9.7.post1, comprobados realmente, el criterio de parada acepta cambio pequeño
de atributos **o** estructura. El wrapper no lo presenta silenciosamente como convergencia completa:
`converged` exige **ambos** cambios bajo tolerancia y comparaciones finales convergidas. De lo
contrario devuelve no convergencia y un aviso aunque POT parase pronto. Tampoco prueba un óptimo
global ni un etiquetado único de nodos. La estimación de memoria incluye los acoplamientos por
iteración guardados por POT; combinaciones excesivas se rechazan antes de resolver.

Ejemplo ejecutable: [matrices ponderadas y baricentro](../examples/fgw_barycenter.py).
Referencia: [POT FGW barycenters](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.fgw_barycenters).

<a id="guide-scientific-authoring"></a>

## Authoring científico: contratos, muestras y ETL

LambdaForge proporciona operaciones científicas, evidencia tipada y el ciclo nativo de Work.
El cliente visual posee canvas, expresiones/selecciones/bucles y generación de Python normal.
No se introduce un intérprete de nodos, otro runner, un registro de datasets paralelo ni otro
optimizador HPO. Las clases generadas heredan de `Work` y ejecutan su único `run(...)` sin GUI.
Se reutilizan GraphSAGE, NNConv, PNA, GINE, GVP/GVPGNN, PaiNN, SchNet, DimeNet y el Transformer
estructural ya existentes en el árbol local; no se reimplementan versiones para el editor.

### Descubrir y seleccionar operaciones

```sh
lf catalog --json
lf catalog PaiNN --kind model --json
lf catalog --kind etl --json
```

```python
from lambdaforge.catalog import OperationCatalog, ComponentFactory, ContractError

catalog = OperationCatalog()
description = catalog.describe("PaiNN", kind="model")
catalog.validate("MLP", {"in_features": 2, "out_features": 1}, kind="model")
selected_class = catalog.resolve(description["id"])
```

`operation-catalog-v1` se distribuye como JSON del paquete. Descubrir lee metadatos, no importa
Torch/sklearn ni módulos de plugins, no abre arrays y no hace peticiones. Resolver importa la
implementación elegida y sus dependencias reales. Los IDs son `modulo:Clase` o
`plugin:TIPO:NOMBRE`; los alias cortos ignoran mayúsculas, guiones y guiones bajos. Si hay
ambigüedad se exige el ID exacto. `ComponentRegistry.aliases("activation"|"normalization")`
expone públicamente los alias registrados, incluidos los de ese proceso. ComponentRegistry
conserva la precedencia histórica de alias nativos; el catálogo exige namespace ante colisiones.
Un alias registrado solo en memoria no es una declaración portable: distribúyelo como plugin.

Los descriptores contienen proveedor/versión, descripción traducible y clave de traducción,
tipos/defaults, puertos de configuración fábrica, métodos/puertos, estado e instancias. No todos
los operadores tienen ecuaciones de forma totalmente legibles por máquina:
`shape_status="see-native-docstring"` y `contract_status="native-signature"` significan
pendiente de resolver, **no** compatibilidad demostrada. MLP y encoders de grafos principales
tienen puertos detallados: PaiNN/GVPGNN separan salida escalar invariante del estado vectorial
polar `[N,C,3]`, GVP devuelve una tupla, atención estructural conserva `[B,N,F]` y padding.
Topología/geometría/unidades las aporta el consumidor, sin inferir vecinos ni tipos de proteína.
Las simetrías solo se afirman donde están documentadas para el operador concreto.

`check_ports(origen, destino)` rechaza conflictos explícitos de tipo/dtype/unidades/rango y
dimensiones fijas; no resuelve todas las formas simbólicas. `validate` comprueba la firma y los
tipos primitivos declarados sin construir un modelo. Rangos específicos, restricciones cruzadas
y tensores reales siguen validándose en las implementaciones nativas. Una GUI no debe presentar
restricciones pendientes como demostradas ni construir modelos al descubrir el catálogo.
Schema distribuido: `schemas/operation-catalog-v1.json`.

Los plugins siguen PluginRegistry y sus entry points/procedencia. Un proveedor puede instalar
`lambdaforge.operations.json` junto a su `METADATA`, con el mismo schema y el ID exacto del
entry point. Este último determina implementación/proveedor/versión, no una redirección arbitraria
del JSON. Sin declaración se informa `contract_status="undeclared"`, sin inventar puertos.
Límite de metadatos: 1 MiB por proveedor. `ContractError.to_dict()` proporciona `code`, `message`,
`details`; los errores CLI de catálogo devuelven código 2 y JSON `operation-error-v1`.
Códigos: `UNKNOWN_OPERATION`, `AMBIGUOUS_ALIAS`, `CONTRACT_VERSION`, `INVALID_CONTRACT`,
`INVALID_PARAMETERS`, `MISSING_DEPENDENCY`, `ABSTRACT_OPERATION`, `INCOMPATIBLE_PORT`,
`INCOMPATIBLE_SHAPE`.

```python
from lambdaforge.nn import MLP

activation = ComponentFactory("prelu", "activation", {"num_parameters": 1})
normalization = ComponentFactory("layernorm", "normalization", {"eps": 1e-6})
model = MLP(2, 1, hidden=[4, 8],
            activation=activation.component_type(), activation_kwargs=activation.kwargs,
            normalization=normalization.component_type(), normalization_kwargs=normalization.kwargs)
a, b = normalization.build(features=4), normalization.build(features=8)
provenance = activation.provenance()
```

Cada construcción crea parámetros independientes; reutilizar una instancia comparte pesos y
buffers deliberadamente. Train/eval son explícitos. Las fábricas de normalización rechazan
anchos impuestos por el editor: los determina la capa receptora. MLP, ResidualMLP, CNN2D, GCN y
GraphSAGE admiten los pares clase/kwargs documentados; consulta `configuration_ports` del
constructor elegido. Arquitecturas de activación fija como PaiNN no adquieren esa inyección.
Las activaciones gated pueden cambiar el ancho; el generador debe tenerlo en cuenta.

### Muestras identificadas con varios assets

API: `from lambdaforge.data.samples import FieldSpec, SampleSchema, SampleDataset,
collate_samples, collate_graphs`. Es un lector perezoso del DatasetArtifact/índice nativo,
no otro registro. Recibe la raíz que resuelve el input Work `{dataset: NOMBRE@VERSION}`;
DatasetService gestiona ubicación/materialización. No descarga implícitamente ni registra nada.
[`examples/sample-schema.json`](../examples/sample-schema.json) contiene un ejemplo real;
`schemas/sample-schema-v1.json` se instala en el paquete. La misma declaración puede estar en
YAML o metadatos del dataset, elegida explícitamente por el consumidor, no por nombres inferidos.

Cada campo declara asset lógico, formato NPZ/NPY/JSON/CSV, array/campo/índice, dtype, forma sin
lote, ejes/unidades, rol y política de ausencia/collate. Puede leer targets/metadatos/particiones/id
sin abrir assets. Los roles separan entrada de modelo, objetivo/loss, metadatos, máscaras,
índices, coordenadas y vectores. `bindings` une argumento del modelo → campo; no permite inyectar
targets/loss/metadatos como entradas. Elegir campos es Python del consumidor y puede exponerse
como hiperparámetro del sweep/search nativo, sin otro motor de optimización.

```python
from lambdaforge.data.samples import SampleSchema, SampleDataset, collate_samples

schema = SampleSchema.from_mapping(declared_mapping)
train = SampleDataset(resolved_dataset_path, schema, partitions={"split": "train"})
report = train.preflight()                  # solo declaraciones/índice
small = train.preview(samples=2, max_bytes=1_048_576, max_seconds=2)
batch = collate_samples([train[0], train[1]], schema)
inputs = schema.model_inputs(batch["fields"])
```

La construcción verifica el checksum exacto del índice y recorre metadatos acotados: por defecto
100.000 miembros, índice de 64 MiB y declaración de 1 MiB. No abre arrays. Preflight comprueba
declaraciones de assets necesarios, no demuestra compatibilidad de sus valores. Al acceder
verifica SHA-256 del asset elegido, dtype y forma. Un campo opcional no declarado puede usar
`none` o default JSON; se conserva su máscara de presencia. Si estaba declarado y faltan sus
bytes, o está corrupto, no se oculta con un default. Se conservan ids, particiones e identidad
exacta de contenido y assets en procedencia.

El límite por muestra es de 32 MiB codificados/decodificados y 5 segundos cooperativos. Preview
comparte un presupuesto global entre 0–20 muestras. Son guardas conservadoras, no cuotas RSS ni
preempción de llamadas nativas bloqueadas. NPY solo headers 1/2; NPZ comprueba expansión antes
de decodificar. No acepta arrays object/pickle, symlinks, escapes de raíz ni URIs remotas sin
materialización nativa previa. CSV conserva cadenas salvo conversión explícita de la receta.
Inventarios mayores requieren otra estrategia de acceso/batching: no se truncan silenciosamente.

`collate_samples` permite `stack` con shape/dtype iguales, `ragged`/`list`, o padding de ceros
explícito únicamente en el primer eje con `masks["CAMPO:padding"]`, separado de presencia.
`collate_graphs(..., node_fields=(...), edge_fields=(...))` valida índices/cardinalidades,
desplaza aristas y devuelve `batch`, `ptr`, ids/procedencia. No aplana `[N,C,3]`, mezcla grafos
ni promociona tipos silenciosamente. El consumidor convierte NumPy a tensores explícitamente.

### ETL Python y estado aprendido solo en train

`lambdaforge.etl` exporta `ListFiles`, `ReadRecords`, `WriteRecords`, `SelectFields`,
`JoinRecords`, `PartitionRecords`, `WriteSampleAssets`, `ValidateRecords`, `TrainStandardizer`, `Download`.
Expresiones para derivar campos, condiciones y bucles siguen siendo Python ordinario del
generador. Los escritores rechazan sobrescritura/object arrays y usan rutas de staging
gestionadas explícitas, no son publicadores alternativos. `ValidateRecords` valida filas
decodificadas con las mismas reglas dtype/shape de SampleSchema, preservando valores y
particiones sin imputación implícita. `JoinRecords` ofrece left/inner
uno-a-uno, IDs únicos, índice derecho limitado, sin colisiones de columnas ni desacuerdos de
partición. Renombra antes; many-to-many no está soportado. `SelectFields` conserva id/particiones
y filtra por igualdad explícita. `PartitionRecords` usa hash de semilla + grupo/id, independiente
del orden, sin sobrescribir splits. Son proporciones esperadas, no estratificación exacta.

`TrainStandardizer.fit(values, ids=..., splits=["train", ...], fold="cv-1")` se ajusta una sola
vez sobre filas train identificadas, finitas y 2D. `transform(..., fold="cv-1")` no reajusta ni
acepta otro fold; `state_dict/from_state_dict` usan JSON portable. Crea estado independiente por
fold/trial. No ajusta valid/test ni demuestra que las etiquetas train aportadas sean honestas.
Otros imputers/tokenizers deben respetar esa separación; no se afirma un motor genérico de
tokenización. [Guía de fuga de datos](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

`Download(timeout=30, retries=3, max_bytes=268435456, credential=None)` delega en
`work.cache.fetch` con SHA-256 obligatorio, límite decodificado, caché atómica y cancelación entre
lecturas/reintentos. El timeout limita sockets bloqueados, no hay preempción inmediata. Rechaza
URLs con credenciales/query. Las referencias bearer `env:NOMBRE`/`keyring:IDENTIFICADOR` usan
CredentialService, HTTPS y redirecciones del mismo origen exacto. Configura las referencias en
el host de ejecución: no se reenvían secretos del portátil implícitamente, ni se guardan en
registros de caché o motivos HTTP autenticados. Descubrir no descarga. La API de caché también
acepta explícitamente `max_bytes`, `cancelled` y `credential`.

Para reanudar usa `Work.resume_map` y checkpoints; para publicar, `outputs.dataset(...)` con
schema/targets/assets/lineage y registro nativo. No publiques tras cancelar. El ejemplo almacena
dependencias lógicas ManagedFile, no rutas de máquina, y publica desde `source_checkpoint`.
Un fallo conserva checkpoints sin registrar datasets parciales; el reintento nativo reutiliza
los elementos válidos. No se debilita identidad ni recuperación del framework.

### Estimadores clásicos, no módulos diferenciables

Extra opcional `lambdaforge[estimators]`. `SklearnEstimator(algorithm, parameters=None, seed=0,
threads=1)` carga sklearn al ajustar: árboles, random forest, gradient boosting convencional de
clasificación/regresión y SVC/SVR. `capabilities` declara fit/predict/probabilidades, sin transform,
CPU y no diferenciable. SVC exige `probability=True` para probabilidades. El Work aporta métricas,
objetivos y HPO nativos; no hay otro trainer. Semillas/hilos son explícitos. GradTree/NODE/GRANDE
siguen siendo modelos distintos.

Persistencia opcional `lambdaforge[estimator-persistence]`: `save`, `inspect_file` y
`load(..., trusted_types=tipos_revisados)` usan skops, nunca pickle. Revisa los tipos desconocidos
exactos antes de aprobarlos; no confíes en listas de un atacante. Usa ficheros gestionados de
checkpoint/output. No instala extras al importar. Compatibilidad entre versiones del backend y
confianza siguen siendo responsabilidades explícitas. [Persistencia sklearn](https://scikit-learn.org/stable/model_persistence.html).

### Ejemplo CPU y distribución al cliente visual

Copia `visual_research.py`, `visual_prepare.yaml`, `visual_train.yaml`, `visual_search.yaml` de
los ejemplos instalados a un proyecto Python desechable/importable. Ajusta el módulo de `run`
solo si cambia el nombre del fichero. Desde ese proyecto:

```sh
python visual_research.py raw          # CSV + JSON sintéticos, no sobrescribe directorio
lf run visual_prepare.yaml             # Work ETL + visual-demo@1 atómico
lf run visual_train.yaml               # sweep exacto de campo/activación
lf run visual_search.yaml              # mismas operaciones, planner HPO nativo
```

Combina dos entradas nombradas: features NPZ y sidecar JSON; un campo opcional conserva
default/presencia y los targets no entran al modelo. Publica scaler/preview como artifacts.
`use_context` y activación se varían con el diseño nativo. No contacta clusters ni modifica WISDOM.

Construcción desde el **árbol de trabajo real**, incluidos ficheros nuevos, no solo HEAD:

```sh
python tools/build_operation_catalog.py
python tools/build_scientific_wheel.py /ruta/absoluta/salida-nueva
```

El builder usa setuptools/wheel ya instalados y offline, comprueba catálogo actualizado, fija
SOURCE_DATE_EPOCH/PYTHONHASHSEED y guarda `build-receipt.json` con versión, HEAD, dirty flag,
SHA-256 del wheel/contenido empaquetado y hashes por miembro. Usa una salida vacía; dos builds
sin cambios con el mismo epoch deben coincidir. No instala dependencias ni produce un payload
editable. Sigue siendo 0.19.0 sin una petición de release: versión, HEAD y contenido no son lo
mismo. Instala el wheel revisado en el runtime/bundle del cliente, fija su SHA exacto, valida
catálogo/recorrido CPU allí y reconstruye el payload con el empaquetado del cliente. El payload
0.17.1 antiguo no obtiene estas APIs al abrir el checkout. No se modifica SimpleAIStudio ni se
afirma integración de su interfaz. Versiones de schema desconocidas fallan sin migraciones
silenciosas ni puertos inventados.

<a id="guide-tabular-connectors"></a>

## Tablas incrementales, SQL y efectos de preview

Ampliación aditiva de `operation-catalog-v1`. Los defaults CSV existentes siguen siendo coma,
una fila de cabecera, UTF-8 y valores string. El cliente genera Python/Work corriente; no hay
intérprete de nodos. `lf catalog --kind etl --json` lee metadatos sin importar backends SQL,
abrir fuentes, descargar datasets ni entrenar.

### Ficheros delimitados y columnas tipadas

```python
from lambdaforge.etl import ReadRecords, WriteRecords, PreviewBudget, PreviewSession
from lambdaforge.etl.tables import Column, TableSchema

schema = TableSchema([
    Column("id"),
    Column("amount", "number", nullable=True, null_values=("", "NA")),
    Column("measured_at", "datetime", date_format="%Y-%m-%d", timezone="UTC"),
], keys=["id"])
reader = ReadRecords("csv", csv={"delimiter": ";", "decimal": ",", "bom": "strip"},
                     schema=schema.to_dict(), columns=["id", "amount"], max_rows=50)
for row in reader.iter_records(source):
    consume(row)                         # cerrar generadores abandonados explícitamente
preview = reader.preview(source, budget=PreviewBudget(max_rows=10, max_bytes=64_000))
```

`csv` configura `CsvOptions`: delimiter, header_rows, columns, encoding, bom
(`preserve|strip|error`), quotechar/escapechar (null desactiva quoting/escape), doublequote,
comment, skip_rows, decimal y parse_errors (`error|skip`). TSV usa tabulador por defecto.
Las cabeceras son registros: **la última** fila da los nombres, o se imponen nombres explícitos.
Sin cabecera es obligatorio declarar nombres. skip_rows cuenta líneas físicas iniciales; los
comentarios se saltan solo entre registros, nunca dentro de campos multilínea entrecomillados.
Registros físicos vacíos se ignoran, como DictReader. Cabeceras vacías/duplicadas, anchos
incorrectos y encoding inválido fallan. Preview informa de registros malformados omitidos con
diagnósticos acotados. La conversión de columnas tiene su política independiente.

La proyección `columns` y `max_rows` se aplican durante la lectura, no tras cargar el fichero.
`__call__` mantiene el resultado materializado y selección interna `path` histórica, con guarda
de memoria. `iter_records`/`preview` requieren CSV/TSV sin path interno. JSON completo no se
anuncia falsamente incremental. `compression="gzip"` comprueba bytes/registros **decodificados**;
no se adivina compresión por extensión. NPZ tiene su lector de arrays. Se mantiene el límite
nativo de tamaño de campo de Python CSV, sin cambiar esa opción global detrás de otros lectores.

`WriteRecords("csv"|"tsv", csv=...)` comparte delimitador/nombres/encoding/quote/escape/decimal,
cabecera de 0 o 1 filas y `write_bom=True` explícito si se desea BOM de salida. Rechaza opciones
exclusivas de lectura (saltos, comentarios, múltiples cabeceras, omitir errores). Escribe filas
incrementalmente, no sobrescribe ni publica: usa rutas de staging/checkpoint/output gestionadas.
La escritura NPY/NPZ/JSON anterior conserva sus defaults.

`table-schema-v1` es un contrato portable de columnas escalares. Ejemplo real:
[`examples/table-schema.json`](../examples/table-schema.json); schema instalado:
`schemas/table-schema-v1.json`. Tipos string/integer/number/boolean/datetime/json, nullable,
null_values, invalid (`error|null`), categories, date_format y timezone. Nada se infiere de un
preview. El vacío solo es null si se declara; no se truncan fracciones a enteros. Invalid-to-null
requiere nullable. Fechas usan formato/ISO explícito y zona para valores sin zona; rechaza horas
locales ambiguas/inexistentes. No aplana arrays/geometría como escalares. `TableSchema.convert`
conserva extras; `keys` declara columnas no nulas, y FromRows/escritores SQL verifican unicidad.

Operaciones Python ordinarias:

| Operación | Contrato |
| --- | --- |
| `CastTable(schema)` | Conversiones declaradas; id/particiones/procedencia no se reescriben. |
| `FillNulls(values)` | Relleno constante explícito, **no** imputer aprendido; campos protegidos intactos. |
| `FilterTable(predicate, keep=True)` | eq/ne/lt/le/gt/ge/in/is_null/not_null escalares, all/any/not acotados; sin eval. Comparaciones con null son falsas salvo operadores null; not es negación booleana normal. |
| `SelectTable(columns, drop=(), rename=None)` | Proyecta/elimina/renombra, conservando campos protegidos; rechaza colisiones. |
| `SortTable(by, descending=False, nulls="last")` | Orden estable y materialización limitada, no ordenación externa en streaming. |
| `DeduplicateTable(keys, keep="first")` | Conserva primero/último explícitamente; no fusiona ids/particiones/procedencias diferentes. |
| `AggregateTable(by, aggregations, identity="derive")` | Nuevos IDs deterministas y lineage completo/acotado de miembros; particiones forman parte del grupo. Count/sum/mean/min/max excluyen null, no los convierten en cero. |
| `ConcatTables()` | Mismo conjunto de columnas; rechaza ids duplicados, no une schemas implícitamente. |
| `FromRows(schema)` | Filas portables declaradas, claves/conversiones explícitas, materialización limitada. |

El argumento de agregación `identity="derive"` es obligatorio, no una reescritura implícita de ids.
Predicado: `{"all": [{"column": "amount", "op": "not_null"},
{"column": "amount", "op": "gt", "value": 0}]}`. Agregación:
`{"mean_amount": {"column": "amount", "op": "mean"}}`. Las operaciones materializadas
tienen límites de 10.000 filas y 8 MiB por defecto; excederlos falla, no muestrea silenciosamente.
Las transformaciones streaming consumen el iterador aportado: el Python generado debe usar
fuentes acotadas/comprobar cancelación. `TrainStandardizer` conserva aislamiento train/fold.
Los calibradores usan su contrato de **calibración held-out**, no entrenamiento o test.

### Presupuesto compartido de preview y staging temporal

```python
from lambdaforge.etl import PreviewArray

budget = PreviewBudget(max_rows=20, max_bytes=1_048_576, max_seconds=5,
                       max_memory=8_388_608, max_files=4, cancelled=stop_requested)
session = PreviewSession([approved_input_root], budget)
table = session.records(reader, source)
array = session.array(PreviewArray("npz", key="features"), array_source)
with session.staging_directory() as scratch:
    inspect_temporary(scratch)           # únicamente salida temporal, nunca publicación
```

Un presupuesto cubre filas, bytes realmente leídos/decodificados, fuentes, tiempo y retención
de previews entre llamadas. Rechaza symlinks/traversal/rutas fuera de raíces aprobadas.
`table-preview-v1` y `array-preview-v1` informan **prefijo**, complete (false conservador al
alcanzar el límite de filas), procedencia/ordinales/rangos de líneas, contadores efectivos y
`content_identity=null`. Un hash del prefijo no es identidad del fichero/dataset. No inventa
estimaciones estadísticas.

NPY usa mmap de solo lectura para filas del primer eje; NPZ descomprime solo el prefijo de un
array seleccionado C-order, acotando datos decodificados y no fiándose del tamaño comprimido.
Informa dtype/shape y tamaño decodificado total sin reservarlo. Rechaza object/records estructurados
y NPZ Fortran multieje. NPY Fortran sí usa mmap. No verifica los bytes que no leyó: verificación/
materialización de datasets inmutables sigue en DatasetService, no un atajo de GUI. Abrir un
descriptor nunca materializa datasets remotos completos.

La lectura parcial del lector es distinta del sellado de inputs nativos de Work: un input
tipado `{file: ...}` sigue verificando su huella exacta antes de ejecutar y puede leer todos
sus bytes. Preview no exime de esa identidad. El cliente debe acotar su acceso de preview
separado, no afirmar que lanzar el Work entero solo lee un prefijo.

Límites cooperativos, **no sandbox RSS del SO**. Se guardan tamaños de objetos retornados y
chunks decodificados; parsers/drivers asignan algunos objetos antes de comprobarlos. I/O de disco,
DNS, DB y llamadas nativas necesitan sus timeouts. PostgreSQL puede entregar un valor grande
antes de la guarda Python; SQLite en Python 3.10 carece de setlimit y en 3.11+ se limita además
el tamaño del valor antes de recuperarlo. Proyección/chunks pequeños y Works con límites de
recursos siguen siendo necesarios para entradas adversarias. Cierra iteradores parciales: salida
de contextos cierra ficheros/cursors y revierte lecturas. El staging limpia solo su propio árbol
incluso tras error, no inputs. No es aislamiento de plugins ni un sistema de permisos de código.

### SQL, fuentes mutables y escrituras deliberadas

SQLite usa stdlib; PostgreSQL es opcional/perezoso `lambdaforge[sql-postgresql]` (Psycopg 3).
Nada se instala ni conecta durante construcción/descubrimiento.

```python
from lambdaforge.etl import ReadSQL

sqlite = ReadSQL("sqlite", file="source.sqlite", table="measurements",
                 columns=["id", "amount"], order_by=["id"], chunk_size=100)
rows = sqlite.iter_records(budget=PreviewBudget(max_rows=1000))
postgres = ReadSQL("postgresql", credential="env:PROJECT_DB", table="measurements",
                   columns=["id", "amount"], order_by=["id"], sslmode="verify-full")
remote_preview = PreviewSession([approved_input_root], allow_network=True).sql(postgres)
```

La referencia de credencial se resuelve con CredentialService en el host ejecutor a JSON con
host/port/dbname/user/password/sslrootcert según corresponda. Ese JSON secreto no va en YAML,
contratos ni logs. Rechaza credenciales sin referencia y TLS desactivado. Timeouts de conexión/
consulta explícitos. TLS por defecto verify-full; require/verify-ca son políticas más débiles
explícitas, no fallback silencioso. Errores de driver se redactan sin encadenar DSN/password.
Usa un rol DB limitado al alcance de lectura/escritura que necesita el proyecto.

Elige **query O table**. Valida y entrecomilla identificadores simples de tabla/columna/orden;
schemas cualificados/expresiones complejas requieren SQL escrito explícitamente, no interpolación
de identificadores. Parámetros separados: SQLite `?`/`:name`, PostgreSQL `%s`/`%(name)s`.
Query controla proyección/orden; aliases de salida únicos. SQLite abre la DB existente mode=ro y
su authorizer rechaza DDL/mutaciones/carga de extensiones. PostgreSQL usa transacción read-only
repeatable-read y cursor de servidor. Ambos recuperan chunks limitados. SQLite comprueba tiempo/
cancelación en su progress handler; PostgreSQL usa statement_timeout y comprueba cancelación entre
llamadas. No es sandbox de SQL arbitrario: funciones del servidor pueden tener efectos externos,
así que se requiere confianza en rol DB/autores de query.

Cada iterador ve una transacción, no un snapshot reutilizable entre Runs. Preview conserva backend,
aislamiento, huella de consulta y hash del **prefijo observado** por separado. La misma consulta
con datos distintos no es el mismo input científico. Para fijarlo, materializa las filas exactas
en staging/checkpoint gestionado, conserva receta/dependencias y publica con `Work.outputs.dataset`;
entrena con ese DatasetInput inmutable. No uses una huella de consulta como identidad de dataset.

`WriteSQL(dialect, table, schema, ..., mode="insert"|"upsert"|"replace", create=False)` solo
escribe tablas declaradas. SQLite exige fichero existente, no crea una DB implícitamente.
TableSchema keys es obligatorio; rechaza claves de entrada duplicadas. Upsert usa la restricción
única real de DB y es idempotente sobre esas claves. Replace elimina **filas**, no tablas.
Crear tabla requiere autorización DDL separada. Filas/DDL son una transacción: ante fallo de
conversión/schema/clave/presupuesto/cancelación se revierte todo. No reintenta indefinidamente.
Columnas JSON necesitan adaptador backend explícito futuro, no se aplanan silenciosamente.

```python
from lambdaforge.etl import WriteSQL, SQLWritePolicy

# Solo código operador/sesión fiable tras revisión humana, no un booleano del nodo.
authorization = SQLWritePolicy(allow_insert=True, allow_update=True)
writer = WriteSQL("sqlite", "measurements", schema.to_dict(), file="target.sqlite", mode="upsert")
receipt = writer(validated_rows, policy=authorization)
```

SQLWritePolicy es un límite de confianza de API, no sandbox contra Python sin restricciones.
El cliente debe conservar autorización del lado del operador. Preview rechaza escritores y no
publica ni borra/reemplaza fuentes/DDL. Tests con PostgreSQL efímero TLS real son opt-in mediante
`LAMBDAFORGE_TEST_EPHEMERAL_PG_REF`; sin referencia se omiten sin contactar servidores. SQLite
incluye fixtures locales de snapshot/rollback/transacciones. Otros dialectos no están soportados:
requieren adaptador DB-API maduro y pruebas de read-only/snapshot/placeholders/timeouts/
identificadores/transacciones/evidencia, no renombrar un comando. Referencias:
[SQLite Python](https://docs.python.org/3/library/sqlite3.html),
[Psycopg transacciones](https://www.psycopg.org/psycopg3/docs/basic/transactions.html).

### Efectos, modelos estructurados y configuración de proveedores

Descriptores incluyen `operation-effects-v1`: read-data/read-metadata, network (condicional en
SQL), learn-state, instance-state, write-cache, write-staging, modify-external-source y DDL
condicional. preview.supported identifica un método nativo limitado, no permiso global. Plugins
tienen preview desactivado por defecto incluso con metadatos; PreviewSession no invoca plugins/
subclases arbitrarios. ContractError.to_dict añade translation_key estable junto a code/message/
details. Se traducen códigos, sin parsear textos de excepción.

`OperationCatalog.validate_descriptor(ID)` importa solo la clase seleccionada y contrasta nombres,
defaults/requeridos, puertos de métodos y estado abstracto/concreto. No construye modelos ni prueba
ecuaciones tensoriales generales. `audit(kind="model")` es metadata-only y separa abstract,
declared-structure y signature-only. Inventario inspeccionado: 93 modelos, 1 abstracto, 15 con
estructura de salida declarada y 77 solo firma. Estructura declarada **no** demuestra toda
componibilidad. Se reutilizan modelos existentes: ensembles retienen módulos/comparten pesos,
multitarea devuelve tensores nombrados, mixtures exponen expertos y top-k sigue evaluándolos todos,
NeuralODE declara dynamics(t,y), forma trayectoria/estado e integración fija, y calibración declara
estado mutable held-out y salidas estructuradas.

`ConfigurationSchemas.schema("work"|"cluster")` y `lf config schema --kind cluster --json`
exponen contratos instalados. El default sigue siendo Work. `cluster-profile-v1.json` incluye
políticas nativas, opciones/condiciones/defaults, unidades y secretos por referencia;
`resolve_cluster(name, mapping)` comprueba schema y llama al **mismo normalizador ClusterProfile**
para defaults efectivos sin conectar/reservar. Restricciones cruzadas/propiedad siguen en la
validación nativa. Work schema es guía estática: firmas Work.run requieren la validación Python
autorizada ya existente. No hay segundo normalizador YAML.

Fixtures sintéticos cubren `command_prefix: [gpu]` → `gpu ejecutable args`, `[gpu, exec]` y otros
argv explícitos con pruebas de visibilidad declaradas. Un nombre de comando no prueba protocolo
ni que una GPU física libre sea una concesión propia. Servicios procesos/SLURM existentes poseen
submit/recibo/identidad, inventario/logs/cancelación y capacidades declaradas pause/resume. Claim/
release argv sueltos siguen rechazados sin prueba de reserva por Job; preserva reservas manuales/
hermanas. **No se añade scheduler arbitrario** ni parser SLURM reutilizado de manera falsa.
Un proveedor nuevo requeriría todo el contrato Scheduler/propiedad/recuperación y sus pruebas.

Migración aditiva: defaults CSV e identidades científicas previas intactos, nuevos contratos
tablas/efectos/preview versionados. Ecuaciones tensoriales completas, Parquet/Arrow y otros SQL
quedan pendientes, no como endpoints anunciados. Ejemplo CPU:
[Work de preview](../examples/tabular_preview.py), [YAML](../examples/tabular_preview.yaml),
junto al recorrido ETL→dataset→modelo→sweep/HPO existente. Para construir el wheel del árbol real,
regenera `tools/build_operation_catalog.py` y `tools/build_cluster_schema.py`: se comprueba que
ambos contratos estén actualizados. El recibo fija bytes/HEAD/dirty del paquete sin afirmar que
se haya actualizado el bundle de la aplicación cliente.
