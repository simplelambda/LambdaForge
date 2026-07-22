# Datos y caché de datasets de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

El paquete <code>lambdaforge.data</code> proporciona adaptadores de datasets
PyTorch indexables y agnósticos de la tarea, además de una caché acotada
opcional. La caché nunca es implícita: el proyecto elige qué etapa determinista
envolver, cuántos datos serializados puede retener cada proceso y si resulta
apropiado usar un backend persistente.

## Índice

- [Mapa de objetos](#mapa-de-objetos)
- [DatasetCache](#datasetcache)
- [Semántica de la cuota RAM](#semántica-de-la-cuota-ram)
- [Backends persistentes](#backends-persistentes)
- [Fingerprints, integridad y serialización segura](#fingerprints-integridad-y-serialización-segura)
- [Coordinación multiproceso y recuperación](#coordinación-multiproceso-y-recuperación)
- [Workers de DataLoader y planificación de capacidad](#workers-de-dataloader-y-planificación-de-capacidad)
- [FileDataset](#filedataset)
- [NumpyMemmapDataset](#numpymemmapdataset)
- [Plugins de dataset](#plugins-de-dataset)
- [Ejemplo YAML completo](#ejemplo-yaml-completo)
- [Estadísticas](#estadísticas)
- [Invalidación y ciclo de vida](#invalidación-y-ciclo-de-vida)
- [Determinismo y claves de caché](#determinismo-y-claves-de-caché)
- [Seguridad y contratos de extensión](#seguridad-y-contratos-de-extensión)

## Mapa de objetos

| Objeto | Responsabilidad |
|---|---|
| <code>DatasetCache</code> | Envuelve un dataset indexable con una LRU RAM local al proceso y un backend de bytes opcional. |
| <code>CacheStats</code> | Snapshot inmutable de estadísticas locales al proceso y uso actual del backend. |
| <code>CacheUsage</code> | Snapshot atómico de entradas/bytes de un namespace coordinado. |
| <code>DatasetFingerprint</code> | Identidad canónica de contenido, transformación determinista y configuración. |
| <code>DiskCacheBackend</code> | Guarda registros verificados bajo una cuota multiproceso atómica. |
| <code>MemoryMappedCacheBackend</code> | Mantiene un lease compartido mientras lee un registro mmap verificado. |
| <code>CacheRecordCodec</code> | Envelope versionado con checksum SHA-256 o HMAC-SHA256. |
| <code>CacheIntegrityMode</code> | Elección cerrada de checksum/autenticación sin strings mágicos en Python. |
| <code>CacheNamespaceManifest</code> | Contrato inmutable de cuota y formato compartidos. |
| <code>DatasetSerializer</code> | Contrato abstracto de conversión entre claves/muestras y bytes. |
| <code>NumpyDatasetSerializer</code> | Codec determinista/acotado para árboles NumPy/Torch sin pickle. |
| <code>PickleDatasetSerializer</code> | Compatibilidad solo para datos locales explícitamente confiables. |
| <code>CacheBackend</code> | Contrato abstracto de almacenamiento de bytes para backends de proyecto. |
| <code>CacheRecord</code> | Posee un payload del backend y su callback de cierre idempotente. |
| <code>FileDataset</code> | Carga de forma lazy una lista ordenada y explícita de archivos con un callable del proyecto. |
| <code>NumpyMemmapDataset</code> | Lee arrays <code>.npy</code> alineados sin cargar los arrays completos en RAM. |

Todos estos nombres son públicos desde <code>lambdaforge.data</code>. El
paquete <code>lambdaforge.training.data</code> tiene otra responsabilidad:
contiene el data module de Lightning y el adaptador protegido de workers de
DataLoader.

## DatasetCache

El constructor es:

~~~python
DatasetCache(
    dataset,
    max_memory_bytes_per_process,
    max_memory_entries=10_000,
    backend=None,
    serializer=None,
    key_fn=None,
    cache_in_workers=False,
    strict=False,
    fingerprint=None,
)
~~~

Un acceso sigue este orden:

1. Calcula el hash del resultado versionado de <code>key_fn(index)</code>, o
   del propio índice.
2. Busca en la LRU RAM local al proceso cuando la caché RAM está habilitada.
3. Busca en el backend opcional.
4. Carga el dataset envuelto tras un fallo, serializa el resultado y lo admite
   en cada capa configurada cuya cuota lo permita.

Los aciertos RAM y de backend se deserializan como objetos nuevos. Esto aísla
listas, mappings, arrays NumPy y tensores CPU mutables de llamadas posteriores.
En el primer fallo, el resultado se serializa antes de devolver el objeto
original del dataset, por lo que una mutación del llamador no cambia los bytes
almacenados.

Se mantiene <code>PickleDatasetSerializer</code> como predeterminado por
compatibilidad. Las cachés persistentes de investigación deberían seleccionar
<code>NumpyDatasetSerializer</code> explícitamente cuando su árbol de muestras
sea compatible. El serializador debe codificar la tupla versionada de clave y
cada muestra. Los tensores CUDA anidados no se cachean: el dataset debe
devolver CPU y delegar la memoria del acelerador a la capa de transferencia.

Con <code>strict=False</code>, los errores de serialización o backend se
contabilizan y el dataset envuelto continúa siendo utilizable. Con
<code>strict=True</code>, se convierten en excepciones
<code>RuntimeError</code> descriptivas. Los parámetros
<code>cache_in_workers</code> y <code>strict</code> requieren valores booleanos
reales.

## Semántica de la cuota RAM

Dos límites independientes acotan la LRU local al proceso:

- <code>max_memory_bytes_per_process</code> es la suma de las longitudes de
  payloads serializados e inmutables retenidos por una réplica de proceso. El
  valor cero desactiva la capa RAM sin desactivar un posible backend.
- <code>max_memory_entries</code> impide que millones de registros pequeños
  agoten la memoria mediante overhead de mappings y claves. Debe ser positivo.

Los registros menos usados recientemente se expulsan hasta satisfacer ambas
restricciones. Leer un registro actualiza su posición. Un payload mayor que
todo el presupuesto de bytes se omite sin vaciar registros pequeños útiles.

El límite de bytes es exacto para los payloads serializados retenidos, pero
**no es un límite duro de RSS**. Excluye:

- strings SHA-256, mapping ordenado, locks y overhead del allocator de Python;
- el dataset envuelto y cualquier dato que ya posea;
- buffers temporales de serialización/deserialización;
- la muestra viva devuelta al llamador;
- colas de prefetch de DataLoader, lotes combinados y memoria fijada;
- estado del modelo, optimizador y métricas;
- cachés de páginas del sistema de archivos y memory maps.

Elige un presupuesto con margen. Un proceso puede no devolver inmediatamente
al sistema operativo las páginas liberadas por el allocator tras una expulsión
LRU.

## Backends persistentes

Los dos backends de disco incluidos usan este constructor:

~~~python
DiskCacheBackend(
    root,
    namespace,
    max_bytes,
    max_entries=100_000,
    record_codec=None,
    lock_timeout_seconds=60.0,
    lock_poll_interval_seconds=0.01,
    remove_invalid_records=True,
)

MemoryMappedCacheBackend(
    root,
    namespace,
    max_bytes,
    max_entries=100_000,
    record_codec=None,
    lock_timeout_seconds=60.0,
    lock_poll_interval_seconds=0.01,
    remove_invalid_records=True,
)
~~~

<code>namespace</code> se transforma en un directorio aislado mediante hash y
los registros usan nombres SHA-256 opacos. Un manifiesto versionado fija
<code>namespace</code>, <code>max_bytes</code>, <code>max_entries</code> y el
fingerprint del codec. Abrir el mismo namespace con otra cuota, modo de
integridad o identificador de clave HMAC falla antes de mutar registros. Usa un
namespace nuevo para otro contrato compartido.

Cada escritura codifica y sincroniza un temporal en el mismo directorio,
reserva espacio expulsando registros completos antiguos y publica después con
<code>os.replace</code>. La reserva ocurre antes de publicar: matar el proceso
inmediatamente después del replace no puede dejar registros completos por
encima de cuota. El orden LRU se desempata por mtime de nanosegundos y nombre.
La entrada de directorio se sincroniza cuando el sistema lo permite.

<code>MemoryMappedCacheBackend</code> solo cambia la ruta de lectura: mapea un
registro serializado en vez de reservar otro resultado de
<code>read_bytes()</code>. <code>DatasetCache</code> cierra ese mapping
inmediatamente después de deserializar. La muestra reconstruida sigue ocupando
su memoria Python/Tensor habitual.

La cuota cuenta envelope y payload de cada <code>.lfcache</code> completo, no
solo la muestra serializada. Un temporal en curso, manifiesto, archivo de lock,
metadatos y unidades de asignación quedan fuera de esa cuota lógica. Como los
writers están serializados solo puede existir un temporal cooperativo por
namespace. Una muestra cuyo registro completo no cabe se rechaza sin expulsar
registros útiles.

### Fingerprints, integridad y serialización segura

<code>DatasetFingerprint(content, transform, configuration=None)</code>
canonicaliza configuración JSON/YAML y la combina con identificadores
explícitos de contenido y transformación determinista. Al pasarlo a
<code>DatasetCache</code>, su digest y el fingerprint del serializador entran
en una clave versionada. Cambiar cualquier componente declarado produce un
miss sin borrar la generación anterior.

Desde Python puede calcularse un snapshot completo y ordenado:

~~~python
fingerprint = DatasetFingerprint.from_files(
    ["data/features.npy", "data/targets.npy"],
    transform="standardize-v4",
    configuration={"epsilon": 1e-6},
)
~~~

La lectura se hace por bloques acotados. Es un snapshot, no un vigilante de
archivos, y no puede inferir la semántica de callables arbitrarios. YAML
construye <code>DatasetFingerprint</code> directamente, por lo que el digest de
archivos debe precalcularse. Omitirlo conserva la derivación de clave heredada;
los experimentos persistentes deberían declarar la identidad.

<code>CacheRecordCodec</code> verifica el envelope antes de cualquier llamada
a <code>DatasetSerializer.loads</code>:

- <code>checksum_sha256</code> es el default y detecta corrupción accidental,
  pero no autentica al escritor;
- <code>hmac_sha256</code> autentica con una clave de al menos 32 bytes por
  defecto. En YAML usa <code>authentication_key_env</code> para no versionar
  secretos.

La firma liga formato, modo, namespace, clave del registro, longitud y payload;
mover un registro válido a otra clave o namespace falla. La comparación usa
una primitiva de tiempo constante. HMAC da autenticidad e integridad, no cifra,
no impide borrado ni protege ante replay de un registro válido anterior.

<code>NumpyDatasetSerializer</code> admite mappings anidados con claves string,
listas/tuplas, escalares tipo JSON, bytes, arrays NumPy ordinarios sin objetos y
tensores CPU densos no cuantizados. Su formato ZIP/NPY determinista usa
<code>allow_pickle=False</code>. Cantidad de arrays, bytes de archivo,
manifiesto y datos decodificados, y profundidad son configurables. Valida
tamaños y cabeceras NPY antes de materializar. Rechaza arrays object/structured,
clases arbitrarias y tensores CUDA, sparse o cuantizados.

### Coordinación multiproceso y recuperación

Los backends persistentes incluidos usan un lock del sistema operativo:
compartido para lecturas copiadas y exclusivo para mutación, reconciliación y
reserva. Un <code>CacheRecord</code> mmap conserva su lease compartido hasta
<code>close()</code>, evitando expulsar un mapping Windows abierto. Timeout y
polling son explícitos; el sistema libera el lease tras salida normal o abrupta
del proceso.

La construcción y <code>recover()</code> eliminan <code>*.tmp</code> huérfanos,
validan el manifiesto y reconcilian registros completos con la cuota.
<code>usage()</code> devuelve entradas y bytes del mismo escaneo coordinado.
Los tokens de generación hacen condicional
<code>remove_if_unchanged</code>, de forma que una lectura corrupta no borre un
reemplazo concurrente más nuevo (caso ABA).

Estas garantías se limitan a procesos LambdaForge cooperativos sobre un
filesystem local con locks y replace atómico compatibles. No es una caché de
red, consenso distribuido ni reserva física dura; valida NFS/SMB por separado.

## Workers de DataLoader y planificación de capacidad

<code>DatasetCache</code> descarta deliberadamente entradas RAM, contadores y
locks al serializarse para <code>spawn</code>. También detecta un cambio de PID
tras <code>fork</code> y reinicia la RAM heredada antes del acceso. No crea
ningún <code>multiprocessing.Manager</code>, segmento de memoria compartida o
proceso auxiliar de caché.

El valor seguro predeterminado es <code>cache_in_workers: false</code>:

- con <code>num_workers: 0</code>, el acceso ocurre en el proceso de
  entrenamiento y su caché RAM está activa;
- con <code>num_workers > 0</code>, la caché RAM de los workers queda
  desactivada y solo se usa el backend opcional;
- con <code>cache_in_workers: true</code>, cada worker posee una LRU RAM vacía
  e independiente con todos los límites de bytes y entradas por proceso.

Sea <code>J</code> el número de entrenamientos independientes simultáneos,
<code>R</code> los ranks DDP por trabajo y <code>S</code> los pools DataLoader
vivos simultáneamente que envuelven un <code>DatasetCache</code>. Para la
partición <code>s</code>, sean <code>W_s</code> sus workers y
<code>C_s</code> su presupuesto de bytes:

~~~text
replicas_s = 1                              cuando W_s = 0
replicas_s = W_s                            cuando W_s > 0 y cache_in_workers
replicas_s = 0                              cuando W_s > 0 y la caché worker está apagada

techo conservador de RAM serializada
    = J × R × sum(replicas_s × C_s para cada partición viva s)
~~~

Añade por separado cualquier caché padre calentada manualmente. Los DataLoaders
de entrenamiento, validación y test pueden poseer pools distintos, y los pools
persistentes pueden coexistir. <code>persistent_workers: true</code> permite
reutilizar las cachés worker entre épocas, pero mantiene residente su memoria;
si se desactiva, las cachés worker desaparecen cuando termina su proceso.

El backend de disco puede compartirse entre workers/ranks cuando todos usan
exactamente el mismo namespace confiable y contrato de origen determinista.

## FileDataset

El constructor es:

~~~python
FileDataset(files, loader, root=None)
~~~

<code>files</code> es una secuencia ordenada explícita, no un glob ni un único
string. Las rutas se normalizan durante la construcción, pero los archivos de
muestras solo se abren desde <code>__getitem__</code>. Si se proporciona
<code>root</code>, las rutas relativas se resuelven dentro de él y se rechazan
las que escapen. Los archivos ausentes fallan al intentar acceder.

El loader recibe un <code>pathlib.Path</code> resuelto y debe ser un callable
importable y serializable cuando los workers DataLoader usan
<code>spawn</code>. <code>FileDataset</code> no retiene muestras; envuélvelo en
<code>DatasetCache</code> si la decodificación determinista resulta costosa.

## NumpyMemmapDataset

El constructor es:

~~~python
NumpyMemmapDataset(arrays, as_tensors=True)
~~~

<code>arrays</code> relaciona nombres de salida no vacíos con rutas
<code>.npy</code>. La construcción no abre archivos. El primer
<code>len(dataset)</code> o acceso indexado los abre mediante:

~~~python
np.load(path, mmap_mode="r", allow_pickle=False)
~~~

Todos los arrays deben ser no escalares y compartir su primera dimensión. Se
rechazan arrays de objetos y datos serializados con pickle. Cada fila indexada
se copia a memoria escribible antes de devolverla;
<code>as_tensors=True</code> convierte esa copia con
<code>torch.from_numpy</code>, mientras que false devuelve arrays NumPy
independientes. Por tanto, mutar un resultado no puede cambiar el archivo de
solo lectura ni muestras posteriores. El parámetro requiere un booleano real.

Los mappings son locales al proceso. La serialización spawn elimina los
handles, y un hijo creado con fork vuelve a abrirlos tras detectar su PID.
Llama explícitamente a <code>close()</code>, usa el dataset como context
manager o deja la salida del proceso como último recurso. El cierre explícito
es importante antes de reemplazar o eliminar archivos mapeados en Windows.

Este adaptador y <code>MemoryMappedCacheBackend</code> son distintos:
<code>NumpyMemmapDataset</code> mapea arrays fuente y copia una muestra;
<code>MemoryMappedCacheBackend</code> mapea un registro serializado de caché y
después lo deserializa. Ninguno impone un límite duro sobre la caché de páginas
del sistema operativo.

## Plugins de dataset

Las distribuciones reutilizables pueden publicar clases Dataset mediante el
grupo de entry points <code>lambdaforge.datasets</code>. Cada split selecciona
la clase explícitamente y pasa todos los argumentos del constructor por YAML:

~~~yaml
data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {root: datasets/acme, split: train}
~~~

La clase debe heredar de <code>torch.utils.data.Dataset</code>; cada
construcción crea una instancia nueva y el registro no retiene muestras ni
datasets. Un plugin puede anidarse como parámetro <code>dataset</code> de
<code>DatasetCache</code>, pero esto nunca activa caché implícitamente. Mantén
los constructores importables, lazy y spawn-safe. <code>IterableDataset</code>
cumple el contrato, aunque necesita un data module propio porque el adaptador
predeterminado baraja el split map-style de entrenamiento. Consulta la
[guía de plugins](../plugins/README.es.md) para publicación y procedencia.

## Ejemplo YAML completo

La siguiente es una configuración completa y endurecida. El loader debe
devolver un mapping compatible con <code>x</code> y <code>target</code>; el
callable de clave recibe un índice. Define
<code>LAMBDAFORGE_TRAIN_CACHE_HMAC_KEY</code> fuera del YAML con un secreto de
al menos 32 bytes. Sustituye el digest de ejemplo por el snapshot real
precalculado.

~~~yaml
schema_version: "1.0"

experiment:
  name: cached_file_training
  output_root: runs/experiments
  seeds: [7, 17]
  resume: true

data:
  train:
    target: lambdaforge.data.DatasetCache
    params:
      dataset:
        target: lambdaforge.data.FileDataset
        params:
          root: data/train
          files:
            - sample-000.npy
            - sample-001.npy
          loader:
            ref: my_project.data.load_training_sample
      max_memory_bytes_per_process: 67108864  # 64 MiB por proceso habilitado
      max_memory_entries: 4096
      backend:
        target: lambdaforge.data.MemoryMappedCacheBackend
        params:
          root: .cache/lambdaforge
          namespace: my-project/train-decoder-v3
          max_bytes: 4294967296               # cuota de registros completos de 4 GiB
          max_entries: 100000
          record_codec:
            target: lambdaforge.data.CacheRecordCodec
            params:
              integrity: hmac_sha256
              authentication_key_env: LAMBDAFORGE_TRAIN_CACHE_HMAC_KEY
      serializer:
        target: lambdaforge.data.NumpyDatasetSerializer
        params:
          compressed: false
          max_arrays: 1024
          max_decoded_bytes: 1073741824
      fingerprint:
        target: lambdaforge.data.DatasetFingerprint
        params:
          content: "sha256:sustituir-por-digest-real-del-dataset"
          transform: "my_project.data.load_training_sample:v3"
          configuration:
            normalization: none
      key_fn:
        ref: my_project.data.training_cache_key
      cache_in_workers: true
      strict: true

  val:
    target: lambdaforge.data.NumpyMemmapDataset
    params:
      arrays:
        x: data/validation/features.npy
        target: data/validation/targets.npy
      as_tensors: true

  test:
    target: lambdaforge.data.NumpyMemmapDataset
    params:
      arrays:
        x: data/test/features.npy
        target: data/test/targets.npy
      as_tensors: true

  datamodule:
    target: lambdaforge.training.data.LightningDataModule
    params:
      batch_size: 64
      num_workers: 4
      persistent_workers: true
      prefetch_factor: 2

model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 1
    hidden: [128, 64]

losses:
  - target: lambdaforge.nn.losses.BinaryCrossEntropyWithLogitsLoss
    params:
      output_key: logits
      target_key: target

val_metrics:
  - target: lambdaforge.metrics.classification.BinaryAUROC
    params:
      pred_key: logits
      target_key: target

task:
  params:
    model_input_key: x
    model_output_key: logits

trainer:
  max_epochs: 20
  accelerator: auto
  devices: 1
  checkpoint_policy: last_and_best
  enable_progress_bar: false

execution:
  mode: sequential
  dataloader_num_workers_per_job: 4
~~~

Con un trabajo, un rank, cuatro workers y solo la caché de entrenamiento
habilitada en los workers, el techo configurado de RAM serializada es
<code>4 × 64 MiB = 256 MiB</code>. El prefetch y los lotes vivos son
adicionales.

## Estadísticas

<code>cache.stats()</code> devuelve un <code>CacheStats</code> congelado:

| Campo | Significado |
|---|---|
| <code>memory_hits</code> | Lecturas servidas por la LRU RAM de este proceso. |
| <code>backend_hits</code> | Lecturas servidas por el backend. |
| <code>misses</code> | Peticiones que cargaron el dataset envuelto. |
| <code>writes</code> | Fallos admitidos en al menos una capa configurada. |
| <code>evictions</code> | Expulsiones de la LRU RAM local al proceso. |
| <code>skipped_oversize</code> | Muestras serializadas que no cabían en ninguna capa habilitada. |
| <code>serialization_errors</code> | Fallos al serializar claves/valores, payloads corruptos o muestras CUDA rechazadas. |
| <code>backend_errors</code> | Fallos del backend, incluidos registros checksum/HMAC rechazados. |
| <code>memory_entries</code>, <code>memory_bytes</code> | Uso actual de payload RAM del proceso. |
| <code>max_memory_bytes_per_process</code>, <code>max_memory_entries</code> | Límites RAM configurados. |
| <code>backend_entries</code>, <code>backend_bytes</code> | Un snapshot coordinado del uso del backend. |
| <code>process_id</code> | Proceso propietario de los contadores y LRU RAM. |

Los contadores son locales al proceso y se reinician al inicializar el estado
después de spawn/fork. No se agregan automáticamente entre workers. Vaciar
registros no reinicia los contadores históricos.

## Invalidación y ciclo de vida

- <code>cache.invalidate(index)</code> elimina una clave de RAM y, por defecto,
  del backend. Usa <code>include_backend=False</code> para eliminar solo RAM.
- <code>cache.clear()</code> vacía únicamente la RAM de este proceso. Usa
  <code>cache.clear(include_backend=True)</code> para vaciar también el
  namespace del backend configurado.
- Prefiere un <code>namespace</code> de backend nuevo para una versión nueva
  del dataset o transformación. La invalidación índice a índice no puede
  demostrar que encontró todos los registros obsoletos.
- La RAM se libera con el ciclo de vida de la caché/proceso. Los registros de
  disco sobreviven intencionadamente a la salida del proceso hasta la
  expulsión por cuota o el vaciado explícito.
- <code>NumpyMemmapDataset.close()</code> es idempotente y debe llamarse de
  forma determinista si se moverán, reemplazarán o eliminarán los archivos.
- Un consumidor directo del backend debe cerrar cada
  <code>CacheRecord</code> devuelto; <code>DatasetCache</code> ya lo hace
  mediante un context manager.

## Determinismo y claves de caché

Cachear un aumento aleatorio congela el resultado que gane el primer fallo.
Cachea primero la carga/decodificación determinista y aplica después el aumento
estocástico.

La clave predeterminada es el índice. Un <code>key_fn</code> puede aportar un
identificador estable de muestra, pero debe ser determinista, compatible con
spawn y con el serializador. Evita lambdas y valores aleatorios por proceso.
<code>DatasetFingerprint</code> separa la identidad
dataset/transformación/configuración de la identidad de cada muestra. Sin
fingerprint se conserva el envelope heredado; al configurarlo se usa la
versión con fingerprint e identidad de formato del serializador.

Varios workers pueden calcular el mismo fallo. El reemplazo atómico conserva
el registro completo, pero gana el último escritor. Por ello, todos los
productores que compartan un namespace deben implementar el mismo contrato
determinista.

## Seguridad y contratos de extensión

<code>PickleDatasetSerializer</code> puede ejecutar código durante la carga. Un
checksum no vuelve seguro pickle ante un escritor malicioso. Úsalo solo con
una raíz controlada por el mismo usuario/proyecto y selecciónalo explícitamente
si <code>NumpyDatasetSerializer</code> no representa la muestra. HMAC prueba
que el productor tenía el secreto, pero no que su código sea confiable. Nunca
uses pickle con directorios descargados, compartidos no confiables o escribibles
por un atacante.

El constructor de compatibilidad es
<code>PickleDatasetSerializer(protocol=pickle.HIGHEST_PROTOCOL)</code> y su
<code>format_fingerprint</code> incluye el protocolo. Cambiar el serializador
también aísla claves con fingerprint, aunque un namespace nuevo sigue siendo
la frontera operativa más clara.

Un <code>DatasetSerializer</code> propio implementa:

~~~python
dumps(value) -> bytes
loads(payload) -> object
~~~

Un <code>CacheBackend</code> propio implementa:

~~~python
read(key) -> CacheRecord | None
write(key, payload) -> bool
remove(key) -> None
clear() -> None
current_bytes: int
entry_count: int
usage() -> CacheUsage
remove_if_unchanged(key, token) -> bool
~~~

Las claves de backend son digests SHA-256 en minúsculas proporcionados por
<code>DatasetCache</code>. Los backends son responsables de la persistencia de
bytes y del cierre de recursos; no deben inferir semántica de tarea. Mantén los
objetos personalizados importables y serializables si un experimento los
construye desde YAML y sus workers DataLoader usan spawn. La clase base aporta
fallbacks compatibles para <code>usage</code> y
<code>remove_if_unchanged</code>; sobrescríbelos para uso atómico y eliminación
segura por generación.

El formato endurecido falla de forma cerrada. Si un namespace contiene
<code>.lfcache</code> heredados pero no manifiesto, el constructor lanza un
error en vez de adivinar el formato o ejecutarlo. Elige otro namespace o vacía
explícitamente esa caché antigua y desechable.
