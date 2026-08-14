# Descubrimiento de plugins de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Este paquete descubre clases publicadas por distribuciones instaladas por separado mediante la
metadata estándar de entry points de Python. Los plugins complementan las rutas YAML `target`
completas; instalar o usar un plugin nunca requiere editar LambdaForge.

## Contenidos

- [Empieza aquí](#empieza-aquí)
- [Objetos públicos](#objetos-públicos)
- [Grupos de entry points](#grupos-de-entry-points)
- [Publicar un plugin](#publicar-un-plugin)
- [Usar plugins en YAML](#usar-plugins-en-yaml)
- [API de Python](#api-de-python)
- [Descubrimiento por línea de comandos](#descubrimiento-por-línea-de-comandos)
- [Validación](#validación)
- [Procedencia de plugins cargados](#procedencia-de-plugins-cargados)
- [Carga, caché y seguridad](#carga-caché-y-seguridad)
- [Contratos y precedencia](#contratos-y-precedencia)

## Empieza aquí

La mayoría de proyectos **no** necesita un plugin. Si una clase vive en el mismo proyecto de
investigación instalado, referencia directamente `target: mi_proyecto.modulo.Clase`. Crea un plugin
sólo cuando otra distribución Python deba publicar una extensión reutilizable con nombre y sus
consumidores no deban conocer la ruta del módulo.

| Situación | Opción preferida |
|---|---|
| Una clase pertenece a un proyecto | `target` YAML completo. |
| Varios proyectos instalan un paquete reutilizable | `plugin` mediante entry point. |
| Se pasa un callable existente sin construirlo | `ref` YAML. |

El descubrimiento lee metadata de distribuciones instaladas de forma lazy. Resolver importa código
Python externo: instala sólo paquetes de confianza y audita la distribución/versión registrada con
cada run.

## Objetos públicos

| Objeto | Responsabilidad |
|---|---|
| `PluginKind` | Conjunto cerrado de contratos admitidos y sus grupos de metadata. |
| `PluginReference` | Selección `(kind, name)` inmutable y validada. |
| `PluginDescriptor` | Metadata inmutable instalada/resuelta, incluida distribución y versión. |
| `PluginRegistry` | Descubrimiento lazy, contratos, caché de clases y diagnóstico del proceso. |
| `PluginUsageSession` | Snapshot por ejecución, gestionado por contexto, de resoluciones correctas. |
| `PluginResolutionError` | Fallo contextual por plugins ausentes, ambiguos, no cargables o inválidos. |

Importa estos objetos desde `lambdaforge.plugins`; los módulos de cada clase son detalles de
implementación.

## Grupos de entry points

| Tipo YAML | Grupo de distribución | Clase exportada requerida |
|---|---|---|
| `model` | `lambdaforge.models` | Subclase de `torch.nn.Module` |
| `metric` | `lambdaforge.metrics` | Subclase de `lambdaforge.metrics.Metric` |
| `activation` | `lambdaforge.activations` | Subclase de `lambdaforge.nn.activations.Activation` |
| `normalization` | `lambdaforge.normalizations` | Subclase de `lambdaforge.nn.normalizations.Normalization` |
| `loss` | `lambdaforge.losses` | Subclase de `lambdaforge.nn.losses.Loss` |
| `distance` | `lambdaforge.distances` | Subclase de `lambdaforge.nn.distances.Distance` |
| `pooling` | `lambdaforge.pooling` | Subclase de `lambdaforge.nn.pooling.Pooling` |
| `similarity` | `lambdaforge.similarities` | Subclase de `lambdaforge.nn.similarities.Similarity` |
| `kernel` | `lambdaforge.kernels` | Subclase de `lambdaforge.nn.kernels.Kernel` |
| `encoding` | `lambdaforge.encodings` | Subclase de `lambdaforge.nn.encodings.Encoding` |
| `regularization` | `lambdaforge.regularization` | Subclase de `lambdaforge.nn.regularization.Regularization` |
| `dataset` | `lambdaforge.datasets` | Subclase de `torch.utils.data.Dataset` |
| `callback` | `lambdaforge.callbacks` | Subclase de `lambdaforge.integrations.Lightning.Callback` |
| `logger` | `lambdaforge.loggers` | Subclase de `lambdaforge.integrations.Lightning.Logger` |
| `task` | `lambdaforge.tasks` | Subclase de `lambdaforge.tasks.Task` |

Los grupos están separados deliberadamente: un nombre identifica una implementación dentro de un
contrato y el mismo nombre puede existir legítimamente en grupos distintos. Los nombres distinguen
mayúsculas y deberían contener únicamente letras, números, guiones bajos, puntos y guiones.

## Publicar un plugin

Un paquete externo declara las clases en su propio `pyproject.toml`:

```toml
[project]
name = "acme-lambdaforge"
dependencies = ["lambdaforge>=0.7,<0.8"]

[project.entry-points."lambdaforge.models"]
acme_encoder = "acme_lambdaforge.models:AcmeEncoder"

[project.entry-points."lambdaforge.metrics"]
calibrated_auc = "acme_lambdaforge.metrics:CalibratedAUROC"

[project.entry-points."lambdaforge.activations"]
acmegelu = "acme_lambdaforge.activations:AcmeGELU"

[project.entry-points."lambdaforge.losses"]
calibrated_focal = "acme_lambdaforge.losses:CalibratedFocalLoss"

[project.entry-points."lambdaforge.datasets"]
acme_records = "acme_lambdaforge.data:AcmeRecords"

[project.entry-points."lambdaforge.callbacks"]
artifact_marker = "acme_lambdaforge.callbacks:ArtifactMarker"

[project.entry-points."lambdaforge.loggers"]
jsonl_logger = "acme_lambdaforge.logging:JsonLinesLogger"

[project.entry-points."lambdaforge.tasks"]
surface_builder = "acme_lambdaforge.tasks:SurfaceBuilder"
```

El valor usa la sintaxis estándar `modulo.importable:atributo` de los entry points. Cada atributo ha
de ser una clase que cumpla el contrato de su grupo. Instala la distribución en el mismo entorno de
Python que LambdaForge antes de resolverla; una instalación editable sirve durante el desarrollo.

Los strings de activación y normalización pasan por la normalización de `ComponentRegistry`: se
ignoran mayúsculas, guiones bajos y guiones. Publica su nombre de entry point normalizado, por
ejemplo `acmegelu`; el YAML podrá usar `acme-gelu`, `Acme_GELU` o `acmegelu`.

## Usar plugins en YAML

Una especificación de plugin es explícita y separa `params` de la metadata de descubrimiento:

```yaml
model:
  plugin:
    kind: model
    name: acme_encoder
  params:
    in_features: 32
    hidden_features: 128

val_metrics:
  - plugin:
      kind: metric
      name: calibrated_auc
    params:
      output_key: logits
      target_key: target

data:
  train:
    plugin: {kind: dataset, name: acme_records}
    params: {split: train, root: datasets/acme}
  val:
    plugin: {kind: dataset, name: acme_records}
    params: {split: validation, root: datasets/acme}

callbacks:
  - plugin: {kind: callback, name: artifact_marker}
    params: {filename: finished.txt}

trainer:
  logger:
    plugin: {kind: logger, name: jsonl_logger}
    params: {path: metrics.jsonl}
```

Un documento de tarea genérica selecciona el contrato específico en su raíz:

```yaml
schema_version: "1.0"
kind: task
name: build-surfaces
task:
  plugin: {kind: task, name: surface_builder}
  params: {resolution: 1.0}
```

`ObjectFactory` resuelve la clase y construye `params` recursivamente igual que con `target`. Cada
construcción crea una instancia nueva. Las formas `target` y `ref` existentes siguen admitidas y se
pueden anidar dentro de los parámetros de un plugin. `trainer.logger` admite un logger o una lista
no vacía que combine `target`, `ref` y plugins logger; los modos `csv`/`lightning_csv`/`none` no
cambian.

Los plugins de activación y normalización usan su alias corto en parámetros de modelos compatibles:

```yaml
model:
  target: lambdaforge.nn.models.MLP
  params:
    in_features: 32
    out_features: 1
    hidden: [64]
    activation: acme-gelu
```

## API de Python

```python
from lambdaforge.experiments import ObjectFactory
from lambdaforge.plugins import PluginKind, PluginReference, PluginRegistry

registry = PluginRegistry.default()
solo_metadata = registry.discover(PluginKind.MODEL)
with registry.usage_session() as uso:
    clase_modelo = registry.resolve(PluginReference(PluginKind.MODEL, "acme_encoder"))
    modelo = ObjectFactory.build(
        {
            "plugin": {"kind": "model", "name": "acme_encoder"},
            "params": {"in_features": 32},
        },
        plugins=registry,
    )

usados_en_este_contexto = uso.descriptors()
resueltos_por_el_proceso = registry.resolved_plugins()
```

`discover()` devuelve descriptores inmutables y no carga sus clases. `resolve()` devuelve una clase
validada, no una instancia. Proporcionar un registro a `ObjectFactory` facilita la inyección de
dependencias y las pruebas; el código de aplicación normal puede usar el registro del proceso.
`usage_session()` registra solo resoluciones correctas dentro de su contexto, incluidos aciertos de
caché y aliases externos de activación/normalización. Elimina duplicados y ordena canónicamente el
snapshot. `resolved_plugins()` es el historial diagnóstico más amplio del proceso y no sustituye a
una sesión por ejecución.

Llama a `registry.refresh()` después de instalar o eliminar distribuciones dentro de un intérprete
activo. `refresh(PluginKind.METRIC)` invalida únicamente una categoría.

## Descubrimiento por línea de comandos

```powershell
lambdaforge plugins
lambdaforge plugins --kind metric
lambdaforge plugins --json
```

El comando lista nombre, grupo, referencia del objeto, distribución y versión desde la metadata sin
importar los módulos. Se muestran todos los proveedores duplicados; intentar resolver un
`(kind, name)` ambiguo falla en vez de escoger uno según el orden del entorno.

## Validación

La validación normal comprueba el JSON Schema, confirma que el entry point existe, importa su clase
y verifica el contrato sin instanciarla:

```powershell
lambdaforge validate experiment.yaml
```

El schema exige `kind: model` para `model`, `kind: metric` para listas de métricas, `kind: loss`
para pérdidas, `kind: dataset` para `data.train/val/test`, `kind: callback` para `callbacks` y
`kind: logger` para `trainer.logger`. Todas las categorías siguen siendo válidas en parámetros
construidos recursivamente. Un tipo superior incorrecto se informa antes de ejecutar y los objetos
anidados validan su contrato cuando `ObjectFactory` los resuelve.
El Schema independiente de tareas 1.0 exige `kind: task` para el plugin de su raíz.

La validación de plantillas puede omitir deliberadamente toda carga externa de `target`, `ref` y
plugins:

```powershell
lambdaforge validate experiment.yaml --no-imports
```

Este modo valida la estructura y la expansión, pero no demuestra que los paquetes referenciados
estén instalados ni cumplan sus contratos. El informe registra que no se comprobaron los imports.
La validación con imports tampoco cuenta deliberadamente como uso: una ejecución posterior solo
registra el plugin si su grafo de objetos llega a resolverlo.

## Procedencia de plugins cargados

Cada ejecución real materializada registra automáticamente en `environment.json` los entry points
que resolvió correctamente:

```json
{
  "plugins": [
    {
      "kind": "model",
      "name": "acme_encoder",
      "group": "lambdaforge.models",
      "value": "acme_lambdaforge.models:AcmeEncoder",
      "distribution": "acme-lambdaforge",
      "version": "2.1.0"
    }
  ]
}
```

La lista es determinista y contiene exactamente los seis campos mostrados; distribución y versión
son `null` cuando la metadata instalada no los expone. No aparecen plugins instalados pero no
usados, solo descubiertos, fallidos, ambiguos o con contrato inválido. Una clase resuelta
correctamente sí permanece si después falla su constructor o el entrenamiento. Los imports
completos mediante `target`/`ref` no se inventan como distribuciones de plugins.

La procedencia queda aislada al contexto y proceso de la ejecución. Ejecuciones secuenciales,
validaciones anteriores, el listado CLI y el padre de un worker `spawn` no la contaminan. Un dry-run
no importa objetos y escribe `"plugins": []`. Las resoluciones realizadas únicamente por procesos
hijo creados por el usuario o workers de DataLoader pertenecen a esos procesos y no se atribuyen al
padre sin una integración IPC explícita del usuario.

## Carga, caché y seguridad

El descubrimiento solo lee metadata de las distribuciones instaladas. La resolución llama al método
`load()` del entry point elegido, que importa su módulo y puede ejecutar Python a nivel de módulo.
Por tanto, los plugins deben tratarse como código instalado de confianza, igual que los targets YAML
completos; este mecanismo no es un sandbox de procesos.

El registro guarda en caché la metadata descubierta y las clases resueltas correctamente. Nunca
guarda instancias de modelos o métricas, datasets, tensores ni datos del usuario. `refresh()`
invalida las cachés de descubrimiento/clase, pero no reescribe el hecho diagnóstico de que un
descriptor ya se resolvió. Cada proceso creado con `spawn` posee su propio registro; un cambio de PID
tras `fork` sustituye locks, cachés, historial diagnóstico y contextos activos heredados. No se
transfieren instancias vivas ni recursos abiertos entre entrenamientos.

Los fallos de importación conservan la excepción original como causa. Los nombres ausentes muestran
los disponibles y los conflictos enumeran cada distribución y referencia. No existe fallback al
primer proveedor encontrado.

## Contratos y precedencia

Los modelos pueden ser subclases ordinarias de `torch.nn.Module`; no es obligatorio heredar de la
base opcional `Model`. Las métricas deben implementar el ciclo de vida completo de `Metric`, incluido
el estado distribuido si se usan con DDP. Pérdidas, activaciones, normalizaciones, distancias,
poolings, similitudes, kernels, codificaciones y regularizadores deben heredar de su clase base de
LambdaForge correspondiente.

Los plugins de dataset exponen una clase derivada de `torch.utils.data.Dataset`, lo que incluye
`IterableDataset`. El datamodule predeterminado baraja el split de entrenamiento; un dataset
iterable/streaming debe aportar un `data.datamodule.target` compatible. Los autores de callbacks y
loggers deberían heredar de `lambdaforge.integrations.Lightning.Callback` y
`lambdaforge.integrations.Lightning.Logger`, que siguen la selección Lightning moderna/legada de
LambdaForge. Los entry points exponen clases, no singletons ni factorías. Sus constructores deberían
ser spawn-safe y posponer archivos, sockets y servicios hasta el ciclo de ejecución normal.
Los plugins `task` deben heredar de `lambdaforge.tasks.Task`; los `target` de tarea siguen admitiendo
duck typing para facilitar código local del proyecto.

Para aliases de activación y normalización se comprueban primero el registro explícito del proceso y
los componentes incorporados. Por ello un paquete descubierto no puede reemplazar silenciosamente
`relu`, `layernorm` u otro alias existente. Las referencias explícitas de plugins de cualquier
categoría declaran siempre su tipo y nombre exacto, sin inferencia contextual ni strings mágicos.
