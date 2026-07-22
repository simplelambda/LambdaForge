# Migraciones de configuración de experimentos

[Guía de experimentos](../README.es.md) · [Guía del repositorio](../../../../README.es.md) ·
[English](README.md)

Este paquete gestiona la detección de versiones, las migraciones deterministas hacia delante, la
selección exacta de JSON Schema y las previsualizaciones no destructivas del YAML de experimentos
de LambdaForge. Una migración no construye objetos configurados, resuelve plugins, importa rutas
`target`/`ref` del usuario, inicia procesos ni crea directorios de ejecución.

## Contenidos

- [Contrato de compatibilidad](#contrato-de-compatibilidad)
- [Flujo CLI seguro](#flujo-cli-seguro)
- [Formatos de previsualización y códigos de salida](#formatos-de-previsualización-y-códigos-de-salida)
- [API Python](#api-python)
- [Modelo de objetos](#modelo-de-objetos)
- [Garantías de round trip y persistencia](#garantías-de-round-trip-y-persistencia)
- [Validación y modos de fallo](#validación-y-modos-de-fallo)
- [Añadir una migración futura](#añadir-una-migración-futura)
- [Alcance actual](#alcance-actual)

## Contrato de compatibilidad

La versión actual del Schema de experimentos es el string entrecomillado `"1.1"`:

```yaml
schema_version: "1.1"
experiment:
  name: study
# ...
```

Los dos JSON Schemas empaquetados, el histórico 1.0 y el actual 1.1, exigen `schema_version`. Debe
emplear la forma exacta y entrecomillada `MAJOR.MINOR`; un valor YAML sin comillas como `1.1` es un
número y se rechaza en vez de convertirlo silenciosamente.

Las configuraciones históricas de LambdaForge no contenían este campo. Siguen admitiéndose como
entrada legacy `unversioned` mediante `UnversionedToV1Migration`. Esa migración inserta primero
`schema_version: "1.0"` y no cambia la semántica. La migración consecutiva
`ExperimentV1ToV1_1Migration` cambia después solo esa declaración a 1.1. El Schema 1.1 añade el
bloque `retention` estricto y opcional; omitirlo equivale a disabled, por lo que los experimentos
históricos no activan mutaciones de artefactos. Todo archivo nuevo debe declarar 1.1.

`ExperimentConfig` normaliza la entrada legacy en memoria en las fronteras de configuración,
incluidos `Experiment.from_yaml`, los mappings de runner/agregador y la carga de runs materializados.
El archivo de origen no se modifica. `ExperimentValidator` informa de las versiones de origen y
destino y de los pasos aplicados, valida cada resultado intermedio contra su Schema exacto y
finalmente valida el mapping normalizado contra el Schema 1.1. Usa el comando explícito para revisar
y persistir el YAML canónico.

El registro por defecto contiene una ruta consecutiva determinista:

```text
unversioned --unversioned_to_1_0--> 1.0 --1_0_to_1_1--> 1.1
```

Se rechazan downgrades, versiones inferidas y saltos sin una ruta consecutiva registrada.
Consulta la [guía de retención de artefactos](../retention/README.es.md) para la semántica en
ejecución del bloque Schema 1.1; migrar solo declara compatibilidad y nunca aplica retención.

## Flujo CLI seguro

Empieza con el diff unificado por defecto:

```powershell
lambdaforge migrate legacy.yaml
```

Previsualiza el documento migrado completo o un resultado legible por máquinas:

```powershell
lambdaforge migrate legacy.yaml --format yaml
lambdaforge migrate legacy.yaml --format json
lambdaforge migrate legacy.yaml --target-version 1.1  # actual; también es el default
lambdaforge migrate legacy.yaml --target-version 1.0  # detenerse en el Schema histórico
```

No se escribe nada salvo que `--output` indique una ruta diferente:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
```

Se rechaza un destino existente. `--force` permite sustituir ese destino, pero nunca el origen:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml --force
```

`--force` sin `--output` no es válido. `--check` está pensado para CI y no se puede combinar con
`--output`:

```powershell
lambdaforge migrate experiment.yaml --check
```

La previsualización elegida también se imprime al usar `--output` o `--check`. `--format` solo
controla esa salida estándar; el archivo indicado por `--output` siempre contiene el YAML migrado
completo. La confirmación de escritura y los errores van a la salida de error, por lo que la salida
estándar JSON sigue siendo procesable.

## Formatos de previsualización y códigos de salida

| Formato | Salida |
|---|---|
| `diff` | Diff unificado entre origen y versión de destino; es el default. Un no-op ya actualizado imprime un estado breve. |
| `yaml` | YAML resultante completo, incluido un documento actual sin cambios. |
| `json` | Sobre estable con versiones, `changed`, pasos, avisos, diff y configuración resultante. |

El sobre JSON tiene su propio `migration_result_version`. Este campo versiona el protocolo de
resultado y es independiente del `schema_version` del experimento. Un YAML leído desde archivo ya
contiene valores portables. En un mapping programático, el `result.config` semántico conserva
tuplas, rutas y fechas de Python, mientras el sobre JSON las proyecta de forma determinista a
arrays JSON, strings de ruta nativos y strings ISO; un objeto opaco no soportado se representa con
una etiqueta estable de su tipo. Esta proyección nunca altera la configuración usada por el
framework.

| Condición | Código de salida |
|---|---:|
| Previsualización válida o escritura explícita completada | `0` |
| `--check` y no hace falta migrar | `0` |
| `--check` y hace falta al menos un paso de migración | `1` |
| YAML/Schema/versión/ruta inválidos, salida insegura o fallo de escritura | `1` |
| Sintaxis de terminal rechazada por `argparse` | `2` |

Como `--check` usa `1` deliberadamente para señalar una configuración desactualizada, ejecuta
primero la previsualización normal cuando necesites distinguir esa condición esperada de un error.

## API Python

La fachada es el punto de entrada de solo lectura más corto:

```python
from lambdaforge import LambdaForge
from lambdaforge.experiments import MigrationPreviewFormat

preview = LambdaForge.preview_migration("legacy.yaml")
print(preview.source_version)  # unversioned
print(preview.target_version)  # 1.1
print(preview.changed)
print(preview.render(MigrationPreviewFormat.DIFF))
```

Los objetos públicos de bajo nivel admiten archivos y mappings:

```python
from lambdaforge.experiments import ExperimentConfigMigrator

migrator = ExperimentConfigMigrator.default()
file_result = migrator.preview_file("legacy.yaml")
mapping_result = migrator.preview_mapping(raw_config)

payload = file_result.to_dict()
file_result.write_yaml("experiment.v1_1.yaml")
```

`preview_mapping` hace una copia profunda, conserva los tipos Python programáticos y nunca muta los
objetos anidados del llamador. El payload JSON con forma de mapping es de solo lectura en
profundidad; `result.config` y `to_dict()` devuelven copias defensivas independientes.
`write_yaml(path, overwrite=False)` es una segunda operación explícita; previsualizar no tiene
efectos de persistencia. Usa `overwrite=True` únicamente para un destino diferente que
intencionadamente se pueda sustituir.

Las abstracciones principales se exportan de forma lazy desde `lambdaforge.experiments`. Los
objetos para contribuidores, incluidos el paso incorporado y el codec round trip, también se
exportan desde `lambdaforge.experiments.migrations`. Los módulos de cada clase son detalles de
implementación.

## Modelo de objetos

| Objeto | Responsabilidad |
|---|---|
| `ExperimentSchemaVersion` | Objeto valor `MAJOR.MINOR` exacto y ordenable, más el marcador interno `unversioned`. |
| `ExperimentSchemaCatalog` | Asocia versiones exactas con Schemas Draft 2020-12 empaquetados, cachea validadores por objeto y detecta divergencias en su declaración. |
| `ExperimentConfigMigration` | Contrato abstracto para una transformación exacta hacia delante. |
| `UnversionedToV1Migration` | Paso incorporado solo de compatibilidad que declara el Schema 1.0. |
| `ExperimentV1ToV1_1Migration` | Paso incorporado consecutivo que declara el Schema 1.1 actual conservando la semántica de todos los campos 1.0. |
| `ExperimentConfigMigrationStep` | Descriptor inmutable de identificador/versiones/descripción guardado en resultados. |
| `ExperimentConfigMigrationRegistry` | Planificador inmutable, determinista y solo hacia delante. |
| `ExperimentConfigMigrator` | Copia, planifica, aplica, valida y renderiza una cadena de migraciones. |
| `ExperimentConfigMigrationResult` | Previsualización inmutable compatible con mappings, renderizadores y escritor YAML atómico explícito. |
| `MigrationPreviewFormat` | Enum para `diff`, `yaml` y `json`. |
| `RoundTripYamlCodec` | Codec YAML UTF-8 que rechaza claves duplicadas y conserva presentación. |

El registro rechaza identificadores duplicados, más de una migración saliente desde la misma
versión y pasos que no avancen. `with_migration(...)` devuelve un registro nuevo en vez de mutar
estado global.

## Garantías de round trip y persistencia

Las previsualizaciones de archivo usan YAML round trip para conservar comentarios, orden de
mappings, estilo de strings entrecomillados, anchors y el convenio de saltos de línea dominante
cuando la estructura transformada lo permite. Un no-op devuelve el texto original exacto. Un
documento modificado todavía puede recibir normalización inocua de presentación, algo advertido en
el resultado; inspecciona siempre el diff.

El lector acepta exactamente un documento mapping en UTF-8 y rechaza claves duplicadas. No invoca
la factoría de objetos, el registro de plugins ni la validación de imports de LambdaForge, por lo
que una previsualización no puede instanciar modelos, pérdidas, datasets, callbacks o loggers
configurados. No es un sandbox general: el texto resultante y el sobre JSON pueden contener
credenciales u otros valores ya presentes, y LambdaForge no los oculta.

La persistencia está separada intencionadamente de la planificación:

1. Origen y destino deben resolverse a rutas diferentes incluso si se permite sobrescribir.
2. Un destino existente exige `overwrite=True`/`--force` explícito.
3. El padre del destino solo se crea cuando se solicita una escritura.
4. El YAML se escribe y vacía en un temporal único junto al destino.
5. Sin permiso de sobrescritura, el temporal completo se enlaza atómicamente sobre un destino
   ausente, por lo que dos writers concurrentes no pueden pisarse. Con permiso explícito de
   sobrescritura se usa sustitución atómica.
6. El residuo temporal se elimina tras un éxito o una excepción gestionados.

No existe modo in-place ni copia de seguridad automática porque el origen nunca es destino de
escritura. Una terminación abrupta del proceso o de la máquina todavía puede dejar el temporal de
nombre único, que nunca se confunde con el destino.

## Validación y modos de fallo

Con el `validate=True` por defecto, el migrador valida cada paso aplicado contra el Schema de destino
exacto. También valida un no-op actual. El marcador `unversioned` no tiene Schema independiente:
la entrada legacy se transforma y valida primero como 1.0, y después como 1.1.

La migración falla antes de persistir cuando:

- la raíz YAML no es un mapping, tiene claves duplicadas o no se puede parsear;
- `schema_version` no es un string `MAJOR.MINOR` exacto y entrecomillado;
- el destino solicitado carece de Schema empaquetado;
- el registro no tiene una ruta hacia delante completa o se solicita un downgrade;
- un paso emite una versión equivocada, un valor no mapping o un documento de destino inválido;
- la salida es el origen, ya existe sin permiso o no se puede sustituir.

La validación de migraciones no comprueba intencionadamente imports `target`/`ref` ni disponibilidad
de plugins. Ejecuta después `lambdaforge validate migrated.yaml` para validar expansión, recursos e
imports opcionales:

```powershell
lambdaforge migrate legacy.yaml --output experiment.v1_1.yaml
lambdaforge validate experiment.v1_1.yaml
```

## Añadir una migración futura

Un cambio incompatible futuro del Schema debe introducirse como una cadena revisada de la release,
no inferirse a partir del contenido:

1. Empaqueta el nuevo Schema Draft 2020-12 y declara su `schema_version` exacto.
2. Añádelo a `ExperimentSchemaCatalog.DEFAULT_SCHEMA_FILES`.
3. Implementa una clase `ExperimentConfigMigration` con identificador estable, versiones
   origen/destino exactas, descripción para el usuario y un método `apply` determinista.
4. Añade ese objeto al registro inmutable por defecto en orden consecutivo.
5. Prueba el no-op, la cadena completa, salidas intermedias inválidas, presentación round trip,
   previsualizaciones CLI y persistencia atómica.
6. Documenta los cambios semánticos en ambas guías antes de convertir la versión en actual.

Cada paso se comprueba inmediatamente contra su Schema de destino. El registro permite
deliberadamente un solo paso saliente por versión, de modo que la evolución futura conserva una
historia lineal determinista salvo que se rediseñe conscientemente ese contrato.

## Alcance actual

Los Schemas 1.0 y 1.1 están empaquetados y la cadena real actual es
`unversioned → 1.0 → 1.1`. El segundo paso introduce la superficie opcional del Schema para retener
artefactos sin activarla. No están implementados el downgrade de Schema, la reescritura in-place,
las fuentes remotas, la ocultación de secretos ni las migraciones aportadas por plugins. Estos
límites mantienen revisables los cambios de compatibilidad y local la persistencia, mientras futuras
releases añaden pasos consecutivos solo cuando el Schema cambie.
