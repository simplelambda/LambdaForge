# Datasets de primera clase

Español | [English](DATASETS.md) | [Guía raíz](../README.es.md)

## 1. Entidad y fuente de verdad

Un registro nombra una versión inmutable (`nombre@versión`) y su `dataset_id` derivado del
contenido. Incluye muestras/splits, productor, linaje y placements físicos. Cada placement guarda
clúster, raíz exacta, bytes, número de ficheros y verificación. `DatasetArtifact`, junto a los
bytes, es autoritativo; `.lambdaforge/datasets.json` es un índice atómico reconstruible.

Un preprocesado correcto que produzca `dataset-artifact.json` se registra automáticamente. Si el
índice falla se escribe un evento, pero no se invalida o reescribe el manifiesto científico. El
`DataCatalog` YAML anterior sigue soportado para referencias/loaders declarativos, pero no es
obligatorio para listar, inspeccionar o gestionar el ciclo de vida.

## 2. Descubrimiento e inspección

```bash
lambdaforge datasets list
lambdaforge datasets list --on atlas
lambdaforge datasets list --all
lambdaforge datasets show corpus@v3
lambdaforge datasets locations corpus@v3
lambdaforge datasets stats corpus@v3 --on atlas
lambdaforge datasets verify corpus@v3 --on atlas
lambdaforge datasets lineage corpus@v3
```

El inventario remoto consulta la ruta registrada; no escanea el filesystem. Stats universales son
bytes, ficheros, muestras/splits del manifiesto y formato simple. No se adivinan labels ni dominio.

Para clasificación se aporta un schema explícito:

```yaml
task: classification
format: csv
file: examples.csv
target: label
classes: [cat, dog]
```

`datasets stats corpus@v3 --schema classification.yaml` calcula conteos/proporciones, targets
ausentes e imbalance. Un schema de proyecto puede declarar
`profiler: {target: mi_proyecto.data.WisdomProfiler}` con contrato
`profile(root, record, schema) -> mapping`. El núcleo no incorpora supuestos WISDOM.

## 3. Registro, remove y delete

```bash
lambdaforge datasets add RUN/dataset-artifact.json --root RUN
lambdaforge datasets remove corpus@v3 [--on atlas]
lambdaforge datasets delete corpus@v3 --on atlas
lambdaforge datasets delete corpus@v3 --on atlas --apply
```

`remove` sólo modifica el registro. `delete` selecciona un placement exacto y es preview por
defecto. Apply exige manifiesto/identidad/hashes válidos, rechaza rutas amplias/home/root y
consumidores activos; después quita el placement. GC jamás selecciona datasets.

## 4. Materialización y réplica

```bash
lambdaforge datasets materialize corpus@v3 --on atlas
lambdaforge datasets materialize corpus@v3 --on atlas --strategy replicate --apply
lambdaforge datasets replicate corpus@v3 --from local --to atlas
lambdaforge datasets replicate corpus@v3 --from local --to atlas --apply
```

El plan determinista es `NOOP`, `REPLICATE` o `BUILD`; muestra bytes y si el controlador local debe
seguir online. Nada copia datos grandes durante `run --on` ni en preview. La transferencia incluida
admite fuente local y destino local/SSH con `rsync` explícito. No finge que un relay por portátil
sobrevive a su desconexión. Filesystems compartidos o sistemas del centro deben aportar un provider
o tarea durable; BUILD pide ejecutar el productor cuando sus entradas estén presentes.

## 5. Contratos Python

Los objetos públicos son `DatasetRecord`, `DatasetPlacement`, `DatasetRegistry`, `DatasetService`,
`DatasetMaterializationPlan`, `DatasetDeletionPlan` y `DatasetProfiler` en `lambdaforge.data`. Usa
el servicio, no globs sobre el fichero de registro. Un profiler recibe raíz/record exactos y no debe
inferir semántica científica silenciosamente.
