# Resultados de Work, evidencia histórica y productos independientes

[English](WORK_RESULTS.md) · [Manual](MANUAL.es.md) · [Productos](PRODUCTS.es.md)

Un Work es una ejecución científica, no necesariamente un entrenamiento. Un Study añade un diseño
de parámetros/semillas y coordinación científica. Comparten runner, Execution/Run/Attempt, outputs
gestionados y las transacciones de exportación/importación. El informe de Work no inventa objetivos,
ajusta modelos estadísticos ni mezcla evidencias de Attempts distintos.

## Inspección e informes

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

## Seleccionar dependencias históricas exactas

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

### ResultInput: evidencia de un Execution

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

### ProductInput: contenido científico duradero

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

## Promoción explícita desde un Work ordinario

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

## Portabilidad y disponibilidad remota

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
