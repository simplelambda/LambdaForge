# Productos científicos durables: base implementada

[English](PRODUCTS.md)

Esta es la capa funcional de **modelo/registro/transporte e inputs Work explícitos**, no el sistema
completo de orquestación de Studies. Hay decisiones nativas, selección local de modelos,
dependencias Work tipadas y registros de consumo real. Un Study nativo individual puede declarar
publicación automática de decisiones/modelos al finalizar. Import/export nativo de Study de un host
incluye productos publicados. La consola incluye un navegador de metadata/auditorías e import de
Study con confirmación. Siguen pendientes espera/replanificación y export Fleet completo.
Véase el [registro de reforma](ARCHITECTURAL_REFORM.es.md). Ejecución,
recovery y publicación exacta de datasets actuales no cambian.

## Identidades y contratos

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

## API de publicación explícita

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

## CLI nativa

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

## Selección post-Study de ModelSet

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

## StudyDecision nativa

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

## Publicación declarada y retry de publicación sin entrenar

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

## Dependencias Work tipadas

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

## Export/import portable de productos

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

## Importar un Study portable sin ejecutarlo

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

Export incluye productos de la declaración/recibo nativo, sus bytes independientes e historial del
productor. Transferencia trae solo esos objetos, no todo el catálogo; muestreo de telemetría nunca
altera bytes de productos. Ausencias fallan explícitamente. Publicaciones de operador no declaradas
usan `products export`, sin recorrer catálogos ajenos para descubrirlas.

Registro del archivo hace commit atómico antes de aliases de productos con locks independientes.
Un fallo posterior deja el archivo verificado disponible: repetir import completa aliases. No es
transacción distribuida entre stores. Borrar evidencia importada conserva productos independientes.
