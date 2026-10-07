# Datasets publicados y reconstrucción científica

Español · [English](DATASET_RECONSTRUCTION.md)

`NOMBRE@VERSION` identifica un único content ID inmutable. Una validación científica **PASS** no
demuestra igualdad de bytes ni equivalencia de todos los miembros. LambdaForge nunca sobrescribe
esta versión, relaja checksums de transporte, redondea arrays o adivina qué campos internos de un
NPZ u otro formato son operativos.

## Elegir la intención antes de calcular

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
   contenido publicado, no reconstruye arrays. Se mantienen las restricciones de transferencia
   entre remotos; no hay retransmisión implícita de grandes assets por el controlador.
2. **Reconstruir**: calcular un candidato independiente, sellado y no registrado para compararlo.
3. **Publicar**: usar una versión nueva para bytes distintos, incluso con equivalencia científica.
   No se admiten representaciones alternativas bajo la misma versión ni placements con otro hash.

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

## Tres identidades separadas

- **Científica**: `scientific_identity` explícita del proyecto; exige fuentes, selección, etiquetas,
  configuración y contrato del algoritmo. Manifiesto/registro exponen `scientific_id`. Es una
  declaración, no prueba de reproducibilidad. Los manifiestos históricos sin declaración siguen
  siendo legibles; para bytes diferentes su equivalencia queda sin resolver.
- **Contenido**: `dataset_id == content_id`, checksums exactos de assets, tamaños e índice lógico
  canónico. No cambia transporte ni verificación. Una versión existente tampoco puede adquirir
  silenciosamente otra declaración científica aunque tenga los mismos bytes.
- **Operativa**: procedencia de construcción, rutas, Execution/Run/Attempt, hardware y fechas del
  manifiesto. No pertenecen a content ID. Meterlas dentro de un asset sí cambia sus bytes.

## Comparar reconstrucciones completas

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

## Recuperar una publicación sin recalcular

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

## Conflictos entre registros

`lf datasets list --all` y Datasets en la consola descubren los índices pequeños de todos los
clústeres configurados. Mismo nombre/versión/content fusiona ubicaciones; contenido distinto sigue
en filas separadas y marcadas **CONFLICT**. Listar no modifica registros y el descubrimiento
incompleto queda visible. Las filas conflictivas solo exponen resumen cacheado, no operaciones de
miembros/eliminación con selector lógico ambiguo. Inspecciona cada destino con
`lf datasets reconcile corpus@6 --on CLUSTER`: copias divergentes existentes/inaccesibles se
rechazan; solo una inscripción cuyo directorio se demuestra ausente puede retirarse con `--apply`.
El investigador elige la referencia, no el orden de descubrimiento ni la mayoría. Conserva los bytes
divergentes y publícalos bajo versión nueva explícita después de revisar su evidencia.

## Cambios necesarios en WISDOM (no realizados aquí)

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
