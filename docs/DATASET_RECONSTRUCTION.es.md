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
   contenido publicado, no reconstruye arrays. La réplica nativa comprimida admite ubicaciones
   locales/remotas y prefiere SSH directo entre clústeres; véase abajo.
2. **Reconstruir**: calcular un candidato independiente, sellado y no registrado para compararlo.
3. **Publicar**: usar una versión nueva para bytes distintos, incluso con equivalencia científica.
   No se admiten representaciones alternativas bajo la misma versión ni placements con otro hash.

## Publicar una vez, replicar el contenido exacto

```bash
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --json
# Revisar ruta, tamaño y destino; después aplicar:
lf datasets replicate corpus@7 --source gpu12 --destination gpu16 --apply
```

En `lf`, abre **Datasets → DatasetVersion → Replicate…**, elige origen/destino, revisa el plan nativo
y confirma. La consola sigue respondiendo y muestra fases/tiempo transcurrido y bytes comprimidos
en la ruta retransmitida. Mantén la sesión abierta hasta terminar. Locations se actualiza tras el
registro. El destino requiere `storage.dataset_root` y runtime LambdaForge disponible (existente o
preparado mediante bootstrap); no ejecuta Work científico/claim GPU. Mantiene el scope del proyecto.

`--route auto` comprueba SSH desde origen a destino con clave de host fiable, sin prompts ni
forwarding de agente/credenciales. Si funciona, los bytes viajan directamente entre hosts. Configura
en el origen autenticación no interactiva autorizada por el centro y claves de host fiables para
habilitarlo; LambdaForge no instala claves ni copia credenciales locales. `--route direct` exige esa
conexión. `--route relay` usa los transportes autenticados del controlador con un **stream comprimido
en memoria acotada**, no un archivo intermedio en tu equipo ni en `/tmp`.

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

### Fallos antes del sellado (0.17.1)

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

## Conflictos entre registros

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

## Certificados de equivalencia durables

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
