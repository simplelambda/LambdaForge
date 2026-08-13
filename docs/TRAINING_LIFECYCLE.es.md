# Auditoría del lifecycle de training y acciones post-run

[Guía del repositorio](../README.es.md) · [English](TRAINING_LIFECYCLE.md)

## 1. Decisión

La lógica ligada al loop ya pertenecía a callbacks Lightning configurables. Tasks, operaciones de
modelo y Workflows ya cubrían trabajo posterior con scheduling independiente. El hueco real era más
estrecho: trabajo declarativo tras un run concreto, con checkpoint, completion/fallo y artifacts
reanudables, sin repetir fit al cambiar sólo ese análisis.

Por ello se conservan callbacks, se añade `PostRunAction` para ese hueco y Task/Workflow sigue siendo
la frontera de recursos/lifecycle. No se añadió ningún concepto de MIL, proteínas, geometría,
reconstrucción o modelo de dominio.

## 2. Auditoría de los cambios locales provisionales

| Cambio provisional | Clasificación | Resolución |
|---|---|---|
| `TerminalEvaluationContext` con config/run/checkpoint | USEFUL BUT NEEDS GENERALIZATION | Sustituido por `PostRunContext` inmutable con result, seed/variant, roles estrictos, digest, artifact path y estado reanudable. |
| `TerminalEvaluationService` con fallback best→last | BADLY PLACED y USEFUL BUT NEEDS GENERALIZATION | Sustituido por `PostRunService`; selección explícita, recibos, artifacts y política de fallo. |
| Evaluar después de escribir `result.json` correcto | BADLY PLACED | Training se confirma aparte; el éxito canónico sólo se publica tras las acciones required. |
| Flag global `fail_on_error` | USEFUL BUT NEEDS GENERALIZATION | Sustituido por `required` por acción; fallos optional permanecen visibles. |
| Un `terminal-evaluation.json` propio | DUPLICATES EXISTING API | Eliminado; se reutilizan `ArtifactDeclaration`/`TaskArtifact` dentro de recibos. |
| Evaluación terminal dentro de `RunFingerprint` | BADLY PLACED | `post_run` tiene sub-fingerprint y no cambia identidad de training. |
| Guardar pausas HPO para evaluar al podar | BADLY PLACED | Eliminado: una pausa no es éxito final. Default sólo confirmation; todos los terminales correctos exige scope explícito. |
| Casos especiales completed/early/cancelled/pruned | USEFUL BUT NEEDS GENERALIZATION | Reducidos a un gate genérico; cancelación y pausa nunca ejecutan acciones de éxito. |
| Construcción de `callbacks:` mediante `ObjectFactory`/`LightningRunner` | NECESSARY | Conservada sin wrappers LambdaForge para hooks Lightning. |
| Lógica científica de callback en core | TOO DOMAIN-SPECIFIC | No se conserva ni añade; sólo se exponen outputs genéricos desacoplados. |

## 3. Capacidad existente y hueco confirmado

| Requisito | Callback | Task/Workflow | Hook interno previo | PostRunAction |
|---|---:|---:|---:|---:|
| Hook batch/época | sí | no | no | no |
| Reutilizar forward de validation | sí, ahora con `model_outputs` | no | no | no |
| Otros recursos/clúster | no | sí | no | no |
| Contexto estable run/checkpoint/result | parcial/vivo | bindings, otro run | privado | sí |
| Completion required/optional | no | por nodo | no | sí |
| Hash/provenance de artifacts | proyecto | sí | no | sí, tipos compartidos |
| Reanudar sólo downstream | depende de Lightning | sí, DAG aparte | no | sí |
| Cambiar informe sin fit | no aplica | sí | sin API declarativa | sí |

`on_run_finished` sigue siendo una notificación programática de executor/suite para refrescar
agregados, no un lifecycle científico público. `InferenceTask`, `EvaluationTask` y `ExportTask`
siguen siendo correctos cuando el downstream merece identidad o allocation propios.

## 4. Secuencia de completion e identidad

```text
fit/test correcto en rank cero
  -> confirmar .lambdaforge/post-run/training-result.json
  -> seleccionar checkpoint y calcular identidad por acción
  -> reutilizar recibo verificado o ejecutar
  -> hashear artifacts declarados con TaskArtifact
  -> fallo required: publicar resultado fallido y conservar training
  -> required correctas: publicar result.json reutilizable atómicamente
```

El fingerprint de training excluye `post_run`. La identidad de acción incluye target, params,
checkpoint, política required, artifacts declarados, identidad científica y digest del checkpoint.
Una interrupción no crea recibo de éxito; conserva su state dir por identidad. Relanzar reconcilia
acciones antes de cualquier fit.

## 5. Fronteras explícitas

- `current` significa checkpoint final/actual persistido (`last`), no pesos vivos irreproducibles.
- Las acciones son secuenciales y reutilizan allocation; no son schedulers.
- Sólo ejecuta rank global cero; la reducción científica distribuida pertenece al proyecto.
- Los objetivos HPO se publican durante validation; post-run no cambia una decisión ya observada.
- Checkpoints HPO intermedios pausados/podados no se presentan como runs finales correctos.

