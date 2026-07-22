# Métricas de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Las métricas son objetos con estado, ciclo de vida explícito y dirección de optimización. Son
independientes de Lightning y sirven en cualquier bucle que proporcione mapas de salida y lote.

## Contenidos

- [Contrato de métrica](#contrato-de-métrica)
- [Métricas distribuidas](#métricas-distribuidas)
- [Clasificación binaria](#clasificación-binaria)
- [Métricas de curva exactas y streaming](#métricas-de-curva-exactas-y-streaming)
- [Clasificación multiclase](#clasificación-multiclase)
- [Regresión](#regresión)
- [Escribir una métrica](#escribir-una-métrica)

## Contrato de métrica

```python
metric.reset()
for outputs, batch in evaluation_stream:
    metric.update(outputs, batch)
value = metric.compute()
```

Cada `Metric` tiene `name`, `higher_is_better` y una `direction` derivada (`max` o `min`). Las
implementaciones separan los tensores de autograd y normalmente guardan estado en CPU.
`LightningTask` copia profundamente las métricas por etapa, reinicia al comenzar la época, actualiza
por lote y registra el escalar final.

Los nombres deben ser únicos dentro de una etapa porque se convierten en claves de monitorización y
logging YAML. Envuelve una métrica en `MetricAlias` si necesitas reutilizar una implementación:

```yaml
val_metrics:
  - target: lambdaforge.metrics.MetricAlias
    params:
      name: strict_accuracy
      metric:
        target: lambdaforge.metrics.classification.BinaryAccuracy
        params: {threshold: 0.8}
```

Los casos vacíos o indefinidos devuelven `NaN` cuando no existe un valor razonable, como AUROC con
una única clase objetivo. La agregación debe conservar esa ausencia, no convertirla en cero.

## Métricas distribuidas

Una métrica no lineal no se puede calcular por rank y promediar. Antes de `compute`,
`Metric.synchronize` reúne el `distributed_state` de cada rank y llama a
`merge_distributed_state`. Las métricas incluidas fusionan predicciones, targets, conteos o
estadísticos suficientes.

Una métrica propia con estado en DDP debe implementar ambos métodos. Si hay varios ranks y falta el
contrato, LambdaForge genera un error descriptivo en lugar de un valor verosímil pero incorrecto. El
contrato genérico usa `all_gather_object`, por lo que las métricas que conservan tensores de muestras
pueden ser caras. Las curvas binarias y multiclase streaming sobrescriben `synchronize()` con un
`all_reduce` tensorial aditivo cuyo tamaño no depende del número de muestras.

## Clasificación binaria

Las implementaciones binarias y multiclase viven en subpaquetes físicos distintos para que cada
carpeta tenga un contrato claro. Los imports públicos estables siguen siendo
`lambdaforge.metrics.classification.<NombreClase>`; el consumidor no debe depender de la distribución
interna de archivos.

Las métricas derivadas de confusión comparten `pred_key`, `target_key` y umbral configurables:

- `BinaryAccuracy`
- `BinaryBalancedAccuracy`
- `BinaryPrecision`
- `BinaryRecall`
- `BinarySpecificity`
- `BinaryF1`
- `BinaryMCC`
- `BinaryCohenKappa`

Usa probabilidades con un umbral como `0.5`, o un umbral coherente con la representación de scores.
`BinaryConfusionCounts` almacena TP/TN/FP/FN como estado suficiente compartido.

## Métricas de curva exactas y streaming

`BinaryAUROC` y `BinaryAUPRC` son las variantes exactas. Acumulan cada score de ordenación y target
en CPU, por lo que su memoria es proporcional al número de muestras. Los scores pueden ser logits o
probabilidades porque sigmoid no altera la ordenación exacta. TorchMetrics es la implementación
instalada; scikit-learn queda como fallback opcional.

`StreamingBinaryAUROC` y `StreamingBinaryAUPRC` son alternativas explícitas de memoria acotada.
Conservan dos histogramas CPU `int64`, uno positivo y otro negativo, de `num_bins` elementos. La
carga útil de los histogramas ocupa exactamente `16 * num_bins` bytes, sin contar los pequeños
overheads de Python y del allocator. El valor predeterminado de 4096 bins usa 65.536 bytes de carga
útil por métrica, con independencia del tamaño del dataset; procesar una actualización aún requiere
memoria temporal proporcional a ese lote.

```yaml
val_metrics:
  - target: lambdaforge.metrics.classification.StreamingBinaryAUROC
    params:
      pred_key: logits
      target_key: y
      from_logits: true
      num_bins: 4096
  - target: lambdaforge.metrics.classification.StreamingBinaryAUPRC
    params:
      pred_key: logits
      target_key: y
      from_logits: true
      num_bins: 4096
```

`from_logits` es deliberadamente explícito. Usa `true` para aplicar sigmoid antes de agrupar; con
`false`, cada score debe ser finito y estar ya en `[0, 1]`. Autodetectarlo por lote sería inseguro
porque distintos lotes podrían emplear transformaciones incompatibles. Los targets solo pueden
contener cero y uno.

Los bins son intervalos uniformes de probabilidad. Las muestras de un mismo bin se tratan como
empates:

- AUROC streaming cuenta pares positivo-negativo concordantes y da medio acierto a los pares dentro
  de un mismo bin;
- AUPRC streaming calcula average precision, ponderando precisión por incrementos de recall al
  recorrer bins de mayor a menor score. No emplea integración trapezoidal.

Más bins aumentan la resolución, pero ninguna métrica promete un error universal como
`1 / num_bins`; el error también depende de cómo ocupan los bins los scores positivos y negativos.
Para uso científico, compara varios números de bins con la métrica exacta en un subconjunto
representativo, fija la resolución antes del experimento final y consérvala en YAML. Las variantes
exactas y streaming devuelven `NaN` si el estado está vacío o falta cualquiera de las dos clases.

El estado streaming es serializable y fusionable. `reset()` limpia ambos histogramas e inicia un
nuevo ciclo. Tras sincronizar DDP se rechazan actualizaciones hasta el reset para no mezclar conteos
globales ya reducidos con nuevos conteos locales. La sincronización suma un tensor fijo
`2 × num_bins` mediante `all_reduce`, con estado y comunicación `O(num_bins)` respecto al dataset.
`MetricAlias` delega esta sincronización especializada.

### Curvas multiclase streaming

`StreamingMulticlassAUROC` y `StreamingMulticlassAUPRC` aplican el mismo enfoque acotado por
histogramas one-vs-rest. Su estado persistente son exactamente dos tensores CPU `int64` con forma
`(num_classes, num_bins)`, es decir, `16 * num_classes * num_bins` bytes de carga útil.
`num_classes` es obligatorio porque reservar o cambiar ese estado implícitamente haría impredecible
el contrato de memoria.

```yaml
val_metrics:
  - target: lambdaforge.metrics.StreamingMulticlassAUROC
    params:
      num_classes: 10
      num_bins: 4096
      average: macro
      undefined_class_policy: ignore
      pred_key: logits
      target_key: y
      from_logits: true
  - target: lambdaforge.metrics.StreamingMulticlassAUPRC
    params:
      num_classes: 10
      num_bins: 4096
      average: weighted
      undefined_class_policy: ignore
```

`average` acepta `macro`, `weighted` o `micro`. `compute_per_class()` expone siempre todos los
resultados one-vs-rest. `undefined_class_policy` controla las clases sin positivos o negativos con
`ignore`, `nan` o `zero`; nunca se infiere silenciosamente de un lote. Con `from_logits: true` los
scores pasan por softmax. Con `false` deben ser finitos, estar en `[0, 1]` y, por defecto, cada fila
debe sumar uno dentro de `probability_tolerance`; `validate_probability_sum` puede desactivar solo
esta última comprobación.

La cautela por agrupación en bins es la misma que en binario: la resolución es configurable, pero el
número de bins no implica una cota de error universal. Compara con la métrica exacta en un
subconjunto representativo antes de fijar la configuración final. DDP reduce un único tensor fijo
`(2, num_classes, num_bins)`, por lo que la comunicación tampoco depende del número de muestras. Las
reducciones siguen las definiciones one-vs-rest actuales documentadas por
[TorchMetrics AUROC](https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html) y
[average precision](https://lightning.ai/docs/torchmetrics/stable/classification/average_precision.html).

## Clasificación multiclase

- `MulticlassAccuracy`
- `MulticlassBalancedAccuracy`
- `MulticlassF1`
- `MulticlassAUROC`
- `MulticlassAUPRC`
- `StreamingMulticlassAUROC`
- `StreamingMulticlassAUPRC`

Las entradas son scores/logits `(muestras, clases)` y targets enteros `(muestras,)`. `num_classes`
puede validarse explícitamente o inferirse en las métricas exactas; las curvas streaming lo exigen.
Las curvas usan reducciones one-vs-rest y su valor predeterminado es macro.

## Regresión

- `MAE`, `MSE`, `RMSE`
- `R2Score`
- `PearsonCorrelation`, `SpearmanCorrelation`
- `MeanMetric` para promediar cualquier salida escalar con nombre

Las claves de predicción y target son configurables. Las métricas aditivas fusionan estadísticos
suficientes entre ranks. Las correlaciones fusionan valores; Spearman asigna rangos medios en
empates. R² y correlaciones devuelven `NaN` si el denominador o número de muestras las hace
indefinidas.

## Escribir una métrica

Crea una clase en un módulo e implementa:

```python
class ProjectMetric(Metric):
    def update(self, outputs, batch, context=None): ...
    def compute(self) -> float: ...
    def reset(self) -> None: ...
    def distributed_state(self): ...
    def merge_distributed_state(self, state): ...
```

Separa el estado, valida formas pronto y declara la dirección correcta. Usa claves genéricas pasadas
al constructor. Prueba estado vacío, valores normales, extremos, reset, varias actualizaciones y
estados fusionados. Exporta desde el inicializador de `classification`, `regression` y/o superior.
Un proyecto externo no necesita modificar LambdaForge: indica la clase cualificada de su propio
paquete en `train_metrics`, `val_metrics` o `test_metrics`. Usa listas de etapa explícitas si la
métrica posee un recurso que no se puede copiar profundamente.
