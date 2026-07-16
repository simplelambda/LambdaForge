# Métricas de LambdaForge

[Guía del repositorio](../../../README.es.md) · [English](README.md)

Las métricas son objetos con estado, ciclo de vida explícito y dirección de optimización. Son
independientes de Lightning y sirven en cualquier bucle que proporcione mapas de salida y lote.

## Contenidos

- [Contrato de métrica](#contrato-de-métrica)
- [Métricas distribuidas](#métricas-distribuidas)
- [Clasificación binaria](#clasificación-binaria)
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
contrato, LambdaForge genera un error descriptivo en lugar de un valor verosímil pero incorrecto. Se
usa `all_gather_object`; reunir tensores enormes puede ser caro y motiva futuras métricas streaming.

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

`BinaryAUROC` y `BinaryAUPRC` acumulan scores de ordenación y targets. Pueden ser logits o
probabilidades porque sigmoid es monótona. Devuelven `NaN` si no están ambas clases. TorchMetrics es
la implementación instalada; scikit-learn queda como fallback opcional.

## Clasificación multiclase

- `MulticlassAccuracy`
- `MulticlassBalancedAccuracy`
- `MulticlassF1`
- `MulticlassAUROC`
- `MulticlassAUPRC`

Las entradas son scores/logits `(muestras, clases)` y targets enteros `(muestras,)`. `num_classes`
puede validarse explícitamente o inferirse. Las curvas usan reducciones one-vs-rest de TorchMetrics;
el valor incluido por defecto es macro.

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
