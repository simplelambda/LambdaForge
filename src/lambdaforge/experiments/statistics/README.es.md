# Comparaciones estadísticas

[Guía de experimentos](../README.es.md) · [Guía del repositorio](../../../../README.es.md) ·
[English](README.md)

Este paquete ofrece estimaciones de incertidumbre y pruebas pareadas orientadas a objetos y
seleccionables mediante YAML para comparar experimentos entre semillas. Opera sobre métricas
escalares ya materializadas; no carga modelos, datasets ni checkpoints.

## Contenidos

- [Contrato de comparación](#contrato-de-comparación)
- [Referencia YAML completa](#referencia-yaml-completa)
- [Emparejamiento y semántica de improvement](#emparejamiento-y-semántica-de-improvement)
- [Intervalos de confianza](#intervalos-de-confianza)
- [Pruebas pareadas](#pruebas-pareadas)
- [Artefactos y compatibilidad](#artefactos-y-compatibilidad)
- [API Python](#api-python)
- [Interpretación](#interpretación)

## Contrato de comparación

`StatisticalComparisonConfig.from_mapping` lee únicamente `aggregation.comparisons`, rechaza claves
del framework desconocidas y materializa todos los defaults. Los ajustes específicos de una
estrategia se siguen validando y escribiendo en `statistical_protocol` aunque dicha estrategia no
esté activa, de forma que un cambio posterior de método sea explícito y reproducible.

| Clave YAML | Default | Valores válidos |
|---|---:|---|
| `alpha` | `0.05` | Número estrictamente entre 0 y 1. |
| `target_power` | `0.80` | Número estrictamente entre 0 y 1. |
| `min_pairs_for_verdict` | `3` | Entero de al menos 1. |
| `confidence_interval.method` | `normal` | `normal`, `bootstrap_percentile`. |
| `confidence_interval.confidence_level` | `0.95` | Número estrictamente entre 0 y 1. |
| `confidence_interval.resamples` | `10000` | Entero entre 1 y 10.000.000. |
| `confidence_interval.seed` | `0` | Entero no negativo. |
| `confidence_interval.batch_size` | `1024` | Entero entre 1 y 1.000.000. |
| `confidence_interval.max_batch_elements` | `1000000` | Entero entre 1 y 100.000.000. |
| `paired_test.method` | `sign` | `sign`, `wilcoxon`. |
| `paired_test.alternative` | `observed_direction` | `two_sided`, `greater`, `less`, `observed_direction`. |
| `paired_test.calculation` | `auto` | `auto`, `exact`, `asymptotic`. |
| `paired_test.zero_method` | `wilcox` | `wilcox`, `pratt`, `zsplit`. |
| `paired_test.continuity_correction` | `false` | Booleano. |
| `paired_test.exact_max_pairs` | `50` | Entero entre 0 y 200. |
| `paired_test.zero_tolerance` | `1.0e-12` | Número finito no negativo. |
| `paired_test.round_decimals` | `12` | `null` o entero entre 0 y 15. |

Es válido omitir `aggregation`, `comparisons` o cualquiera de los mappings de estrategia. Omitir
todo el bloque reproduce el protocolo agregado anterior: intervalo normal al 95 %, prueba pareada
exacta de signos, `observed_direction`, alpha 0,05, potencia objetivo 0,80 y tres pares antes de
emitir un veredicto.

## Referencia YAML completa

```yaml
aggregation:
  comparisons:
    alpha: 0.05
    target_power: 0.80
    min_pairs_for_verdict: 3
    confidence_interval:
      method: bootstrap_percentile
      confidence_level: 0.95
      resamples: 10000
      seed: 0
      batch_size: 1024
      max_batch_elements: 1000000
    paired_test:
      method: wilcoxon
      alternative: two_sided
      calculation: auto
      zero_method: wilcox
      continuity_correction: false
      exact_max_pairs: 50
      zero_tolerance: 1.0e-12
      round_decimals: 12
```

El [ejemplo canónico](../../../../examples/experiment.yaml) incluye el mismo bloque.

## Emparejamiento y semántica de improvement

Las comparaciones son pareadas, nunca se tratan como muestras independientes:

1. Una variante `parent__ablation` usa `parent` como baseline cuando existe. Otras variantes
   distintas de base usan el literal `base` si está presente.
2. Solo entran las semillas con valores finitos en ambos lados. `n_pairs` y `paired_seeds` exponen
   la muestra resultante.
3. `delta = variante - baseline`.
4. `improvement = delta` para métricas `max` y `-delta` para métricas `min`. Una improvement
   positiva siempre significa mejor.

El modo explícito de la métrica o monitor tiene prioridad. Si no existe metadata de modo, los
nombres que contienen `loss`, `time`, `seconds`, `mem` o `rss` se tratan como `min` y el resto como
`max`. Configura el modo del monitor de forma explícita siempre que este fallback pueda resultar
ambiguo.

Si no hay semillas comunes, pares finitos o baseline aplicable se genera metadata no disponible
explícita, nunca un fallback no pareado.

## Intervalos de confianza

Ambos estimadores tienen como objetivo la media aritmética de las improvements pareadas.

### Normal

`NormalConfidenceInterval` usa la desviación estándar muestral, el error estándar y un valor crítico
normal bilateral para `confidence_level`. Menos de dos pares producen
`status: unavailable`/`reason: insufficient_samples`. Una varianza nula genera un intervalo
degenerado. Es el default de compatibilidad.

### Bootstrap percentil determinista

`BootstrapConfidenceInterval` remuestrea las improvements con reemplazo, guarda cada media
remuestreada y toma cuantiles lineales inferior/superior. También informa la desviación estándar
bootstrap de esas medias.

La reproducibilidad es local a cada comparación. LambdaForge combina mediante SHA-256 la `seed`
base con la identidad canónica `(baseline_variant, variant, metric)` e inicializa un flujo PCG64 con
la `effective_seed` de 64 bits resultante. Reordenar métricas o añadir otra comparación no modifica
un intervalo existente.

El uso de memoria está acotado deliberadamente:

- el array conservado contiene una media `float64` por remuestreo, es decir `O(resamples)`;
- la matriz temporal de índices tiene
  `min(batch_size, max(1, max_batch_elements // n_pairs)) * n_pairs` elementos;
- si un único remuestreo ya supera `max_batch_elements`, una fila es el mínimo inevitable.

Con menos de dos pares se devuelve un resultado no disponible sin reservar matrices de remuestreo.
Las muestras constantes devuelven extremos constantes y `degenerate: true` sin usar aleatoriedad.

## Pruebas pareadas

`SignTest`, el default de compatibilidad, aplica una prueba binomial exacta a los signos de
improvements fuera de `zero_tolerance`. Los empates no entran en su tamaño efectivo.

`WilcoxonSignedRankTest` ordena los rangos absolutos de las improvements pareadas y evalúa su suma
con signo. Usa rangos promedio para magnitudes iguales, informa `has_rank_ties` y admite:

| Ajuste | Significado |
|---|---|
| `alternative: two_sided` | Usa el doble de la cola menor. |
| `alternative: greater` | Contrasta improvement positiva (variante mejor). |
| `alternative: less` | Contrasta improvement negativa (variante peor). |
| `alternative: observed_direction` | Elige `greater` o `less` según la media observada. |
| `calculation: exact` | Enumera la distribución condicional de signos; por encima de `exact_max_pairs` devuelve no disponible. |
| `calculation: asymptotic` | Usa aproximación normal, opcionalmente con corrección de continuidad. |
| `calculation: auto` | Exacta hasta `exact_max_pairs` pares no nulos; asintótica por encima. |

El cálculo exacto usa programación dinámica determinista sobre los rangos observados (escalados a
medios rangos), por lo que admite rangos promedio sin desempates aleatorios. `exact_max_pairs` acota
este trabajo; un cálculo exacto pedido explícitamente nunca cambia a asintótico en silencio.

`round_decimals` se aplica antes de detectar ceros y asignar rangos. Después, los valores cuya
magnitud absoluta no supera `zero_tolerance` son ceros:

- `wilcox` elimina los ceros antes de asignar rangos;
- `pratt` los incluye al asignar rangos pero excluye sus rangos de la suma aleatoria de signos;
- `zsplit` sigue el ranking de Pratt y reparte a partes iguales la contribución de los rangos cero
  entre las estadísticas positiva y negativa informadas.

Si todos los pares son cero se obtiene
`status: unavailable`/`reason: no_nonzero_differences` en lugar de NaN. `PairedTestResult` siempre
expone el p-valor seleccionado, diagnósticos bilaterales/mejor/peor, cálculo pedido/usado,
estadísticas de rango y conteos efectivos/de ceros.

## Artefactos y compatibilidad

La agregación con Schema versión 4 escribe:

| Artefacto | Contenido estadístico |
|---|---|
| `aggregate/baseline_comparisons.csv` | Una fila por comparación baseline/variante/métrica con intervalo y prueba seleccionados. |
| `aggregate/reliability.json` | Protocolo materializado, regla de baseline, comparaciones, primarias y recomendaciones de semillas. |
| `aggregate/summary.json` | Resumen del protocolo, conteos de comparaciones y rutas de artefactos. |

Los campos neutrales respecto al método incluyen `confidence_interval_method`, `confidence_level`,
`confidence_interval_low/high`, `confidence_interval_standard_error`, estado/motivo del intervalo,
metadata de semillas bootstrap, `paired_test_method`, alternativa, cálculo pedido/usado,
estadísticas de rango, conteos efectivos/de ceros, todos los p-valores pareados y estado/motivo de
la prueba.

`p_value_directional` es el p-valor elegido por `alternative`. La corrección Benjamini-Hochberg
sobre todas las filas disponibles produce `q_value_bh_directional` y el p-valor seleccionado
controla el veredicto. La recomendación de semillas sigue siendo una aproximación normal basada en
el efecto observado y se informa junto a su motivo.

Por compatibilidad, `ci95_improvement_low/high` conserva siempre el intervalo normal histórico al
95 %, y `wins`, `losses`, `ties` junto a `p_value_sign_*` conservan los diagnósticos de la prueba
exacta de signos. `p_value_wilcoxon_*` solo se rellena cuando se selecciona Wilcoxon. Las
estimaciones y p-valores ausentes usan `null` en JSON/celdas vacías en CSV, nunca NaN.

## API Python

El namespace estable `lambdaforge.experiments` exporta:

- `StatisticalComparisonConfig`;
- `ConfidenceIntervalMethod` y `ConfidenceIntervalResult`;
- `PairedAlternative`, `PairedTestMethod` y `PairedTestResult`;
- `WilcoxonCalculation` y `WilcoxonZeroMethod`.

El namespace específico `lambdaforge.experiments.statistics` exporta además
`StatisticalComparisonEngine`, `NormalConfidenceInterval`, `BootstrapConfidenceInterval`,
`SignTest` y `WilcoxonSignedRankTest`.

```python
from lambdaforge.experiments import StatisticalComparisonConfig
from lambdaforge.experiments.statistics import StatisticalComparisonEngine

protocol = StatisticalComparisonConfig.from_mapping(
    {
        "aggregation": {
            "comparisons": {
                "confidence_interval": {"method": "bootstrap_percentile", "seed": 17},
                "paired_test": {"method": "wilcoxon", "alternative": "two_sided"},
            }
        }
    }
)
engine = StatisticalComparisonEngine(protocol)
interval = engine.confidence_interval(
    [0.02, 0.01, 0.03],
    identity=("base", "candidate", "val_auroc"),
)
test = engine.paired_test([0.02, 0.01, 0.03])
assert interval.status == "ok"
assert test.method == "wilcoxon"
```

Ambas clases de resultado son dataclasses congeladas con slots y `to_dict()` para artefactos o
integraciones. Los constructores de configuración, estimadores y pruebas validan la seguridad
numérica; YAML recibe las mismas comprobaciones mediante el Schema empaquetado y el objeto de
configuración.

## Interpretación

Estas salidas son primitivas exploratorias del framework, no un diseño de estudio universal. Decide
dirección de métricas, alternativa, nivel de confianza, número de semillas y política de
multiplicidad antes de interpretar un experimento confirmatorio. En particular,
`observed_direction` sigue intencionadamente la media observada y resulta útil en informes
exploratorios; para inferencia confirmatoria es preferible declarar de antemano `two_sided`,
`greater` o `less`. Las muestras pareadas muy pequeñas tienen poca resolución incluso con una
prueba exacta, y Wilcoxon asintótico y las recomendaciones de semillas siguen siendo aproximaciones.
