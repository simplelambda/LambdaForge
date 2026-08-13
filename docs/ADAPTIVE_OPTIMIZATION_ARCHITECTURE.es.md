[English](ADAPTIVE_OPTIMIZATION_ARCHITECTURE.md) | [Español](ADAPTIVE_OPTIMIZATION_ARCHITECTURE.es.md)

# Arquitectura de optimización adaptativa

Este documento es para cambiar controlador/scheduler de HPO. La guía de uso está en el README.

## 1. Autoridad y ciclo

`AdaptiveExperimentOptimizer` posee `state.json`, `events.jsonl` y `summary.json` bajo
`SUITE/.lambdaforge/adaptive/STUDY_ID`. Al relanzar el mismo YAML reconcilia acciones pendientes con
resultados ordinarios. Cada decisión es START_NEW, RESUME o ADD_SEED; CONFIRM es una fase separada.
Una promoción expresa presupuesto acumulado y el runner reanuda checkpoint sin repetir epochs.

`AdaptiveExperimentController` compone searcher, política de fidelidad, racer de seeds,
`LearningCurveModel`, coste/memoria, admisión y selector. Los límites públicos son duck typed,
devuelven valores inmutables y mantienen orden estable. El runner de training no se sustituye.

## 2. Adquisición y curvas

Sobol inicializa. Con BoTorch, categorías no ordenadas usan Hamming, ordinales conservan orden,
condiciones llevan sentinel+mask, fidelidad entra en `f(x,b)` y live actions en `X_pending`. Todas
las acciones se comparan con la misma aproximación de Knowledge Gradient gaussiano dividida por
coste y multiplicada por viabilidad de memoria. Si falla el fit, se reintenta numérica segura, se
registra `HPO_SURROGATE_FALLBACK` y se usa Sobol.

Para `n` seeds, la incertidumbre de la media es:

$$
\operatorname{Var}(\bar{\mu}) = \frac{\tau^2}{n} +
\frac{v_1 + \cdots + v_n}{n^2}.
$$

`tau²` mide variación entre seeds y `v_s` la varianza de estimación de cada curva. Diferencias
pareadas con seeds compartidos alimentan racing/pruning; no se aplica dos veces `1/n`.

## 3. Memoria y scheduling

La capacidad es UNKNOWN, UNBOUNDED o KNOWN(N). `FeatureAwareMemoryModel` usa parámetros y
`resource_features`, conserva OOM censurado como `M(x,z)>L` y añade margen/cuántil conservador.
`allocator_cap` es un techo defensivo de PyTorch, no aislamiento, y nunca reduce batch.

Preflight recibe candidato y contexto, ejecuta forward/backward/step representativo en child
aislado y se reserva para candidatos fríos, inciertos, OOD o cercanos al límite. Un probe legacy sin
argumentos sigue compatible. `TrainingOrchestrator` observa una finalización y rellena enseguida el
slot. Acciones, asignaciones y límites de acciones/epochs/GPU-seconds quedan persistidos.

## 4. Invariantes

- search seeds compartidos y ordenados; confirmation seeds disjuntos;
- no se mezcla `sweep` con HPO adaptativo;
- objective es columna exacta de `metrics.csv`;
- RESUME necesita checkpoint `last`;
- observaciones conservan curva, score, coste, memoria, OOM, run y error;
- estado se reemplaza atómicamente y eventos usan lock+fsync;
- summary informa seeds, curvas, memoria y confirmación sin seleccionar por mtime.

0.5.3 no cambia esta matemática; resuelve entornos CUDA sin alterar el HPO existente.
