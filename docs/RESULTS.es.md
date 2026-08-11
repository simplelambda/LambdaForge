[English](RESULTS.md) | [Español](RESULTS.es.md)

# Guía de resultados y plots

## 1. Seleccionar evidencia con seguridad

```bash
lambdaforge results list --root runs
lambdaforge results show baseline --root runs
lambdaforge results compare baseline ablation --metric val_loss --direction minimize
lambdaforge results export baseline --series --format csv --output analysis/curves.csv
```

Los selectores aceptan path existente, attempt ID, fingerprint, nombre de run/experimento o
variante. `show` devuelve todos los candidatos y marca ambigüedad: no elige “el último”. El comando
anterior `lambdaforge results CONFIG --write-index --fail-on-ambiguous` sigue compatible como
`results audit`. Para publicar, una ambigüedad de éxitos exige seleccionar attempt explícito.

`MetricSeries` normaliza `metrics.csv` a run, seed, variant, split, metric, step, value y timestamp,
sin duplicar logs. Los nombres de métricas son exactos y los errores enumeran los disponibles.
Los deltas usan el primer selector como baseline; best/worst sólo se etiqueta con
`--direction minimize|maximize`, sin adivinar la semántica de la métrica.

## 2. Curvas y seeds

`plot learning` acepta `individual` o `mean` y `none`, `std` o `ci`. Con `n=1` los límites son null:
no se inventa incertidumbre. Varias métricas usan small multiples. `plot seeds` crea box, violin o
strip. `plot learning JOB --follow` sincroniza sólo métricas pequeñas, reemplaza el plot
atómicamente y termina con el job o Ctrl-C.

## 3. Sweeps, HPO y recursos

`plot sweep CONFIG --x PATH [--y PATH] --metric NAME` agrega correctamente por seeds y conserva
`n`. Las celdas 2-D ausentes siguen ausentes salvo `--interpolate` explícito. `--normalize` aplica
min-max por métrica sobre celdas observadas y conserva valores raw en el spec. `plot hpo` lee el state
durable y muestra objetivo, best-so-far, presupuesto, estado y parámetro opcional; no modifica HPO.
`plot resources` sólo usa telemetría ya registrada y falla indicando nombres disponibles si no hay.

## 4. Reproducibilidad

`VisualizationService` crea primero un `PlotSpec` inmutable; `--json` lo entrega sin renderizar.
PNG/SVG/PDF usan Matplotlib y HTML autocontenido requiere `lambdaforge[viz]`. Escritura y reemplazo
son atómicos. Cada figura guarda `FIGURE.plot.json` con spec/fingerprint/timestamp para regeneración
y caché. Los plots bajo `plots/` de un run exponen figura y spec en `artifact list`.

Véase [clusters](CLUSTERS.es.md) para sincronización y descarga explícita.
