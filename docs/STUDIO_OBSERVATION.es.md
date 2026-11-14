# Observación de estudios en clientes de escritorio

`JobService.study_run` acepta evidencia científica local fuera de la carpeta de
asignación únicamente cuando el proyecto más cercano, su identificador, el nombre
del Work, su huella científica completa y el `execution.json` sellado coinciden
con el Job registrado. Solo permite el árbol `runs/` de esa ejecución exacta y
rechaza rutas simbólicas. No amplía permisos remotos ni cambia la recuperación.

`lambdaforge.analysis.ObservedStudyChart.summarize` proyecta de forma descriptiva
un análisis nativo. Permite elegir métricas, parámetro X (o `trial`), un segundo
parámetro opcional, `center="mean"|"median"` y
`spread="none"|"sd"|"se"|"iqr"|"range"`. Rechaza aplicar el error estándar
de una media a una mediana. No sustituye resultados ausentes o censurados por
mejores observaciones parciales. Todas las combinaciones pesan igual, aunque
tengan distintas cantidades de semillas. Devuelve observaciones individuales,
estimaciones agrupadas, soporte, intervalos exactos y su ámbito. Distingue la
dispersión entre combinaciones de la incertidumbre entre semillas; una observación
única no implica varianza cero.

La ejecución sellada debe identificar el Job. Las ejecuciones históricas sin
ese sello solo se aceptan en la ruta canónica exacta de la huella; se rechazan
repeticiones sin sello y evidencias que pertenecen a otro Job.

Estas gráficas no ajustan modelos, seleccionan ganadores ni modifican HPO. La
interpretación científica procede de `parameter_conclusions` e
`interaction_conclusions`, conservando sus avisos sobre evidencia y estabilidad.
