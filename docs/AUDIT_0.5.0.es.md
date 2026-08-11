[English](AUDIT_0.5.0.md) | [Español](AUDIT_0.5.0.es.md)

# Auditoría de LambdaForge 0.5.0

Esta auditoría se completó antes de implementar 0.5.1. Combina la revisión del código con los logs
fallidos reales de GitHub Actions `31499871979` para el commit `e2fe74d`.

## Fallos confirmados

- `ArchitectureConformanceCase.write_reference()` hace `fsync` sobre un descriptor de solo lectura.
  Windows devuelve `OSError: [Errno 9] Bad file descriptor` en todos los Python soportados.
- `TrainingOrchestrator.run_dynamic()` notifica todos los procesos terminados en un ciclo antes de
  rellenar el primer slot liberado. Bajo carga rompe el contrato observar-decidir del controlador.
- El smoke de distribución instala el wheel con `--no-deps` y después importa el paquete. Al faltar
  deliberadamente la dependencia declarada `jsonschema`, no prueba una instalación real.
- `PreprocessingTask` usa `ThreadPoolExecutor` para cualquier `workers > 1`; `workload: cpu` no
  cambia el comportamiento.
- La reescritura del CSV denso no es atómica y un lector concurrente puede verlo incompleto.

## Fallos potenciales no reproducidos

- No hay fallos independientes de Python 3.12-3.14 en Ubuntu. El patrón observado procede de los
  defectos temporales y específicos de plataforma anteriores.
- No existe evidencia para reducir la matriz de Python o eliminar Windows.
- El transporte SSH conserva la política normal de OpenSSH; no se encontró un bypass de seguridad.

## Funcionalidad ausente

- Preprocesado CPU multiproceso con protocolo seguro para `spawn` y sink propiedad del padre.
- Alias amigables de training y referencias tipadas a datasets dentro de especificaciones.
- Entornos gestionados e identificados por contenido con builds exactos y wheelhouse offline.
- Sincronización remota ligera y descarga lógica de artefactos.
- Servicios reutilizables para resultados, series, plots, inspección segura y debug muestral.
- Parejas inglés/español de todas las guías mantenidas.

## Soluciones existentes que se reutilizan

- Los Schemas estrictos y runners actuales siguen siendo el IR y la única ruta de ejecución.
- `ResultCatalog` ya es la fuente autoritativa y `ExperimentRegistry` exporta JSON/CSV/Parquet; no se
  añadirá otra base de datos.
- `metrics.csv`, agregación, estado HPO, recursos y `JobStore` ya contienen la evidencia necesaria.
- Transportes, schedulers, bundles, jobs persistentes y replicación explícita forman la base remota.
- Los workflows ya muestran placement y rechazan correctamente ejecución remota insegura.

## Restricciones de compatibilidad

- Todo YAML estricto 0.5.0 válido seguirá funcionando.
- Se mantienen los runners/resultados y `lambdaforge results SOURCE`.
- La política `existing` continúa disponible; `managed` es explícita por clúster.
- Sólo se resuelven referencias tipadas, nunca strings arbitrarios que parezcan paths.
- No se añaden placement automático, workflows multiclúster, réplica implícita de datos grandes,
  instalación CUDA del sistema ni GUI.

## Plan de implementación

1. Corregir CI/runtime y añadir pruebas aisladas de wheel/procesos.
2. Extender autoría y resolución de datos conservando el runner estricto.
3. Añadir identidad/proveedores de entorno, bundles exactos, bootstrap y sync pequeño.
4. Añadir servicios de resultados, métricas, visualización, artifacts y debug.
5. Delegar la CLI, sincronizar documentación bilingüe y ejecutar verificación de release.
