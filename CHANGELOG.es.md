# Historial de cambios

[English](CHANGELOG.md) | [Español](CHANGELOG.es.md)

Este fichero sigue Keep a Changelog y Semantic Versioning. El historial inglés es la fuente de
detalle para releases anteriores; esta edición mantiene el resumen operativo en español.

## [Sin publicar]

## [0.5.2] - 2026-08-12

### Añadido

- Catálogos fusionados usuario/proyecto/explícito con inspección de fuente/conflictos y export seguro.
- Contraseña SSH opcional por prompt oculto, keyring o referencia de entorno, con Paramiko,
  verificación estricta de host, SFTP y timeouts.
- `SlurmProfile` por clúster para mapping CPU/memoria/GPU/tiempo, flags/repeticiones, comandos argv,
  parseo de ID, hooks confiables, preview detallado y doctor ampliado.

### Seguridad

- El valor secreto no entra en flags, YAML, jobs, bundles, fingerprints ni logs; errores conocidos se
  redactan. OpenSSH sigue recomendado y `scheduler_options`/catálogos antiguos siguen compatibles.

## [0.5.1] - 2026-08-11

### Añadido

- Semántica real de preprocesado secuencial/I-O threads/CPU spawn/GPU segura y debug de N registros.
- Training conciso, recursos portables y datasets lógicos directos/anidados en experimentos con
  fingerprint independiente del mount.
- Entornos managed exactos por wheels locales, venv de usuario idempotente, wheelhouse offline,
  bootstrap/doctor y jobs reconectables con filtros/follow.
- Sync remoto ligero y fetch explícito, `ResultService`, `MetricSeries`, comparación/export,
  `PlotSpec` y plots learning/seeds/sweeps/HPO/recursos reproducibles.
- Toolkit de artifacts con NumPy/tablas seguro y acotado, validadores, geometría explícita y plugins;
  inspección de DatasetArtifact y documentación humana/agentes EN/ES.

### Cambiado

- Los bundles remotos llevan wheels exactas de framework/proyecto en vez de depender de otro checkout.
- La CLI analítica delega en servicios reutilizables; `results SOURCE` sigue compatible.
- Índice raíz jerárquico, nuevos extras `viz`, `graph` y `viz3d`, y packaging de todas las guías.
- Workers/workload de preprocesado ya no cambian identidad científica; normalización y dirección de
  métricas son explícitas en plots/comparaciones.

### Corregido

- `fsync` Windows en referencias de arquitectura, refill del scheduler dinámico y escritura atómica
  de métricas densas.
- El smoke CI instala la wheel con dependencias y la importa fuera del source checkout.

### Seguridad

- NumPy desactiva pickle; previews/estadísticas y sync remoto están acotados; geometría/fetch son
  explícitos y con containment. SSH conserva política estándar y LambdaForge no instala CUDA.

## Resumen de releases anteriores

- **0.5.0 (2026-08-11):** autoría concisa, identidad lógica de datos/código, lifecycle explícito,
  recursos portables, plano de control local/SSH+SLURM, jobs persistentes y replicación explícita.
- **0.4.1 (2026-08-10):** hardening del HPO mixto multi-fidelidad, curvas bayesianas, KG común,
  memoria feature-aware y tests CUDA/sintéticos.
- **0.4.0 (2026-08-09):** optimizador adaptativo por acciones, resume de fidelidad, racing de seeds,
  admisión VRAM y estado/eventos atómicos.
- **0.3.0 (2026-08-09):** tasks genéricas, preprocesado, workflows/configuración, operaciones,
  backends/stores/registry/observabilidad y componentes científicos.
- **0.2.0 (2026-08-08):** framework instalable con experimentos YAML, training, métricas/modelos,
  procesos, resultados, plugins, retención y documentación para agentes.
- **0.1.0:** base inicial de modelos, pérdidas, métricas y ejecución reproducible.

Para el desglose exacto por categoría de esas versiones, consulta [el changelog inglés](CHANGELOG.md).
