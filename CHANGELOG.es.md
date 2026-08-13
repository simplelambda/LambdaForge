# Historial de cambios

[English](CHANGELOG.md) | [Español](CHANGELOG.es.md)

Este fichero sigue Keep a Changelog y Semantic Versioning. El historial inglés es la fuente de
detalle para releases anteriores; esta edición mantiene el resumen operativo en español.

## [Sin publicar]

## [0.6.0] - 2026-08-13

### Añadido

- Lifecycle genérico `PostRunAction` con checkpoint explícito, política required/optional por acción,
  artifacts compartidos verificados, fingerprint separado y recibos seguros ante interrupción.
- Outputs de validation desacoplados para callbacks, diagnóstico sin segundo forward y métricas
  `val_*` normales utilizables por checkpointing/HPO.
- Plano terminal sin servidor, ProcessScheduler asíncrono con supervisor durable, identidad/grupo
  seguros, heartbeat/uso/logs, timeout de runtime y reconciliación.
- Lifecycle completo de jobs, grupos multiclúster independientes, vistas globales y observación de
  recursos directa/SLURM con admisión cooperativa CPU/RAM/GPU.
- Datasets de primera clase con auto-registro, inventario remoto, stats/profilers explícitos,
  verificación, linaje, CRUD y planes NOOP/REPLICATE/BUILD.
- Descubrimiento de configs/experimentos/tasks por nombre y gestión de storage/entornos con docs
  técnicas bilingües.

### Cambiado

- Training se confirma antes de post-run; cambiar/recuperar análisis reutiliza checkpoint sin
  repetir fit. Sólo rank cero ejecuta acciones y HPO usa confirmation por defecto, nunca pausas o
  cancelaciones.
- OpenSSH reutiliza por defecto una conexión autenticada con caducidad por inactividad; conexión,
  auth, banner, keepalive y timeout de comando son independientes.
- Estado, caché, work y datasets usan raíces separadas. Los entornos se construyen temporalmente,
  se verifican antes de publicar y comparten caché pip; los entornos 0.5 siguen legibles.

### Seguridad

- Señales a procesos verifican PID, creación, grupo y comando. Delete de datasets y GC son preview,
  exact-root y nunca seleccionan resultados, datasets o checkpoints retenidos.

### Corregido

- Compatibilidad real con Torch 2.1+ para dtypes unsigned opcionales de índices y APIs autocast
  antiguas por dispositivo; el chequeo de tipos completo vuelve a pasar.

## [0.5.3] - 2026-08-13

### Añadido

- Resolución remota por Python/arquitectura, driver NVIDIA, compute capability y disponibilidad real
  de wheel oficial; política `pytorch.channel`/`require_cuda` incluida en identidad y bootstrap.

### Corregido

- Bootstrap managed ya no permite que `torch>=2.1` instale cu130 en drivers incompatibles: fija
  primero Torch compatible, restringe pip y valida CUDA antes de crear/reutilizar el entorno.
- Los canales automáticos usan mínimos nativos de driver del toolkit sin presuponer compatibilidad
  menor para CUDA 12/13; capability legacy puede usar el mínimo compatible cu118 si existe wheel y
  supera el probe CUDA.
- Doctor falla si hay GPU visible/requerida pero Torch no inicializa CUDA y muestra driver/error.

### Seguridad

- La selección falla cerrado si no demuestra compatibilidad; no cambia drivers/CUDA del sistema,
  no instala forward-compat ni cae silenciosamente a CPU.

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
