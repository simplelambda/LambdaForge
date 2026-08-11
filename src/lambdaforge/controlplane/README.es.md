# Plano de control de LambdaForge

[Guía principal](../../../README.es.md) · [English](README.md)

## 0. Contenidos

- [1. Modelo mental](#1-modelo-mental)
- [2. Catálogo de clústeres](#2-catálogo-de-clústeres)
- [3. Envío y bundles](#3-envío-y-bundles)
- [4. Jobs](#4-jobs)
- [5. Proveedores](#5-proveedores)
- [6. Seguridad y límites](#6-seguridad-y-límites)

## 1. Modelo mental

El plano de control es un servicio de aplicación local. Materializa YAML, prepara un
`ExecutionBundle` pequeño, envía el comando LambdaForge normal mediante `Transport` y `Scheduler`,
y conserva un `JobRecord`. No sustituye runners ni necesita servidor. Los resultados científicos
siguen en `result.json`; el ciclo del scheduler queda en el almacén de jobs.

## 2. Catálogo de clústeres

`ClusterCatalog.load()` busca ruta explícita, `LAMBDAFORGE_CLUSTERS`, fichero del proyecto y
configuración del usuario. `local` siempre existe. Un perfil SSH exige workspace absoluto.

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login
    scheduler: slurm
    workspace: /scratch/user/lambdaforge
    python: /shared/env/bin/python
    data_environment: atlas
    command_prefix: [apptainer, exec, /images/project.sif]
    scheduler_options: {partition: gpu}
profiles:
  una-gpu:
    cluster: atlas
    resources: {cpus: 8, memory: 32GiB, gpus: 1, time: 4h}
```

No guardes credenciales aquí. Comprueba con `doctor --on atlas` o `clusters test atlas`.

## 3. Envío y bundles

```bash
lambdaforge run config.yaml --on atlas --dry-run
lambdaforge run config.yaml --profile una-gpu
```

El bundle cachea YAML estricto, manifiesto y rutas pequeñas (máximo 10 MiB por defecto). Una ruta
grande se rechaza y debe usar `DataCatalog`. El entorno remoto debe contener las versiones fijadas
del framework y proyecto. Bootstrap sólo crea workspace y verifica imports.

## 4. Jobs

`JobStore` escribe JSON atómico bajo `$XDG_STATE_HOME/lambdaforge/jobs` o
`~/.local/state/lambdaforge/jobs`. `JobService` ofrece list/get/logs/cancel/retry. Un retry crea otro
ID con `retry_of`; nunca pisa el job ni el intento científico anterior.

## 5. Proveedores

- `LocalTransport` y `SshTransport` ejecutan y stagean.
- `LocalScheduler` ejecuta síncronamente.
- `SlurmScheduler` genera script, envía, reconecta, lee logs y cancela.
- `ControlPlaneFactory` selecciona defaults y permite inyección para otros centros/tests.

Otro proveedor implementa `Transport`/`Scheduler` y devuelve valores portables; el runner no debe
conocer el proveedor.

## 6. Seguridad y límites

- Remoto requiere `run --on` sin dry-run; replicar datos exige `--apply`.
- OpenSSH conserva verificación y credenciales; LambdaForge no las desactiva.
- Opciones/argv se validan y no se usa shell local.
- La ubicación multiclúster aparece en planes, pero el runner rehúsa un DAG mixto hasta garantizar
  recuperación durable y transferencia de artefactos.
- La elección de clúster es explícita; no se afirma descubrir capacidad/cola/coste ni placement
  automático. `DataCatalog` resuelve inputs de tareas, no rutas ocultas en objetos de experimento.
