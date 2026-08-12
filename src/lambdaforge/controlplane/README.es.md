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

`ClusterCatalog.load()` fusiona usuario, proyecto y explícito en ese orden; `local` siempre existe.
`clusters inspect` muestra fuente/conflictos y los nuevos perfiles van a usuario por defecto.

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login
    auth: {mode: openssh}
    scheduler: slurm
    workspace: /scratch/user/lambdaforge
    python: /shared/env/bin/python
    environment: managed
    project_module: mi_proyecto
    data_environment: atlas
    command_prefix: [apptainer, exec, /images/project.sif]
    resource_mapping: {gpu: {option: gres, value: "gpu:{gpus}"}}
    scheduler_directives: {partition: gpu}
profiles:
  una-gpu:
    cluster: atlas
    resources: {cpus: 8, memory: 32GiB, gpus: 1, time: 4h}
```

OpenSSH es preferido. Contraseña opcional conserva solo descriptor interactivo/`keyring:`/`env:` y
usa Paramiko con host verificado; nunca guardes el valor. La guía completa explica credenciales y
configuración de recursos/comandos/scripts. Comprueba con `doctor --on` o `clusters test`.

## 3. Envío y bundles

```bash
lambdaforge run config.yaml --on atlas --dry-run
lambdaforge run config.yaml --profile una-gpu
```

El bundle cachea YAML, manifiesto, wheels exactas de framework/proyecto y rutas pequeñas. Una grande
usa `DataCatalog`. `managed` crea venv de usuario idempotente por identidad; `existing` sólo
verifica. Offline exige wheelhouse compatible. No clona branches ni instala drivers/CUDA. Véase la
[guía completa](../../../docs/CLUSTERS.es.md).

## 4. Jobs

`JobStore` escribe JSON atómico bajo `$XDG_STATE_HOME/lambdaforge/jobs` o
`~/.local/state/lambdaforge/jobs`. `JobService` ofrece list/get/logs/cancel/retry. Un retry crea otro
ID con `retry_of`; nunca pisa el anterior. Usa `status`, `logs JOB --follow`, `cancel` y `retry`;
`results sync JOB` trae evidencia pequeña y `artifact fetch JOB NAME` un artifact pesado explícito.

## 5. Proveedores

- `LocalTransport`, `SshTransport` OpenSSH y `PasswordSshTransport` Paramiko ejecutan/stagean.
- `LocalScheduler` ejecuta síncronamente.
- `SlurmScheduler` aplica un `SlurmProfile` de mapping/directivas/comandos/script por clúster.
- `CredentialProvider` cubre prompt oculto, keyring del SO y referencia de entorno.
- `ControlPlaneFactory` selecciona defaults y permite inyección para otros centros/tests.

Otro proveedor implementa `Transport`/`Scheduler` y devuelve valores portables; el runner no debe
conocer el proveedor.

## 6. Seguridad y límites

- Remoto requiere `run --on` sin dry-run; replicar datos exige `--apply`.
- OpenSSH no debilita verificación. Contraseña usa RejectPolicy/timeouts y el valor no entra en
  CLI/YAML/registro/bundle/fingerprint.
- Placeholders están allowlisted y comandos son argv; prologue/epilogue son perfil confiable sin
  interpolación de secretos.
- La ubicación multiclúster aparece en planes, pero el runner rehúsa un DAG mixto hasta garantizar
  recuperación durable y transferencia de artefactos.
- La elección de clúster es explícita; no se afirma descubrir capacidad/cola/coste ni placement
  automático. `DataCatalog` resuelve inputs y referencias de experimento tipadas, no strings.
