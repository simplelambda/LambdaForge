[English](CLUSTERS.md) | [Español](CLUSTERS.es.md)

# Clústeres seguros y personalización del scheduler

## Índice

1. [Modelo mental](#1-modelo-mental)
2. [Ámbitos y precedencia](#2-ámbitos-y-precedencia)
3. [Autenticación SSH](#3-autenticación-ssh)
4. [Dialecto SLURM por clúster](#4-dialecto-slurm-por-clúster)
5. [Registrar, inspeccionar y diagnosticar](#5-registrar-inspeccionar-y-diagnosticar)
6. [Enviar y reconectar](#6-enviar-y-reconectar)
7. [Seguridad y límites](#7-seguridad-y-límites)

## 1. Modelo mental

LambdaForge sigue siendo un plano de control local: materializa la misma tarea/experimento, crea un
bundle acotado por contenido, elige un `ClusterProfile`, llega mediante un `Transport` y pide a su
`Scheduler` ejecutar el `python -m lambdaforge run ...` normal. Acceso/recursos forman identidad de
ejecución; contraseñas y paths físicos nunca forman identidad científica. OpenSSH y SLURM estándar
siguen siendo los defaults sin configuración extra.

## 2. Ámbitos y precedencia

Los catálogos se fusionan por nombre en este orden (el último gana):

1. usuario: `$XDG_CONFIG_HOME/lambdaforge/clusters.yaml`, normalmente
   `~/.config/lambdaforge/clusters.yaml`;
2. proyecto: el `PROJECT/lambdaforge.clusters.yaml` más cercano (existente o raíz del
   `pyproject.toml` más cercano);
3. explícito: `--clusters-file PATH`/`--clusters PATH`, o `LAMBDAFORGE_CLUSTERS` sin flag.

`clusters add` guarda por defecto en usuario para no commitear hosts personales por accidente. Usa
`--scope project` para un descriptor compartible revisado. Un archivo explícito sobrescribe pero no
oculta otros perfiles. `clusters inspect NAME` muestra fuente ganadora y fuentes tapadas;
`clusters export NAME --output FILE` exporta el descriptor, como mucho su referencia de credencial,
nunca el valor.

## 3. Autenticación SSH

### 3.1 OpenSSH (recomendado y predeterminado)

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login                 # alias de ~/.ssh/config
    user: research                    # opcional
    port: 22
    auth: {mode: openssh}
    ssh_options: [-o, BatchMode=yes]
    scheduler: slurm
    workspace: /scratch/research/lambdaforge
```

`SshTransport` llama a `ssh`/`scp` mediante argv. La configuración nativa conserva claves, agente,
`known_hosts`, certificados y ProxyJump; LambdaForge nunca desactiva la verificación. Es la ruta
preferida porque reutiliza exactamente la política que ya funciona con `ssh atlas-login`.

### 3.2 Contraseña (opcional)

```bash
python -m pip install 'lambdaforge[cluster-password]'

# Prompt oculto en cada proceso, sin persistencia.
lambdaforge clusters add legacy --host login.example.org --user me \
  --workspace /scratch/me/lambdaforge --auth password

# Prompt oculto y almacenamiento en el keyring del sistema; no existe --password.
lambdaforge clusters add legacy --host login.example.org --user me \
  --workspace /scratch/me/lambdaforge --auth password --store-password
lambdaforge clusters credentials set legacy
lambdaforge clusters credentials delete legacy
```

Paramiko usa SFTP/SSH con `RejectPolicy`, `known_hosts` del sistema, timeouts acotados y sin fallback
a claves/agente. Se pueden fijar `known_hosts: /fichero/revisado` y `ssh_timeout: 20`. Paramiko se
conecta a un host concreto; los aliases/ProxyJump pertenecen a OpenSSH, así que usa el default en
esos casos.

Solo se persisten estas referencias, nunca su valor:

```yaml
auth: {mode: password}
auth: {mode: password, credential: keyring:cluster/legacy/me@login.example.org}
auth: {mode: password, credential: env:LEGACY_SSH_PASSWORD}
```

`keyring:` delega en el almacén del sistema mediante el paquete opcional maduro `keyring`. Si no
está disponible se explica cómo instalarlo y se puede elegir prompt o `env:`. `env:` lee la variable
al conectar; es útil para un secret efímero de CI, pero el entorno del proceso/CI amplía el límite
de confianza, por lo que en una estación conviene OpenSSH o keyring.

## 4. Dialecto SLURM por clúster

`ResourceRequest` permanece portable. Un único `SlurmResourceMapping` lo traduce; una directiva
estática no puede duplicar una opción de recurso. Los defaults son `ntasks`, `cpus-per-task`, `mem`,
`gpus` y `time`.

```yaml
clusters:
  atlas:
    transport: ssh
    host: atlas-login
    scheduler: slurm
    workspace: /scratch/me/lambdaforge
    resource_mapping:
      processes: {option: ntasks, value: "{processes}"}
      cpu: {option: cpus-per-task, value: "{cpu_per_process}"}
      memory: {option: mem, value: "{memory_gib}G"}
      gpu: {option: gres, value: "gpu:a100:{gpus}"}
      time: {option: time, value: "{minutes}"}
    scheduler_directives:
      partition: accelerated
      account: project123
      exclusive: true
      constraint: [nvlink, ssd]
    scheduler_commands:
      submit:
        command: site-sbatch
        args: [--parsable, "{script}"]
        job_id_pattern: "^(\\d+)(?:;.*)?$"
      queue: {command: site-squeue, args: [-h, -j, "{job_id}", -o, "%T"]}
      accounting: {command: site-sacct, args: [-n, -X, -j, "{job_id}", -o, State]}
      cancel: {command: site-scancel, args: ["{job_id}"]}
    job_script:
      shell: /bin/bash
      prologue: [module load cuda/13.0]
      epilogue: [echo job-finished]
```

GPU puede usar `gpus`, GRES genérico `gpu:{gpus}` o GRES tipado como arriba. Una regla `omit` indica
que un wrapper del centro aporta ese recurso; dry-run y doctor avisan con fuerza porque SLURM no
aplicará directamente la petición portable.

Las plantillas solo admiten campos numéricos documentados (`cpu_cores`, `cpu_per_process`,
`processes`, `memory_bytes/mib/gib`, `gpus`, `seconds/minutes/hours`). Los comandos solo admiten
`{job_id}`, `{script}` y `{work_dir}` donde corresponda; producen argv, nunca `shell=True`/`eval`.
`job_id_pattern` debe capturar el ID. Prologue/epilogue son líneas shell de perfil deliberadamente
confiables, sin interpolación de secretos, y deben revisarse como código. `scheduler_options` antiguo
sigue funcionando como directivas estáticas.

## 5. Registrar, inspeccionar y diagnosticar

```bash
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/me/lambdaforge \
  --scheduler slurm --environment managed
lambdaforge clusters inspect atlas
lambdaforge doctor --on atlas
lambdaforge clusters bootstrap atlas
```

`doctor` es read-only: comprueba transporte/autenticación, workspace, Python, imports exactos,
PyTorch/CUDA, cada ejecutable configurado, traducción de recursos y partición; nunca envía un job.
`managed` construye wheels locales exactas y un venv de usuario; `existing` no instala. Offline exige
wheelhouse compatible. Ningún modo instala drivers, CUDA del sistema ni cuDNN.

## 6. Enviar y reconectar

```bash
lambdaforge run experiment.yaml --on atlas --dry-run
lambdaforge run experiment.yaml --on atlas --cpus 8 --memory 32GiB \
  --resource-gpus 1 --time 4h
lambdaforge status --on atlas --state running
lambdaforge logs JOB --follow
lambdaforge cancel JOB
```

Dry-run devuelve clúster, recursos portables, script, directivas, avisos y argv exacto sin contactar
al scheduler. Los jobs reales conservan IDs local/remoto, bundle, entorno, identidades y paths para
reconectar. El sync pequeño y el fetch pesado siguen siendo explícitos.

## 7. Seguridad y límites

Ninguna contraseña se acepta en argv, serializa, registra, empaqueta, fingerprinta ni coloca en YAML.
Los secretos conocidos en memoria se redactan de errores. Transportes/proveedores plugin deben
aplicar el mismo contrato. La personalización del scheduler es configuración confiable, no input de
experimento no fiable.

No hay colocación automática, daemon, recuperación distribuida de workflows, copia implícita de
datos/checkpoints, bastion separado de OpenSSH, instalación de drivers ni síntesis de wheels.
