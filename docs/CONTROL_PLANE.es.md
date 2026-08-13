# Plano de control terminal de LambdaForge 0.6

Español | [English](CONTROL_PLANE.md) | [Guía raíz](../README.es.md)

## 1. Modelo mental

LambdaForge no necesita un servidor. Cada invocación de CLI es un controlador breve que lee estado
local pequeño, consulta sólo los clústeres solicitados y termina. El trabajo largo pertenece a
SLURM o a un `ProcessSupervisor` separado por job en la máquina de ejecución. Cerrar el portátil o
perder SSH no mata un job cuya recepción ya se confirmó.

```text
YAML -> configuración materializada -> caché de bundle inmutable
                                          |
JobRecord local <- identidad scheduler <- submit
                                          |
                            job/work + estado + logs durables
```

`JobRecord` permite reconectar; no es el resultado científico ni sustituye el estado remoto.
`result.json`, métricas y checkpoints son la evidencia. `jobs reconcile` reconstruye el índice.

## 2. Primer uso

```bash
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/me/lambdaforge \
  --scheduler local --environment managed --cache-root /scratch/me/lf-cache \
  --run-root /scratch/me/lf-jobs --dataset-root /project/data
lambdaforge clusters bootstrap atlas
lambdaforge doctor --on atlas
lambdaforge datasets list
lambdaforge experiments list
lambdaforge experiments run baseline --on atlas --dry-run
lambdaforge experiments run baseline --on atlas
lambdaforge jobs list --all
```

En 0.6 `scheduler: local` es el scheduler de procesos durable y asíncrono; ya no bloquea la CLI.
`scheduler: slurm` conserva a SLURM como autoridad. Ambos usan `JobService`.

## 3. Reutilización SSH y timeouts

OpenSSH es la opción recomendada. LambdaForge inicia un cliente `ssh` pequeño por operación, pero
por defecto todos reutilizan un socket maestro autenticado durante 60 segundos desde el último uso.
También se reutiliza entre invocaciones sucesivas de la CLI: no se repiten TCP, intercambio de
claves y autenticación. El directorio XDG del socket es privado (`0700`). Se respetan alias, agente,
claves, `known_hosts` y `ProxyJump`.

```yaml
connection:
  connect_timeout: 15s
  auth_timeout: 30s
  banner_timeout: 30s
  keepalive: 30s
  multiplex: true
  persist: 2m
  command_timeout: null
```

Los límites de conexión/autenticación no son límites del comando. Doctor, probes e inventario usan
límites explícitos cortos; los comandos científicos no tienen límite de transporte implícito.
`resources.time` limita el runtime: el supervisor marca `timeout` y termina el grupo verificado.
Paramiko reutiliza una conexión dentro de una invocación, no entre procesos CLI; para uso frecuente
conviene OpenSSH. Desactiva `multiplex` sólo si lo exige la política del centro.

## 4. Layout y migración

```yaml
storage:
  state_root: /home/me/.lambdaforge/state
  cache_root: /scratch/me/lambdaforge/cache
  run_root: /scratch/me/lambdaforge/jobs
  dataset_root: /project/datasets
  cache_max_size: 50GiB
  cache_max_age: 30d
```

Bundles/entornos son caché; estado/logs/work por job son mutables; los datasets no se fuerzan a
`.lambdaforge`. Los entornos y punteros 0.5 siguen siendo legibles. GC nunca considera caché los
resultados, datasets o checkpoints retenidos.

## 5. Vista global y errores honestos

```bash
lambdaforge status
lambdaforge overview --json
lambdaforge resources --all
lambdaforge top --follow
lambdaforge storage status --all
```

Las consultas son paralelas y acotadas. Un clúster inaccesible aparece offline, no como job fallido;
si existe se conserva la última observación. Un error de proveedor produce `unknown`, estado previo
y causa. Nunca se buscan o señalizan PIDs remotos ajenos.

## 6. Límites explícitos

0.6 no añade servidor central, placement automático, HPO multiclúster coordinado ni workflow
multiclúster durable. Repetir `--on` crea réplicas independientes agrupadas; HPO exige
`--independent-hpo` para hacer explícito que no comparte estado del optimizador.
