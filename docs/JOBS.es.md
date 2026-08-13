# Jobs durables y scheduler de procesos

Español | [English](JOBS.md) | [Guía raíz](../README.es.md)

## 1. Ciclo de vida y ficheros

Los estados son `created`, `staging`, `queued`, `running`, `paused`, `succeeded`, `failed`,
`cancelled`, `timeout` y `unknown`. En un host directo cada job posee:

```text
RUN_ROOT/job-ID/
  request.json
  state.json
  heartbeat
  stdout.log
  stderr.log
  usage.jsonl
  work/
  control/
```

Supervisor e hijo registran PID, grupo, instante de creación y hash del comando. Antes de
cancelar/pausar/reanudar se verifica toda la identidad; reutilizar un PID no permite señalizar otro
proceso. La cancelación termina descendientes y grupo y escala tras la gracia. El inventario sólo
acepta directorios cuyos IDs coinciden en request y estado.

## 2. Comandos

```bash
lambdaforge jobs list [--on atlas] [--state running] [--name baseline] [--json]
lambdaforge jobs show JOB
lambdaforge jobs logs JOB [--tail 200] [--follow]
lambdaforge jobs pause JOB
lambdaforge jobs resume JOB
lambdaforge jobs cancel JOB
lambdaforge jobs retry JOB [--dry-run]
lambdaforge jobs delete JOB
lambdaforge jobs reconcile --on atlas
lambdaforge jobs reconcile --all
lambdaforge jobs group list
lambdaforge jobs group show GROUP
```

`delete` borra sólo metadatos locales terminales, nunca work o resultados. `retry` crea otro ID y
registra `retry_of`. `reconcile` reconstruye el índice desde el inventario LambdaForge remoto.

## 3. Pausa, cola y recursos

Pausa/reanudación son capabilities. ProcessScheduler usa `SIGSTOP`/`SIGCONT`; SLURM sólo lo ofrece
si el perfil define comandos del centro. Pausar conserva RAM, VRAM, leases y workspace.

Los hosts directos usan leases cooperativos con lock. Un job queda `queued` hasta admitir CPU/RAM y
GPU. Se aplica afinidad CPU y límites de threads cuando es posible; las GPUs seleccionadas llegan
por `CUDA_VISIBLE_DEVICES`. Los leases no son aislamiento físico y evitan GPUs con procesos de
cómputo externos observables. En SLURM, SLURM administra capacidad y aislamiento.

`resources.time` es un límite de runtime del supervisor, no de SSH, y termina en `timeout`. El
heartbeat no depende de stdout. Si se mata a la fuerza el supervisor, 0.6 no promete controlar un
proceso arbitrario huérfano: el estado puede quedar stale/unknown y requerir revisión del operador.

## 4. Grupos multiclúster

```bash
lambdaforge experiments run baseline --on atlas --on gpu-lab
lambdaforge jobs group list
```

Son jobs independientes unidos por `group_id`, no entrenamiento distribuido ni HPO compartido. HPO
requiere `--independent-hpo`. Si una entrega parcial falla, los jobs ya aceptados siguen visibles y
cancelables; no se ocultan con un rollback ficticio.
