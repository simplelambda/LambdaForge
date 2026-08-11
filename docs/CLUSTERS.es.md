[English](CLUSTERS.md) | [Español](CLUSTERS.es.md)

# Guía del runtime de clusters

LambdaForge 0.5.1 envía una task o experimento explícito a un cluster con nombre. El YAML científico
no cambia: sólo cambian ubicaciones físicas y política de ejecución.

## 1. Registrar y diagnosticar

```bash
lambdaforge clusters add atlas --host atlas-login --workspace /scratch/me/project \
  --scheduler slurm --environment managed
lambdaforge clusters list
lambdaforge doctor --on atlas
lambdaforge clusters bootstrap atlas
```

El perfil se guarda por defecto en `lambdaforge.clusters.yaml`. OpenSSH conserva claves,
`known_hosts`, agente y ProxyJump; LambdaForge no guarda contraseñas ni desactiva la verificación.
El workspace debe ser escribible sin root.

`managed` construye wheels exactas de LambdaForge y del `pyproject.toml` consumidor más cercano,
las incluye en un bundle por contenido y crea
`WORKSPACE/.lambdaforge/environments/ENVIRONMENT_ID`. Repetir bootstrap reutiliza el entorno. El
estado dirty local se construye tal cual; nunca se hace `git pull main`. `existing` no instala nada
y exige esta versión exacta; `project_module` permite que `doctor` compruebe el proyecto.

Sin Internet, aporte wheels compatibles con la plataforma mediante `wheelhouse` o
`clusters bootstrap --wheelhouse PATH`. Pip usa `--no-index` y falla si faltan dependencias.
LambdaForge nunca instala drivers, CUDA de sistema ni cuDNN: `doctor` informa de lo visible para el
Python seleccionado.

## 2. Datos y recursos portables

```yaml
data_catalog: data-catalog.yaml
environment: local
data: {train: dataset:raw-corpus/train}
resources: {cpus: 8, memory: 32GiB, gpus: 1, gpu_memory: 20GiB, time: 4h}
```

Cada dataset declara `identity`, un `loader` con `path_parameter` y `locations` (`local`, `atlas`,
etc.). El bundle selecciona `data_environment` del destino; referencia e identidad científica no
cambian. Dentro de params se usa `{dataset: raw-corpus, subpath: train}`. Nunca se adivinan strings.
Inputs pequeños declarados pueden viajar; los grandes requieren catálogo o
`lambdaforge data replicate ... --apply` explícito.

## 3. Ejecutar y reconectar

```bash
lambdaforge run experiments/study.yaml --on atlas --dry-run
lambdaforge run experiments/study.yaml --on atlas
lambdaforge status --on atlas --state running --name study
lambdaforge logs JOB --follow
lambdaforge cancel JOB
lambdaforge retry JOB
```

`JobRecord` persiste scheduler ID, identidades, bundle, paths y tiempos. Cerrar el PC no cancela
SLURM; un `status` posterior reconecta. Persistencia no convierte un proceso local en remoto.

## 4. Resultados y límites

`results sync JOB` trae sólo metadata, métricas, manifests, resúmenes y plots pequeños. Use
`plot learning JOB --follow`, `artifact list JOB` y `artifact fetch JOB best-checkpoint` para lo
demás. No hay descarga implícita de checkpoints/datasets.

0.5.1 no soporta placement automático, un workflow repartido entre clusters, coordinador residente,
instalación de drivers ni síntesis de wheels para otra plataforma/CUDA.
