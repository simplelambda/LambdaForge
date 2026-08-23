# LambdaForge

LambdaForge es un runtime gestionado para investigación científica reproducible. El investigador
escribe Python normal en una clase `lambdaforge.Work`; el framework se ocupa de ejecución local o
remota, recursos, reintentos, métricas, artefactos, datasets, procedencia, resultados y limpieza.

## Instalación

El proyecto científico y LambdaForge son paquetes independientes instalados en el entorno del
proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install lambdaforge==0.12.0
python -m pip install -e .
python -m pip check
```

`lf init mi-estudio` genera un proyecto instalable completo. Se requiere Python 3.10 o posterior.

## Primer Work

```python
from pathlib import Path
import lambdaforge as lf


class Resumir(lf.Work):
    def run(self, fuente: Path, limite: int = 100) -> dict[str, int]:
        lineas = fuente.read_text(encoding="utf-8").splitlines()[:limite]
        self.metrics.log("filas", len(lineas))
        self.outputs.value("resumen", {"filas": len(lineas)})
        return {"filas": len(lineas)}
```

```yaml
name: resumen
run: mi_proyecto.work.Resumir
with:
  fuente: {file: data/entrada.txt}
  limite: 100
resources:
  cpu: 2
  memory: 2GiB
```

La clase debe heredar `Work` y `run()` es su única entrada. La firma y el docstring de Python
definen parámetros, tipos y valores por defecto. Solo `{file: ...}`, `{dataset: NOMBRE@VERSION}` y
`{from: paso.salida}` tienen semántica especial; una cadena normal nunca se interpreta como ruta.

```bash
lf validate experiments/resumen.yaml
lf explain experiments/resumen.yaml
lf run experiments/resumen.yaml --dry-run
lf run experiments/resumen.yaml
lf run experiments/resumen.yaml --on cluster-gpu
```

Los envíos remotos son asíncronos. `--wait-for-submit` espera expresamente a la preparación y al
scheduler.

## Servicios y operación

Durante `run()`, `config`, `inputs`, `resources`, `seed`, `trial`, `source_dir` y `resuming` son
inmutables. `outputs`, `metrics`, `checkpoints`, `cache` y `progress` son servicios gestionados;
`run_dir` pertenece al Attempt y `temp_dir` es efímero. `self.map` ofrece concurrencia acotada,
claves estables, progreso y reanudación JSON dentro de un Work.

`steps` expresa una secuencia y `{parallel: [...]}` un grupo paralelo aislado por procesos. `seeds`
crea Runs independientes; `search` expande variantes y `objective` selecciona la métrica escalar
exacta registrada por el Work, promediándola entre las seeds de cada variante. Los recursos YAML
son la reserva absoluta.

```bash
lf top
lf overview --json
lf show WORK
lf logs WORK --follow
lf retry WORK
lf delete WORK              # vista previa
lf delete WORK --apply
lf datasets list
lf results list
lf clean                    # vista previa de caché reconstruible
```

La creación de datasets se realiza con `self.outputs.dataset(...)`; `lf datasets` inspecciona,
verifica, materializa o elimina versiones inmutables. La guía completa de YAML, HPO, clústeres,
metadata y seguridad está en [el manual](docs/MANUAL.md). Para agentes, [AGENTS.es.md](AGENTS.es.md)
es el contrato compacto que evita recorrer todo el repositorio e inventar APIs.
