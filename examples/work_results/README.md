# Study → independent model → visualization Work → portable evidence

This executable example uses toy JSON weights, not a scientific training benchmark. Install the
consumer package in your LambdaForge environment, then run from this directory:

```bash
python -m pip install -e . --no-deps
python demo.py
lf
```

`demo.py` runs the native fixed Study, publishes its top scored artifact as a ModelSet, then runs
the independent visualization Work using the human historical source reference. It registers an
interactive self-contained HTML through `html_section`, publishes an independent ScientificReport,
exports all retained evidence, imports it into `offline/runs` and generates another report entirely
from saved evidence. No training occurs during report/import. In the Console, Results →
toy-model-visualization → Visualizations / Open HTML report opens the same output.

For the usual asynchronous CLI, use `lf run train.yaml`, wait for completion, then
`lf run visualize.yaml`. The historical name must be unique; repeating the Study requires selecting
its exact Execution through **Choose historical input…** or `lf results reference EXECUTION
--product toy-selected-models --json`. Launch-time selection can replace `model` without editing YAML.
The export is available through the Console Export action or `lf export toy-model-visualization
--output ./exports`; `lf import PACKAGE --apply` registers it in the current project.

The example creates project-managed evidence and explicit `exports/`, `offline/` and HTML files;
it neither contacts a cluster nor modifies any other scientific project.

## Español

Instala este paquete consumidor con `python -m pip install -e . --no-deps` y ejecuta `python demo.py`
desde esta carpeta. El Study compara modelos de juguete, publica el mejor archivo como ModelSet y un
Work independiente consume ese producto sin repetir el entrenamiento. Genera una visualización HTML,
publica un ScientificReport y exporta/importa la evidencia con las transacciones nativas. El informe
importado se construye solo con archivos verificados, sin ejecutar el antiguo código científico.
En la consola entra en Results → toy-model-visualization → Open HTML report. Los nombres repetidos
requieren seleccionar explícitamente el Execution; el selector histórico fija los hashes por ti.
