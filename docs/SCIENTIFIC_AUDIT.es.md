# Auditoría de corrección científica

[English](SCIENTIFIC_AUDIT.md)

Base: `c8f2d2e`, LambdaForge 0.17.0. Se cierran las correcciones de orden de evidencia y coherencia
de conclusiones descritas aquí, **no toda la auditoría de ingeniería solicitada**. No se modifica
WISDOM, ningún cluster real, manifests científicos históricos ni GitHub CI.

## Defectos confirmados y correcciones

| Clasificación | Autoridad / comportamiento anterior | Corrección y evidencia |
| --- | --- | --- |
| CONFIRMED_BUG | `SequentialSweep.evaluate` ordenaba la intersección por número de seed; un bloque nuevo podía reordenar observaciones ponderadas anteriores. | Exige `seed_order` persistido, consume exclusivamente su prefijo completo y registra las seeds utilizadas. Pruebas con ProjectSeedStream real, llegadas invertidas y celdas iniciales ausentes. |
| CONFIRMED_BUG | `runner._execute_fixed_evidence_group` reiniciaba `committed_lookahead=None` al recuperar. | Restaura el compromiso de `sweep-blocks.json` mediante validación compartida de solo lectura; impide un segundo bloque especulativo. La integración conserva Runs, manifest original y coste físico tras interrupción. |
| CONFIRMED_BUG | La telemetría de bloques mostraba mínimo/máximo completo aunque hubiera huecos. | Publica únicamente el prefijo completo continuo. El inventario inicial se persiste antes de lanzar, incluso sin ningún resultado terminal. |
| CONFIRMED_BUG | `WorkRunner._summary` contaba solo obligaciones iniciales y podía declarar éxito con una celda anticipada comprometida pendiente. | Recovery y finalización comparten `recovery.fixed_requirements`: bloques creados siguen siendo obligatorios sin reescribir el diseño. Integración conserva tres Runs válidas y muestra la cuarta pendiente, sin reiniciar ni esconder deuda. |
| CONFIRMED_BUG | ScientificDesign convertía predicciones sin soporte a `UNRESOLVED` conservando la estabilidad de otra hipótesis. | La conclusión descriptiva sin soporte tiene cero realizaciones propias; predicción, estabilidad predictiva y motivo permanecen separados. Cobertura no se llama confianza. |
| CONFIRMED_BUG | `_exact_conclusion` y análisis de sweep atribuían `1-probabilidad_modal` al evento no resuelto; una distribución vacía tenía estabilidad uno. | Usan la masa real del token no resuelto o cero sin realizaciones. Un líder numérico débil no transforma equivalencia observada en un producto de probabilidades de otro evento. |
| CONFIRMED_BUG | Study Analysis interpretaba cualquier `stop=true`, incluido `INCOMPLETE`, como resolución formal. | Lector compartido exige una decisión científica; una parada operacional sigue sin resolver. Ausencia de procedimiento secuencial es `None`, no aprobación formal. |

El lector remoto de retry integra el mismo validador puro desde su fuente autoritativa, como los
lectores existentes de recovery/seeds. Una regresión lo ejecuta con Python aislado (`-I -S`),
sin LambdaForge instalado: no equivale a aceptación SSH real.

`ScientificQuestionState` separa resolución descriptiva y formal. Su accessor histórico `resolved`
conserva aplicabilidad descriptiva exploratoria cuando no existe procedimiento formal; no garantiza
cobertura formal.

## Contrato estadístico

La fórmula sigue siendo la secuencia empirical-Bernstein plug-in predecible de
[Waudby-Smith y Ramdas, teorema 2 y ecuación 15](https://rss.org.uk/RSS/media/File-library/Events/Discussion%20meetings/Estimating-means-of-bounded-random-variables-by-betting.pdf).
Los pesos dependen del pasado; las diferencias pareadas están acotadas y se supone media
condicional común. El generador determinista de seeds no demuestra esa hipótesis científica.
Bonferroni de familia primaria y margen declarado no cambian. Monte Carlo comprueba regresiones,
no demuestra el teorema ni calibración de modelos reales.

## Compatibilidad y propiedad

- Inferencia runtime `paired-pm-eb-cs-v2`, orden `committed-acquisition-prefix-v1` y `evidence_seeds`
  exactas. Registros históricos v1 siguen consultables: no se reescriben ni se llaman v2.
- `StudyDesign.replication_policy` histórico permanece inmutable; el registro de evaluación
  determina su versión runtime. No cambian IDs científicos, Runs ni Attempts.
- Inventario de bloques sigue en versión 1, que ya guarda ordinales y compromiso. Recuperación
  rechaza duplicados, discontinuidades, compromisos contradictorios o metadatos de seed erróneos.
  Coordenadas iniciales proceden de `seed_source.resolved`, no del número de seed. Ausencia de
  evidencia autoritativa falla antes de escribir. Previews siguen siendo de solo lectura.
- Snapshot de preguntas científicas versión 4; semántica de conclusiones de parámetro versión 3.
  `confidence` significa estabilidad descriptiva del evento mostrado, no probabilidad de verdad
  ni cobertura formal. Análisis históricos no se recalculan silenciosamente.
- Seeds finitas mantienen orden declarado; sweeps fijos no adquieren pruning adaptativo.
  Retry/continuación mantienen Run lógica y crean Attempts. Evidencia importada sigue inerte.

## Verificación reproducible

```bash
pytest -q tests/hpo tests/work tests/analysis tests/products tests/tui
python benchmarks/sequential_sweep_audit.py
python benchmarks/hpo_scientific_design.py
python benchmarks/resource_scheduling.py
ruff check .
mypy src/lambdaforge
```

El benchmark nuevo inspecciona cada prefijo de 200 trayectorias, hasta 96 bloques, con tres
comparaciones frente a referencia bajo nulos degenerado, de varianza baja y alta. El benchmark
de recursos es una regresión sintética existente: no valida GPUs reales ni una política nueva.
El benchmark científico existente solo cubre tres paisajes pequeños, no los ocho escenarios
requeridos para aceptación científica integral.

Resultados locales de este cambio:

- Suites amplias retenidas HPO/Work/Analysis/productos/Console: **718 aprobados, 628,84 s**, antes
  de las últimas adiciones de deuda comprometida/lector aislado; después, suites backend afectadas
  finales: **144 aprobados, 43,70 s**. No se afirma haber repetido pytest de todo el repositorio.
- Ruff, mypy (**581 ficheros fuente**) y comprobación de espacios aprobados.
- Wheel instalada fuera del checkout: smoke de imports/assets, CLI help/scaffold/validación,
  sweep automático CPU con procesos spawned y cuatro Runs, más preview de recuperación aprobado.
  Dependencias reutilizadas en prefijo temporal; no se reemplaza entorno de proyectos del usuario.
- Calibración: **0/200** violaciones de familia en cada uno de tres escenarios nulos, hasta 96
  inspecciones. Benchmark de tres paisajes conserva regret medio aproximadamente **3,70e-17**
  en ambas políticas. Benchmark sintético existente de recursos: makespan adaptativo **58 s**,
  fijo **109 s**, ambos sin OOM. Son regresiones del scheduler existente, no mejora nueva.
- Hardware CUDA, SSH/SLURM real y nuevos experimentos de recursos con trazas **no ejecutados**.

## Fronteras pendientes explícitas

| Clasificación | Pendiente | Aceptación necesaria |
| --- | --- | --- |
| CORRECTNESS_RISK | Fault injection integral de lifecycle/continuación y costes físicos adicionales en todos los lectores. | Checkpoints/publicación interrumpidos, resultados tardíos, fidelidades heterogéneas y continuación sin información por todas las rutas nativas. |
| DEFERRED_FEATURE | Leases CPU desiguales aprendidos y calibración ARI adicional. | Experimentos reales de demanda/progreso y replay de trazas que demuestren mejora manteniendo grants y afinidades. No se reemplaza el modelo actual especulativamente. |
| DEFERRED_FEATURE | Valoración científica revisada de preguntas correlacionadas y costes. | Comparaciones en ocho paisajes con soporte, calibración, coste y regret; se conservan autoridades actuales. |
| DEFERRED_FEATURE | Extracción más amplia de WorkRunner, sustitución de datasets por contrato y espera de productos. | Caracterización e integración de propiedad/identidad/recovery; política explícita de certificados, sin transitividad inferida. |
| DEFERRED_FEATURE | Recovery/adopción Fleet públicos, transporte reanudable y export distribuido. | Cadena completa lease/fencing/dispatch/acknowledgment/recovery/export validada. Se mantienen límites distribuidos actuales. |

Las suites retenidas cubren productos, certificados, import/export, replay y clasificación de
fallos; son cobertura de regresión, no aceptación de sus funcionalidades pendientes.
