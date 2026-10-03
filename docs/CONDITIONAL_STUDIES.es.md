# Estudios científicos condicionales

[English](CONDITIONAL_STUDIES.md) · [HTML de investigación](RESEARCH_ANALYSIS.es.md)

## Índice

1. [Una pregunta, un Study](#1-una-pregunta-un-study)
2. [Gramática de activación](#2-gramática-de-activación)
3. [Validar y entender el plan](#3-validar-y-entender-el-plan)
4. [Tiempo y capacidad del destino](#4-tiempo-y-capacidad-del-destino)
5. [Identidad y evidencia existente](#5-identidad-y-evidencia-existente)

## 1. Una pregunta, un Study

Usa un único `sweep.space` para comparar alternativas que tienen distintos parámetros aplicables.
Todas las celdas pertenecen al mismo Study y comparten las seeds declaradas. Usa `steps` para un
flujo real de trabajo, no para separar artificialmente ramas comparables: siguen siendo
invocaciones independientes, no estudios fusionados.

```yaml
name: optimizer-comparison
run: my_project.Compare
seeds: [4, 7, 32, 54]
sweep:
  reference: {optimizer: baseline}
  space:
    momentum:
      values: [0.8, 0.9]
      when: {optimizer: {in: [sgd, accelerated]}}
    optimizer: [baseline, sgd, accelerated]
resources: {gpu: 2, time: 24h}
execution: {max_parallel: 4, max_time: 12h}
```

Son cinco candidatos y veinte Runs obligatorios, no doce candidatos ni un Study por optimizador.
`baseline` no recibe `momentum`; se aplica el valor por defecto de la firma Python. La referencia
debe seleccionar exactamente un candidato activo. Cero coincidencias o varias son errores.

## 2. Gramática de activación

Un descriptor admite `when` opcional. Varios padres se combinan con AND:

```yaml
when:
  optimizer: {in: [sgd, accelerated]}
  use_momentum: true
```

`{parent: valor}` conserva la igualdad antigua; `{parent: {eq: valor}}` significa lo mismo. La
pertenencia es explícita: `{parent: {in: [valor1, valor2]}}`. La lista debe contener escalares JSON
únicos, no estar vacía y pertenecer al dominio finito del padre. Se rechazan listas sin operador,
operadores desconocidos, padres inexistentes, ciclos y valores ajenos al dominio. No hay
expresiones, `or`, `not` ni lenguaje recursivo. La igualdad antigua sobre rangos numéricos sigue
funcionando; `in` requiere un dominio finito.

Los hijos pueden escribirse antes que sus padres. `ActivationCondition` y `ParameterSpace`
comparten validación y ordenación de dependencias entre sweeps, generación aleatoria/Sobol/
adaptativa, codificación y análisis. Las claves inactivas se omiten: no son `null` ni multiplican
el diseño para deduplicarlo después. Se rechazan valores repetidos del sweep. El dominio mantiene
el orden declarado; ordenar de otra forma los miembros de `in` no cambia la identidad científica.
La cobertura condicional usa valores observados activos, no una categoría inactiva inventada.

## 3. Validar y entender el plan

```bash
lf validate comparison.yaml
lf explain comparison.yaml
lf run comparison.yaml --on CLUSTER --dry-run --json
```

Validación y explicación incluyen `preflight` acotado: candidatos, seeds compartidas, Runs
obligatorios, conteos de la primera rama categórica independiente, referencia, GPUs solicitadas,
límite paralelo, presupuesto del Study y tiempo del scheduler. La salida humana no enumera cientos
de identidades. `--json` conserva el plan completo para automatización. La validación local indica
que no se ha comprobado la capacidad del destino; un plan remoto de solo lectura también puede
necesitar un entorno/runtime ya preparado. El resumen de ramas no es una API de familias de dominio.
En búsqueda adaptativa, los conteos describen la ventana de propuestas actual (limitada por el
techo de candidatos), no prometen ejecutar todas las propuestas. La replicación se vuelve
obligatoria al proponer el candidato. En un sweep fijo, los conteos cubren todo el diseño declarado.

Solo `resources` raíz se hereda por los pasos salvo sobrescritura. En una composición con `steps`
se rechazan `with`, `seeds`, `replicates`, `search`, `sweep`, `execution`, `objective` y `analysis`
raíz: deben declararse en el paso correspondiente. Nunca se ignora silenciosamente una política.
Preflight comprueba herencia y firmas sin construir un Work ni ejecutar ciencia; no puede detectar
errores semánticos arbitrarios dentro del `run()` del proyecto consumidor.

## 4. Tiempo y capacidad del destino

`resources.time` es el techo de tiempo del scheduler para la unidad planificable. Los niveles
secuenciales suman; un nivel paralelo aporta el máximo de sus miembros. Seis pasos secuenciales
de 168 h solicitan 1008 h (42 días); tres de 24 h, 72 h. Es un techo, no una predicción del tiempo
de entrenamiento. `execution.max_time` es el presupuesto lógico de despacho del Study; se informa
por separado y no sustituye al límite del scheduler.

En un host directo, un inventario UUID vivo y correcto junto con la restricción heredada de
visibilidad, cuando sea inequívoca, permite rechazar una solicitud superior a la capacidad antes
de preparar el bundle, también en dry-run. Ocupación no es capacidad. Una sonda fallida/ambigua
queda como desconocida, nunca inventa un número. El nodo de login de un scheduler no es su
inventario de asignaciones. Los launchers del centro conservan la comprobación existente en
runtime: un Study adaptativo puede reducirse a menos tokens otorgados, nunca ampliar el grant ni
reservar otras GPUs durante preflight. La validación local no contacta destinos remotos.

## 5. Identidad y evidencia existente

La igualdad normalizada mantiene serialización e identidad antiguas. `in` amplía el schema actual
sin nueva versión YAML ni runner de compatibilidad. Los predicados permanecen en diseño e
inicialización del Study y viajan por análisis, exportación y retry. Los registros históricos de
igualdad se leen, no se reescriben. Cambiar condiciones, dominios, seeds u objetivo sigue siendo
un cambio científico, no una excepción automática a las garantías de retry. Un sweep fijo sigue
ejecutando cada celda/seed obligatoria sin poda adaptativa ni replanteamiento científico.

La prueba genérica `tests/work/test_conditional_membership.py` produce ramas de 1, 2, 10, 16, 9 y
18 candidatos: 56 × cuatro seeds = 224 Runs obligatorios. Compara el diseño unido con el conjunto
previsto de ramas independientes, sin importar ni modificar ningún proyecto consumidor.
