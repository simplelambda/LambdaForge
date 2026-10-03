# Espacio de investigación de Studies

[English](RESEARCH_ANALYSIS.md) · [Manual](MANUAL.es.md#16-análisis-de-estudios)

## Índice

1. [Abrir el espacio](#1-abrir-el-espacio)
2. [Declarar significado, no política de ejecución](#2-declarar-significado-no-política-de-ejecución)
3. [Familias y preguntas](#3-familias-y-preguntas)
4. [Cómo funciona el descubrimiento](#4-cómo-funciona-el-descubrimiento)
5. [Explorar y guardar](#5-explorar-y-guardar)
6. [Arquitectura, seguridad y límites](#6-arquitectura-seguridad-y-límites)
7. [Informe de implementación](#7-informe-de-implementación)

## 1. Abrir el espacio

```bash
lf results analyze EXECUTION --recompute --json
lf results report EXECUTION --output research.html
```

El HTML necesita `lambdaforge[analysis-report]`; el análisis numérico no necesita Plotly. Para un
Study remoto usa `lf export STUDY --output ./exports` o Export en la consola. El informe funciona
sin conexión: es una captura, no una conexión viva al clúster. Regenera para incluir nuevas Runs.

**Research** se abre primero: salud de métricas, estado científico/seeds/cobertura ya calculado y
hasta ocho hallazgos priorizados. **Inspect** explica evidencia, método, soporte, ranking y límites;
**Explore these observations** abre los valores registrados. Una asociación exploratoria no es
una nueva conclusión del HPO ni un efecto causal.

**Metrics & health** busca etiquetas, claves estables, descripciones, alias y tags. Filtra por
categoría y ordena por prioridad, cobertura o dispersión. Las métricas constantes, ausentes u
ocultas se omiten al principio, no se borran: activa **Show constants / missing / hidden**.
El inspector muestra unidades, dirección, agregación y ausencias explícitas. Los grupos de
redundancia son descriptivos; sus miembros siguen accesibles. **Ctrl/⌘ K** busca métricas,
parámetros, familias, hallazgos, Trials y vistas guardadas. El mismo selector está junto a los ejes;
marca favoritos con la estrella y reutiliza selecciones recientes.

## 2. Declarar significado, no política de ejecución

Sin declaraciones todo sigue funcionando. Las direcciones/rangos estándar y prefijos simples
identifican parte de la semántica; lo demás sigue desconocido. Añade `analysis` al YAML de un Work,
o un mapping de clase `Work.analysis_profile` con la misma estructura:

```yaml
analysis:
  defaults:
    - pattern: val_*
      metadata: {category: validation, split: validation, unit: ratio}
  metrics:
    val_accuracy:
      label: Accuracy
      description: Fracción de ejemplos de validación clasificados correctamente.
      category: validation/quality
      direction: max
      range: [0, 1]
      aggregation: latest
      visibility: primary
      aliases: [accuracy]
      priority: 10
    rejected_fraction:
      category: integrity
      expected_to_vary: true
      unit: ratio
      range: [0, 1]
    transformed_accuracy:
      derived_from: [val_accuracy]
      transformation: Transformación explícita definida por el proyecto.
```

Las etiquetas no renombran claves. Las unidades no convierten valores automáticamente.
`aggregation` describe evidencia real: `latest`, `best`, `selected_epoch`, `terminal`, `mean` o
`unspecified`. Declarar `best` **no** calcula un óptimo inexistente. La selección conserva su
semántica checkpoint/seed y los diagnósticos conservan sus valores reales; la agregación
desconocida se advierte, no se adivina.

También existen `tags`, `role`, `phase`, `scale`, `notes`, `discovery: false`, `practical_scale` y
`visibility: normal|advanced|hidden`. La escala práctica afecta a la salud visual, no al margen de
equivalencia científica del objetivo. Precedencia: inferencia, reglas de patrón por orden,
declaraciones de clase y overrides YAML. Los campos individuales de una métrica se combinan;
questions/defaults YAML sustituyen esas colecciones de la clase. Familias se combinan por nombre:
la definición YAML sustituye la familia homónima; campos individuales de métricas se combinan.
En una composición
`steps`, declara `analysis` en cada Work o clase, no en la raíz. Las métricas personalizadas pueden
pasar `metadata=` al constructor base de `Metric` y reutilizar `.metadata` en el perfil; no se
construye un Work para descubrir métricas transitorias.

Se validan localmente campos, rangos, categorías, direcciones, alias, coordenadas de familias,
referencias y ciclos de derivación. Métricas de test, incluso derivadas, no pueden gobernar el
objetivo ni restricciones, tampoco dentro de una utilidad compuesta. Siguen disponibles para
análisis final a posteriori, pero se excluyen del descubrimiento automático provisional. El
análisis no inventa restricciones, pesos del objetivo ni reglas de poda.

## 3. Familias y preguntas

Agrupa mediciones repetidas por su significado:

```yaml
analysis:
  families:
    efficiency:
      label: Calidad por fracción de datos y estrategia
      dimensions:
        strategy: {kind: categorical, values: [uniform, balanced]}
        fraction: {kind: ordered, values: [10, 25, 100]}
      template: quality_{strategy}_{fraction}
  questions:
    - {id: efficiency-view, kind: metric_family, family: efficiency}
    - id: capacity-quality
      kind: parameter_screen
      parameters: [width]
      metrics: [val_accuracy]
    - id: expected-agreement
      kind: relationship
      x: val_accuracy
      y: transformed_accuracy
      expected: positive
```

Dimensiones numéricas, ordenadas o categóricas. `members` puede asignar explícitamente claves a
coordenadas en vez de usar template. Las plantillas admiten hasta 4096 miembros únicos. Las
gráficas conservan orden, etiquetas, valores ausentes y soporte. Su SD es dispersión entre
candidatos, no confianza entre seeds; soporte desigual no representa una intervención pareada.
Los miembros de familias no se colapsan automáticamente como redundantes.

Tipos de pregunta: `relationship`, `consistency`, `parameter_screen`, `metric_family`,
`category_summary` y `tradeoff`. Relaciones/consistencia necesitan métricas `x`/`y` y admiten
`expected: positive|negative|equal`; tradeoff enumera al menos dos métricas y category_summary una
categoría. Referencias requeridas desconocidas fallan; `optional: true` admite una métrica ausente
y registra la pregunta no disponible. Las preguntas enfocan la exploración, no planifican Runs ni
crean un protocolo de contraste formal. Resúmenes de familia/categoría y Pareto de recursos ya
existente siguen siendo descriptivos.

## 4. Cómo funciona el descubrimiento

1. Matriz por candidato sobre evidencia **completa comparable**, sin fabricar scores finales de
   candidatos censurados. Screening y confirmación siguen separados; no se mezclan fidelidades.
   Los alias se mapean a claves canónicas de análisis.
2. Perfil de todas las métricas: soporte finito/ausente, valores únicos, rango, media, mediana, SD,
   dispersión robusta y soporte registrado de Runs/seeds. Constancia exige igualdad repetida.
   Las métricas presentes en menos de la mitad de los candidatos muestran `low_coverage`;
   conservan cobertura de candidatos/Runs y soporte de seeds numéricos. El ranking penaliza
   también la cobertura pareada real, sin equiparar unos pocos ejemplos con evidencia completa.
   La clasificación casi constante compara dispersión con rango/escala declarada o magnitud (`1e-6`),
   nunca un epsilon absoluto que destruya señales de unidades pequeñas.
3. Conjunto acotado y determinista, priorizado por la declaración. Grupos de respuestas casi
   redundantes: Spearman absoluto al menos `.995` y seis candidatos pareados, excluyendo familias.
   Es una ayuda de navegación, no prueba de equivalencia científica.
4. Screening acotado parámetro/métrica y métrica/métrica, intercalando ambos tipos y rotando
   parámetros para que el primero no monopolice el presupuesto. Los parámetros numéricos reutilizan
   `ParameterSpace`, incluidas escalas log, categorías y condiciones. Se comparan correlación de
   rangos y respuesta cuadrática de rangos con leave-one-out; categorías usan eta-cuadrado.
   Ninguna de estas relaciones es un nuevo modelo causal multivariante.
5. Hasta 32 relaciones preseleccionadas reciben permutaciones deterministas y bootstrap de signo.
   Se reutilizan matrices pequeñas, sin matrices cuadráticas de candidatos ni productos completos
   de métricas. Tests sin confirmar cuentan como `p=1` en el ajuste Benjamini–Yekutieli. Valores
   por defecto: 64 métricas, 128 relaciones y 256 permutaciones; `analysis.discovery` permite
   ajustar `max_metrics`, `max_pairs`, `resamples` dentro de límites validados o desactivar discovery.
6. Hallazgos priorizados con efecto, soporte y estabilidad separados. La derivación directa,
   transitiva o compartida reduce novedad. También hay invariancia, outliers descriptivos por
   1.5-IQR, cambios de signo en contextos categóricos acotados y hallazgos del análisis existente.
   Persisten componentes del ranking y penalizaciones por redundancia. El inbox muestra hasta
   ocho grupos de evidencia distintos; los equivalentes permanecen en la lista completa.

Efecto, soporte, estabilidad, p ajustado y confianza científica no son lo mismo. **Supported
exploration** exige soporte y p diagnóstico ajustado pequeño; **preliminary** no está confirmado.
La selección adaptativa, comparaciones múltiples, reutilización de datos y ruido de seeds limitan
estos diagnósticos. BY no transforma un Study adaptativo en experimento aleatorizado formal.
El acuerdo del bootstrap no es la probabilidad de que una conclusión sea verdadera. Outliers y
contextos son explícitamente inspection-only, sin confianza inventada. Métodos basados en la
[guía de permutaciones de SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)
y el [ajuste de FDR con dependencia](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html).

## 5. Explorar y guardar

En **Explore**, elige métrica y parámetro; los cambios se previsualizan. El buscador compartido
evita repetir cientos de opciones estáticas. Advanced agrupa tipo, agregación, paleta, datos
parciales y normalización visual explícita.

- Varias Y comparten escala solo con unidades declaradas iguales. Unidades distintas/desconocidas
  usan gráficos separados; **Visual 0–1 normalization only** es opcional y no cambia evidencia.
- Parameters añade métricas de comparación con el mismo buscador o vuelve a una sola.
- Heatmaps/superficies numéricas observadas dejan vacías las combinaciones no probadas. Las
  superficies del surrogate permanecen separadas y etiquetadas. Scatter 3D admite categorías.
- Coordenadas paralelas muestran hasta ocho parámetros y una métrica; excluyen coordenadas
  ausentes explícitamente y no ajustan modelos nuevos.
- Trials conserva marcas de poda/parcialidad. Busca en el ledger y compara dos candidatos en una
  tabla por categorías con valores/diferencias. Dirección desconocida no significa mejora.

Guarda vistas con nombre y notas. Exporta/importa **research-views.json**, que transporta
configuración visual, no evidencia. Se validan versión, campos, referencias, tamaño y cantidad.
Las preferencias del mismo HTML conservan vistas, paletas, tamaños, favoritos y búsquedas;
regenerarlo crea un ámbito nuevo. No hay servicios cloud ni LLM necesarios.

## 6. Arquitectura, seguridad y límites

`MetricCatalog` posee significado; `AnalysisProfile` valida declaraciones inmutables;
`ResearchAnalysis`/`ResearchDiagnostics` calculan read models deterministas; assets offline
modulares los muestran. `StudyAnalysis` conserva efectos, cobertura, seeds, ganador/confirmación
y recursos. El documento versión **8** añade `research` sin eliminar campos. Se elimina el viejo
renderer lineal sin uso, no se mantiene una segunda ruta.

Antes de ejecutar, `analysis-semantics.json` guarda declaraciones resueltas, defaults conocidos
del registro, identidad y versiones. Las especificaciones hijas referencian ese fichero, sin
copiar el catálogo por Run. Finalización, recarga, recuperación y export usan evidencia congelada;
cambiar etiquetas no cambia el fingerprint científico previo ni decisiones HPO. La caché de
análisis verifica además la identidad semántica. Studies antiguos reciben metadata inferida;
documentos HTML antiguos obtienen navegación de nombres, no hallazgos retrospectivos inventados.

Límites deliberados: discovery acotado, no exhaustivo; semántica desconocida permanece así;
señales no lineales/contextuales son herramientas aproximadas de inspección, no un modelo
universal de interacciones; no causalidad ni pesos nuevos; no nueva confianza shared-seed;
sin imputación, datos fabricados ni fitting científico en navegador. HPO y análisis multivariante
previos siguen siendo autoridades. Export/import transporta vistas, no un workspace vivo ni
actualización automática del HTML. Las conclusiones acopladas rendimiento/recursos necesitan
evidencia comparable suficiente de recursos y seeds.

## 7. Informe de implementación

Este informe recoge la implementación y verificación local del 2026-10-03.

1. **Arquitectura previa:** Study Analysis ya normalizaba evidencia y gestionaba ganador/seeds,
   cobertura, efectos multivariantes y recursos. El HTML era principalmente una colección de gráficas.
2. **Causas de usabilidad:** nombres/selectores planos, poca semántica, métricas ausentes/constantes
   compitiendo por atención, navegación monométrica y falta de priorización para investigación.
3. **Estructuras nuevas:** `MetricCatalog`, `AnalysisProfile` inmutable, `ResearchAnalysis` acotado,
   helpers de inspección/ranking y presentación offline modular integrada en el análisis existente.
4. **Catálogo:** `analysis.metrics/defaults` opcional o `Work.analysis_profile`; se reutiliza
   `Metric.metadata` sin construir objetos Work científicos.
5. **Perfil:** declaraciones, familias y preguntas/discovery validados; combina clase/YAML y se
   congela de forma independiente del objetivo y política de ejecución.
6. **Familias:** coordenadas explícitas o plantillas acotadas con placeholders simples; orden,
   categorías y soporte incompleto aparecen juntos sin inventar mediciones.
7. **Salud:** igualdad repetida, casi constancia relativa, ausencias, cobertura de candidatos/Runs/
   seeds, rangos y variación esperada; no borra ni imputa evidencia.
8. **Redundancia:** agrupación determinista por similitud de rangos, representante y miembros
   desplegables; protege familias. Ranking/inbox reducen repetición, no eliminan valores.
9. **Discovery:** screening intercalado parámetro/métrica y métrica/métrica, respuesta de rangos/
   cuadrática, asociaciones categóricas, relaciones declaradas y señales baratas de inspección.
10. **Multiplicidad:** permutaciones deterministas acotadas y ajuste BY sobre hipótesis examinadas;
    tests sin confirmar cuentan como `p=1`. La resolución limitada sigue siendo conservadora.
11. **Fiabilidad:** soporte, efecto, cobertura, estabilidad de signo, p diagnóstico ajustado y
    novedad son componentes separados; ninguno es probabilidad de verdad.
12. **Derivadas:** linaje acíclico validado; relaciones estructurales directas/transitivas/compartidas
    pierden novedad. No se infiere dirección ni equivalencia práctica de una fórmula derivada.
13. **Test:** se rechaza evidencia de test/derivada para objetivo/componentes/restricciones antes
    de ejecutar. Discovery provisional la excluye; análisis terminal conserva sus valores registrados.
14. **Persistencia:** `analysis-semantics.json`, identidad semántica separada y referencias hijas;
    defaults congelados al finalizar, recargar, recuperar y exportar.
15. **HTML:** inbox Research, salud/catálogo, inspectores semánticos, hallazgos diversos y Explore
    progresivo; se retira el renderer lineal viejo sin uso.
16. **Búsqueda:** palabras/subsecuencias sobre claves, etiquetas, alias, descripciones/categorías,
    parámetros, familias, hallazgos, Trials y vistas; teclado, favoritos y recientes.
17. **Parameters multimétrica:** renderer observado compartido; añade métricas, separa escalas
    desconocidas/distintas y normaliza visualmente solo cuando se solicita.
18. **Pares/3D:** scatter/heatmap/superficie observados con categorías y huecos; superficies modeladas
    del objetivo siguen separadas. Coordenadas paralelas acotadas, sin otro modelo científico.
19. **Explore:** preview inmediato, X/Y/Z, varias Y, tipo, agregación, paleta, evidencia parcial y
    normalización explícita; Advanced no domina la experiencia inicial.
20. **Research Views:** especificaciones guardadas con nombre/notas y preferencias por HTML;
    import/export JSON validado/acotado transporta presentación, no datos científicos.
21. **Regresiones añadidas:** precedencia/validación, leakage/linaje, recarga congelada, namespaces,
    censura/ausencias, salud, familias, discovery determinista/no lineal, catálogos grandes y navegador:
    búsqueda, inspectores, unidades, comparaciones y persistencia/import de vistas.
22. **Verificación:** Ruff/mypy correctos; pytest completo: **1118 passed**, cuatro avisos Lightning
    no fatales sobre uso CPU/loaders. Pasan análisis/CUDA/documentación focalizados, smoke de wheel
    instalada con CLI/scaffold/validación/HTML e interacción/capturas Chromium. Matriz sintética de
    500 candidatos, 300 métricas, 15 parámetros y 15 seeds: **9,424 s**, **4.809.115 bytes** y solo
    32 relaciones con remuestreo costoso por defecto. El tiempo depende de la máquina.
23. **Versiones/esquema:** release continúa en **0.16.0**; documento de análisis pasa a **8**;
    documentos research/semánticos comienzan en **1**. YAML añade declaraciones explícitas opcionales.
24. **Evidencia anterior:** no exige declaraciones; ausencia de semántica se infiere explícitamente.
    Se conservan campos científicos. HTML legado obtiene navegación, no hallazgos inventados.
25. **Límites reales:** discovery acotado/observacional puede seguir preliminar; no se afirman modelos
    genéricos de saturación, fairness/subgrupos ni causalidad de alto orden. Se reutilizan análisis
    previos de top-region/recursos/seeds, no se generalizan a toda métrica. No se añade ingestión de
    baselines externos ni color/tamaño/faceting arbitrario de candidatos: la referencia es el Trial
    seleccionado en comparación de dos Trials. Las vistas guardadas no son un workspace vivo.

No se modificó WISDOM ni se contactó ningún clúster real durante la verificación.
