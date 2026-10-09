# Espacio de investigación de Studies

[English](RESEARCH_ANALYSIS.md) · [Manual](MANUAL.es.md#16-análisis-de-estudios)

Versión actual de la aplicación: **0.18.0**. Análisis v8 y vistas guardadas v1 siguen compatibles;
los informes siguientes conservan el contexto histórico de su implementación.

## Índice

1. [Abrir el espacio](#1-abrir-el-espacio)
2. [Declarar significado, no política de ejecución](#2-declarar-significado-no-política-de-ejecución)
3. [Familias y preguntas](#3-familias-y-preguntas)
4. [Cómo funciona el descubrimiento](#4-cómo-funciona-el-descubrimiento)
5. [Explorar y guardar](#5-explorar-y-guardar)
6. [Arquitectura, seguridad y límites](#6-arquitectura-seguridad-y-límites)
7. [Informe de implementación](#7-informe-de-implementación)
8. [Informe de la segunda simplificación](#8-informe-de-la-segunda-simplificación)

## 1. Abrir el espacio

```bash
lf results analyze EXECUTION --recompute --json
lf results report EXECUTION --output research.html
```

El HTML necesita `lambdaforge[analysis-report]`; el análisis numérico no necesita Plotly. Para un
Study remoto usa `lf export STUDY --output ./exports` o Export en la consola. El informe funciona
sin conexión: es una captura, no una conexión viva al clúster. Regenera para incluir nuevas Runs.

**Overview / Resumen** es la primera pestaña y la vista inicial de un informe nuevo. Muestra
estados con recuentos exactos, contexto de selección/confirmación, asociaciones predictivas no
condicionales y recomendaciones que abren el gráfico concreto. El score por número de candidato
queda plegado en **Trials**, no domina el resumen. Al reabrir se conserva la última pestaña.

Los parámetros condicionales no entran en el gráfico de importancia global. Su tabla muestra la
regla de activación, candidatos activos/total, soporte completo, score modelado guardado y
fiabilidad. El score puede mezclar activación de la rama con variación interna; no es un porcentaje
causal, una cuota aditiva de responsabilidad ni una justificación para elegir esa rama.
Interpretation usa un carrusel horizontal desplazable con ratón/teclado. **ⓘ** reúne explicaciones
comunes de estabilidad y ruido; los detalles conservan la conclusión original exacta.

**Language / Idioma** en la cabecera cambia inglés/español inmediatamente y guarda la elección
para ese HTML. Cambian controles, ayudas, estados y etiquetas de gráficos compatibles. Los nombres
declarados y las afirmaciones científicas guardadas conservan su idioma original: ningún servicio
remoto traduce ni modifica la evidencia.

**Research** conserva las tarjetas de salud. Preguntas e inbox exploratorio usan carruseles;
la lista completa está paginada de seis en seis.
**Inspect** explica evidencia, método, soporte, ranking y límites;
**Explore these observations** abre los valores registrados. Una asociación exploratoria no es
una nueva conclusión del HPO ni un efecto causal.

**Metrics & health** busca etiquetas, claves estables, descripciones, alias y tags. Filtra por
categoría y ordena por prioridad, cobertura o dispersión. Las métricas constantes, ausentes u
ocultas se omiten al principio, no se borran: activa **Show constants / missing / hidden**.
El inspector muestra unidades, dirección, agregación y ausencias explícitas. Los grupos de
redundancia son descriptivos; sus miembros siguen accesibles. **Ctrl/⌘ K** busca métricas,
parámetros, familias, hallazgos, Trials y vistas guardadas. Los controles de métrica, parámetro,
ejes, tipo, paleta y filtros usan un desplegable buscable junto al botón, no una modal. El buscador
global **Ctrl/⌘ K** conserva su diálogo con favoritos/recientes. **Categories** es un árbol desplegable: seleccionar
`validation` incluye descendientes como `validation/global` y `validation/surface/quality`.

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
      label: Acuerdo con la accuracy transformada
      priority: 10
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
y registra la pregunta no disponible. `label` y `priority` no negativa son opcionales y nombran/
ordenan tarjetas, no ponderan HPO. Las preguntas enfocan la exploración, no planifican Runs ni
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

En **Explore**, elige **Analyze** (métrica), **By** (chips de parámetros), **Compare with** opcional
(chips de métricas) y el conjunto completo/parcial. **Explore observations** usa el renderer
existente: un parámetro produce scatter, dos heatmap y más coordenadas paralelas. Elimina chips
con ×. Métricas y controles avanzados conservan su preview inmediato. **Advanced visualization
options** agrupa tipo/ejes X/Y/Z explícitos, agregación, paleta/inversión y normalización visual;
tipo/paleta/filtros usan el mismo desplegable buscable. Nombre, notas y guardado quedan visibles.

- Varias Y comparten escala solo con unidades declaradas iguales. Unidades distintas/desconocidas
  usan gráficos separados; **Visual 0–1 normalization only** es opcional y no cambia evidencia.
- En **Parameters**, abre **Analyze metric**, busca `mean`, marca una métrica, busca `auroc` y
  marca otra sin cerrar el desplegable. La selección se conserva entre búsquedas; desmarcar elimina
  la curva y ninguna selección muestra un estado vacío explícito. Flechas navegan, Espacio marca,
  Escape/Listo/clic exterior cierra. **Estilo del gráfico** permite líneas/puntos, barras agrupadas,
  distribuciones de candidatos por X o mapas de calor para muchas métricas. No hay sectores:
  métricas independientes no son partes aditivas de un total. **Distribución** ofrece gráfico
  combinado o un panel por métrica. **Escalas Y → Automáticas** agrupa rangos con al menos un 50%
  de solapamiento del intervalo más estrecho (incluye contención), compatible entre todos los pares
  del grupo. Una constante puede compartir un rango que la contenga. Nunca agrupa unidades conocidas
  distintas; unidades desconocidas pueden compartir escala visual, no significado científico.
  Puedes forzar escalas independientes o compartidas; los paneles separados respetan esa elección.
  También hay puntos por Trial sin promediar, curvas escalonadas y áreas, violines con puntos reales,
  barras horizontales, histogramas y CDF empíricas. Histogramas/CDF agrupan todos los resúmenes
  candidatos coincidentes; cajas/violines permiten comparar valores exactos del parámetro.
  Unir puntos es descriptivo, no un ajuste de respuesta. Los ejes automáticos muestran los nombres
  de sus métricas, no números de escala anónimos.
  No compares alturas entre ejes independientes. **Dispersión** permite ocultar desviación típica,
  usar bigotes o banda ±SD en líneas con X numérico; otros estilos usan bigotes. Las distribuciones
  ya muestran observaciones. La SD es entre candidatos, no confianza entre semillas, y falta para
  una observación. El mapa normaliza colores por métrica (color neutro para constantes); el hover
  muestra medias originales y lo no observado queda vacío. Los ajustes persisten en este HTML en
  el navegador; regenerarlo los reinicia. Explore conserva gráficos separados por defecto y
  normalización visual opcional.
  La selección no modifica agregación, objetivo ni respuesta modelada persistida.
- Heatmaps/superficies numéricas observadas dejan vacías las combinaciones no probadas. Las
  superficies del surrogate permanecen separadas y etiquetadas. Scatter 3D admite categorías.
- Coordenadas paralelas usan los parámetros elegidos (o el default de ocho en vistas antiguas)
  y una métrica; excluyen coordenadas ausentes explícitamente y no ajustan modelos nuevos.
- Trials conserva marcas de poda/parcialidad. Busca en el ledger y compara dos candidatos en una
  tabla por categorías con valores/diferencias. Hay filtro por rama condicional raíz cuando aplica.
  La comparación empieza con métricas principales/prioritarias; buscar o **Show all recorded
  metrics** la amplía. Dirección desconocida no significa mejora.

**Interactions** tiene un panel observado para cualquier métrica con los selectores compartidos
y heatmap/3D/superficie. El modelo predictivo del objetivo permanece separado. **Evidence**
(antes Findings & evidence) conserva conclusiones científicas, diagnóstico del surrogate, evidencia
de seeds/poda, metodología y snapshot estructurado en secciones plegables. Los hallazgos exploratorios aparecen solo en
Research; el dashboard individual de Run no cambia y sigue usando epochs.
El snapshot completo de auditoría se formatea al desplegarlo y reutiliza el catálogo de métricas,
serializado una sola vez. Todos los paneles comparten el catálogo leído, sin selectores independientes.

Guarda vistas con nombre y notas. Exporta/importa **research-views.json**, que transporta
configuración visual, no evidencia. Se validan versión, campos, referencias, tamaño y cantidad.
Las preferencias del mismo HTML conservan vistas, paletas, tamaños, favoritos y búsquedas;
regenerarlo crea un ámbito nuevo. No hay servicios cloud ni LLM necesarios.

### Pestañas HTML del proyecto

La visualización de dominio pertenece al consumidor. Dentro de `Work.run`, declara un HTML
gestionado y escribe el documento completo de tu aplicación:

```python
viewer = self.outputs.html_section("predictions", section="proteins", title="Proteins")
viewer.write_text(render_predictions_html())
```

La acción HTML interactiva del Study, `lf results report SELECTOR --output report.html` y la
exportación portable leen estos archivos solo bajo petición explícita. Una sección crea una pestaña
junto a Explore/Evidence/Resources. Varios outputs o Runs comparten pestaña con selector buscable
etiquetado por Trial/seed/output. Los títulos de una sección deben coincidir. Overview, HPO,
paneles Analysis y sondeos ordinarios no descargan HTML. La lectura remota se pagina y cachea en
su propio host, fuera del heartbeat del planificador.

El documento es HTML UTF-8 autocontenido: incluye JavaScript/CSS, visualizaciones y datos de
predicciones; no funcionan rutas relativas a recursos ni peticiones externas. LambdaForge no
programa lógica de dominio, lee NPZ ni expone callbacks Python al navegador. Cada documento usa un
iframe de origen opaco `sandbox="allow-scripts allow-downloads"` y CSP restrictiva; sus scripts no
acceden al dashboard padre, archivos locales, otros sitios ni credenciales. Esto aísla presentación,
no el Python confiable del Work. Dentro del iframe el HTML sigue siendo interactivo; preferencias
y navegación del dashboard son independientes. Antes de incrustarlo se verifican ownership,
tamaño exacto y fingerprint persistido. Symlinks, contenido ausente/modificado o exceso de límites
(16 MiB/documento, 64 MiB/informe) dan error explícito, no pestañas vacías. Se usa el lifecycle y
retención del output ordinario, sin otro registro, runner o sistema de plugins.

También se pueden aportar documentos directamente sin ejecutar un Work:

```python
from lambdaforge.analysis.Report import write_html

write_html(analysis, "report.html", sections=[
    {"name": "proteins", "title": "Proteins", "label": "Trial 3 · seed 7", "html": html_text},
])
```

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
23. **Versiones/esquema al implementar:** release era **0.16.0**; documento de análisis pasó a **8**;
    documentos research/semánticos comienzan en **1**. YAML añade declaraciones explícitas opcionales.
24. **Evidencia anterior:** no exige declaraciones; ausencia de semántica se infiere explícitamente.
    Se conservan campos científicos. HTML legado obtiene navegación, no hallazgos inventados.
25. **Límites reales:** discovery acotado/observacional puede seguir preliminar; no se afirman modelos
    genéricos de saturación, fairness/subgrupos ni causalidad de alto orden. Se reutilizan análisis
    previos de top-region/recursos/seeds, no se generalizan a toda métrica. No se añade ingestión de
    baselines externos ni color/tamaño/faceting arbitrario de candidatos: la referencia es el Trial
    seleccionado en comparación de dos Trials. Las vistas guardadas no son un workspace vivo.

No se modificó WISDOM ni se contactó ningún clúster real durante la verificación.

## 8. Informe de la segunda simplificación

Este apartado recoge la segunda ronda focalizada del 2026-10-03; el apartado 7 describe la base previa.
La [guía de Studies condicionales](CONDITIONAL_STUDIES.es.md) contiene gramática y ejemplos de planificación.

1. **Controles retirados:** selects estáticos largos de métricas/parámetros en ranking, Parameters y
   X/Y/Z; segundo catálogo de checkboxes/búsqueda para métricas adicionales; select plano de categorías.
   Los valores ocultos solo adaptan el renderer: no quedan listas gigantes de opciones ocultas.
2. **Capacidades conservadas:** ranking, varias métricas/unidades, heatmaps/3D/superficies observadas,
   coordenadas paralelas, marcas parciales/pruned, familias, comparación de candidatos, modelos
   persistidos, vistas/import/export, notas, favoritos/recientes y dashboard individual de Run.
3. **Pestañas:** Findings & evidence pasa a Evidence. Conserva conclusiones/auditoría de seeds/poda/
   surrogate; los hallazgos exploratorios quedan en Research. Las otras pestañas mantienen su finalidad.
4. **Selector:** un catálogo y diálogo dinámico, resultados acotados, búsqueda semántica/aproximada,
   teclado, favoritos/recientes. Los parámetros muestran tipo/dominio/activación, no solo nombres.
5. **Explore:** Analyze / By / Compare with / subconjunto, chips eliminables y elección de gráfica.
   Ejes X/Y/Z, tipo, agregación, paleta y normalización siguen disponibles en Advanced.
6. **Preguntas:** tarjetas con etiqueta/prioridad, estado legible, soporte/motivo e Inspect antes de
   los hallazgos exploratorios. label/priority afectan presentación, no política científica.
7. **Jerarquía:** árbol de categorías desplegable; seleccionar un padre incluye descendientes.
   Las dimensiones de familias de métricas no se confunden con categorías o condiciones de parámetros.
8. **Gramática:** `when: {parent: scalar}`, `{parent: {eq: scalar}}` o
   `{parent: {in: [scalar, ...]}}`; varios padres significan AND. Listas ambiguas/operadores ajenos fallan.
9. **Representación:** predicados ActivationCondition inmutables dentro de ParameterDescriptor;
   orden canónico de padres/miembros, serialización escalar histórica y persistencia JSON.
10. **Autoridad:** normalización Work, sweep exhaustivo, RandomSearch, geometría ParameterSpace/
    Sobol/adaptativa, codificación/decodificación/validez y activación en análisis comparten predicados.
11. **Pertenencia:** valores finitos, únicos y no vacíos comprobados contra el dominio del padre;
    sin OR implícito, expresiones recursivas, expansión del dominio ni API específica de un consumidor.
12. **Igualdad:** eq explícito se normaliza al escalar antiguo. Igualdad sobre rangos numéricos y
    orden declarado de dominios independientes conservan comportamiento; las regresiones protegen identidad.
13. **Inactivos:** claves ausentes de candidatos/argumentos; se aplican defaults de la firma. Sin
    null, categoría inactiva inventada o serialización de valores irrelevantes de hijos.
14. **Duplicados:** enumeración topológica activa evita multiplicar ramas inactivas; valores repetidos
    del sweep fallan en vez de duplicar identidades de evidencia silenciosamente.
15. **Fixture:** seis ramas genéricas dan 1+2+10+16+9+18 = **56 candidatos únicos** × cuatro seeds
    compartidas = **224 Runs obligatorios**, iguales a la unión pretendida de ramas independientes.
16. **Referencia:** el selector existente debe identificar exactamente un candidato generado; cero/
    varios resultados fallan. No se inventan referencias a valores inactivos.
17. **Preflight:** validate/explain/dry-run muestran diseño/seeds/Runs/ramas/referencia, paralelismo,
    GPUs y presupuestos. Dry-run humano no enumera cientos de Runs; JSON conserva el plan completo
    y la observación de capacidad del destino.
18. **Tiempo:** wall-time del scheduler y presupuesto lógico de dispatch siguen separados. Los
    niveles secuenciales suman; los paralelos usan el máximo. Pruebas de 1008 h/42 días y composición mixta.
19. **Capacidad:** sondeo directo UUID/visibilidad acotado rechaza peticiones GPU imposibles conocidas
    antes de empaquetar/enviar. Fallo/ambigüedad permanecen desconocidos; ocupación no es capacidad.
    Scheduler/site-command mantienen su concesión existente; preflight nunca reserva ni amplía GPUs.
20. **Steps:** with/seeds/replicates/search/sweep/execution/objective/analysis raíz fallan explícitamente
    en vez de ignorarse. resources raíz mantiene su herencia documentada.
21. **Hook consumidor:** no se añade. Se reutiliza validación de clase/firma/tipos/marcadores sin
    construir Work. Semántica arbitraria necesita validación del consumidor; no era necesario crear
    otro contrato de callbacks para este cambio focalizado.
22. **Esquema/versión:** condiciones y label/priority opcionales añadidos; se rechaza política raíz
    sin efecto. Implementado sobre 0.16.0; análisis v8 y vistas v1 compatibles. Sin otra versión YAML,
    runner, política científica o DSL paralela de condiciones.
23. **Compatibilidad:** no se reescribe evidencia histórica congelada. Identidad escalar, seeds,
    objetivo/HPO/recursos, retry/export y HTML/vistas antiguos se conservan. Las condiciones viajan
    por StudyDesign, inicialización e identidad de recuperación existentes.
24. **Pruebas:** gramática/AND/dominio/ciclos/orden, inmutabilidad/pickle/esquema/identidad, sweep/
    referencia, geometría Sobol/random/análisis, steps/raíz/tiempos y capacidad previa al submit sintética.
    Chromium cubre 300 métricas/15 parámetros, jerarquía, preguntas, selector/chips, gráficas nuevas y
    previas, comparación y recarga de vistas. Ruff/mypy pasan; pytest completo: **1136 passed**,
    cuatro avisos no fatales de Lightning. Revisión final análisis/navegador/config/docs: **63 passed**;
    política GPU/config/docs/smoke de entrenamiento: **34 passed**. Wheel, instalación aislada y
    scaffold/validate/dry-run también pasan reutilizando dependencias instaladas del entorno local.
    Integración CUDA local explícita: **1 passed**. La regresión de padre numérico comprueba que
    las observaciones no transforman un rango declarado en dominio finito.
25. **Límites:** sin DSL de expresiones, nuevo fitting, fusión automática de pasos, lint heurístico de
    particiones o reglas de consumidor. Capacidad incierta permanece desconocida; in necesita dominio
    finito. Conteos adaptativos describen una ventana, no garantizan ejecución. HTML sigue offline;
    los controles no programan/podan Runs ni modifican evidencia.

Verificación local/sintética. No se editó ningún proyecto consumidor ni se envió ningún Work remoto.
