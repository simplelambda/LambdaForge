/* Offline presentation translations. Metric keys, authored labels and persisted evidence
 * remain untouched. No requests to translation services and no changes to HPO policy. */
window.LambdaForgeStudyLocale = (() => {
  let language = 'en', services;
  const englishStatus = {UNRESOLVED:'Unresolved',NO_CLEAR_PREFERENCE:'No clear preference',
    CONTEXT_DEPENDENT:'Context dependent',PRACTICALLY_EQUIVALENT:'Practically equivalent',
    PRACTICAL_TOP_SET:'Practical top set',PREFERRED:'Supported preference',PREFERRED_REGION:'Preferred region',
    WEAK_PREFERENCE:'Preliminary preference',FLAT:'No stable tendency',RESOLVED:'Resolved',
    scientifically_unresolved:'Scientifically unresolved',scientifically_resolved:'Scientifically resolved'};
  const es = {
    'Overview':'Resumen','Research':'Investigación','Metrics & health':'Métricas y calidad',
    'Trials':'Candidatos','Parameters':'Parámetros','Interactions':'Interacciones','Coverage':'Cobertura',
    'Resources':'Recursos','Evidence':'Evidencia','Explore':'Explorar','Study Analysis dashboard':'Análisis del estudio',
    'Explore persisted evidence; every association remains descriptive or predictive.':'Explora la evidencia guardada; las asociaciones son descriptivas o predictivas.',
    'Candidates':'Candidatos','Complete candidates':'Candidatos completos','Censored Runs':'Ejecuciones censuradas','Leading trial':'Candidato líder',
    'Study progress':'Progreso del estudio','Candidate states':'Estado de los candidatos','Completed':'Completados','Pruned':'Podados',
    'Active':'Activos','Failed':'Fallidos','Other':'Otros','Selection metric':'Métrica de selección','Conclusion':'Conclusión',
    'Evidence source':'Origen de la evidencia','Not recorded':'Sin datos registrados','unknown':'desconocido','unavailable':'sin datos',
    'final':'final','provisional':'provisional','succeeded':'completado','failed':'fallido','running':'en ejecución',
    'UNRESOLVED':'Sin resolver','NO_CLEAR_PREFERENCE':'Sin preferencia clara','PRACTICALLY_EQUIVALENT':'Prácticamente equivalentes',
    'CONTEXT_DEPENDENT':'Depende del contexto','PREFERRED_REGION':'Región preferida','WEAK_PREFERENCE':'Preferencia preliminar','FLAT':'Sin tendencia estable',
    'scientifically_unresolved':'Sin resolver científicamente','scientifically_resolved':'Resuelto científicamente',
    'PRACTICAL_TOP_SET':'Conjunto de mejores alternativas','PREFERRED':'Preferencia respaldada','RESOLVED':'Resuelto',
    'low':'baja','medium':'media','high':'alta','unrated':'sin evaluar',
    'Only the authored objective governs selection. Other metrics are diagnostics.':'Solo el objetivo declarado decide la selección. Las otras métricas son diagnósticas.',
    'Execution success does not guarantee scientific certainty.':'Que la ejecución termine bien no garantiza certeza científica.',
    'Confirmed on fresh seeds.':'Confirmado con semillas nuevas.',
    'Screening leader; confirmation may still be pending.':'Líder provisional; puede quedar confirmación pendiente.',
    'This report is a snapshot, not a live controller.':'Este informe es una instantánea, no un controlador en vivo.',
    'Counts refer to candidates, not seeds. Pruned means incomplete performance evidence, not a software failure.':'Los recuentos son de candidatos, no de semillas. Podado significa evidencia de rendimiento incompleta, no un fallo de software.',
    'Predictive importance':'Importancia predictiva','Predictive association · unconditional parameters':'Asociación predictiva · parámetros no condicionales',
    'Parameter':'Parámetro','Model variation score (not causal responsibility)':'Puntuación de variación del modelo (no responsabilidad causal)',
    'Predictive association, not causal responsibility. Conditional parameters are shown separately with their active branch and support.':'Asociación predictiva, no responsabilidad causal. Los parámetros condicionales se muestran aparte con su rama activa y su soporte.',
    'Conditional parameter evidence':'Evidencia de parámetros condicionales','Active when':'Activo cuando','Observed active / total':'Observados activos / total',
    'Model score':'Puntuación del modelo','Reliability':'Fiabilidad','Support':'Soporte','Open':'Abrir',
    'Model association score':'Puntuación de asociación del modelo','Conclusion values':'Valores de la conclusión',
    'The response depends on other parameters.':'La respuesta depende de otros parámetros.',
    'The supported values are equivalent within the authored practical margin.':'Los valores respaldados son equivalentes dentro del margen práctico declarado.',
    'No stable tendency is visible; this does not establish equivalence.':'No hay una tendencia estable visible; esto no demuestra equivalencia.',
    'A possible preference needs more evidence.':'Una posible preferencia necesita más evidencia.',
    'The available evidence does not resolve this question.':'La evidencia disponible no resuelve esta cuestión.',
    'Inspect the exact conclusion and its supporting values.':'Consulta la conclusión exacta y los valores que la respaldan.',
    'The persisted conclusion supports a preference among the displayed values.':'La conclusión guardada respalda una preferencia entre los valores mostrados.',
    'These scores may mix branch activation with within-branch variation. They do not measure how much a parameter matters inside that branch or justify choosing the branch.':'Estas puntuaciones pueden mezclar la activación de una rama con la variación dentro de ella. No miden cuánto importa el parámetro dentro de la rama ni justifican elegirla.',
    'Interpretation':'Interpretación','Interpretation help':'Ayuda de interpretación','Previous':'Anterior','Next':'Siguiente',
    'No supported conclusion yet. Inspect coverage or the recorded observations.':'Todavía no hay una conclusión respaldada. Consulta la cobertura o las observaciones registradas.',
    'Direct evidence at values':'Evidencia directa en los valores','Conclusion stability':'Estabilidad de la conclusión','More evidence is needed':'Hace falta más evidencia',
    'Conclusion details':'Detalles de la conclusión','View response':'Ver respuesta','Persisted diagnostics':'Diagnósticos guardados',
    'The question remains unresolved.':'La cuestión sigue sin resolverse.','How to interpret this evidence':'Cómo interpretar esta evidencia',
    'Stability measures how often the displayed conclusion survives resampling the available evidence. It is not the probability that the conclusion is true.':'La estabilidad mide cuántas veces se mantiene la conclusión al remuestrear la evidencia disponible. No es la probabilidad de que sea verdadera.',
    'Direct support comes from completed observations; censored support comes from pruned Runs; predictive support comes from the persisted model. They are not interchangeable.':'El soporte directo viene de observaciones completas; el censurado, de ejecuciones podadas; el predictivo, del modelo guardado. No son intercambiables.',
    'Conditional scores may reflect the choice of parent branch, not the effect of changing the parameter within that branch.':'Las puntuaciones condicionales pueden reflejar la elección de la rama padre, no el efecto de cambiar el parámetro dentro de esa rama.',
    'Model variation scores need coverage and validated predictive reliability. They are not additive percentages of responsibility.':'Las puntuaciones de variación necesitan cobertura y fiabilidad predictiva validada. No son porcentajes aditivos de responsabilidad.',
    'Suggested views':'Vistas sugeridas','Where is evidence missing?':'¿Dónde falta evidencia?',
    'Coverage distinguishes tested values from completed response evidence.':'La cobertura distingue valores probados de respuestas completas.',
    'Compare candidates':'Comparar candidatos','Inspect two candidates and their recorded metrics, not just their ranking.':'Compara dos candidatos y sus métricas registradas, no solo su posición.',
    'Inspect a parameter response':'Examinar la respuesta de un parámetro','See recorded means, dispersion and support; other parameters are uncontrolled.':'Consulta medias, dispersión y soporte; los demás parámetros no están controlados.',
    'Inspect a supported relationship':'Examinar una relación respaldada','Open suggested view':'Abrir vista sugerida',
    'Search options':'Buscar opciones','Search names, aliases or categories…':'Buscar nombres, alias o categorías…',
    'selected':'seleccionadas','matches':'coincidencias','Type to narrow the results':'Escribe para acotar los resultados','Done':'Listo',
    'No matching options.':'No hay opciones coincidentes.','Choose evidence':'Elegir evidencia',
    'Research inbox':'Bandeja de investigación','Search everything · Ctrl / ⌘ K':'Buscar todo · Ctrl / ⌘ K',
    'Retrospective associations, not causal effects. Inspect a finding to see support, uncertainty, multiplicity, limitations and the underlying observations.':'Asociaciones retrospectivas, no efectos causales. Abre un hallazgo para ver soporte, incertidumbre, comparaciones múltiples, límites y observaciones.',
    'Configured research questions':'Preguntas de investigación declaradas','Exploratory / integrity / resource signals':'Señales exploratorias / integridad / recursos',
    'All exploratory findings':'Todos los hallazgos exploratorios','Search':'Buscar','Name, alias, description, category…':'Nombre, alias, descripción, categoría…',
    'Show constants / missing / hidden':'Mostrar constantes / ausentes / ocultas','Categories · all':'Categorías · todas','Sort':'Ordenar',
    'Priority':'Prioridad','Name':'Nombre','Spread':'Dispersión','Metric / meaning':'Métrica / significado','Category · split':'Categoría · partición',
    'Unit / direction':'Unidad / dirección','Health / aggregation':'Calidad / agregación',
    'Show all preserves constant evidence. Spread uses authored range/practical scale, or relative magnitude; unknown units and aggregation remain explicitly unknown. Redundancy never deletes recorded metrics.':'Mostrar todo conserva las constantes. La dispersión usa la escala declarada o magnitud relativa; unidades y agregación desconocidas se indican como tales. La redundancia no borra métricas.',
    'Redundancy groups':'Grupos redundantes','Metric families':'Familias de métricas','Close':'Cerrar','Search / metric picker':'Buscador de investigación',
    'Metrics, aliases, families, parameters, findings, Trials, saved views…':'Métricas, alias, familias, parámetros, hallazgos, candidatos, vistas guardadas…',
    'Inspect':'Examinar','Inspect question':'Examinar pregunta','Explore these observations':'Explorar estas observaciones',
    'Meaning':'Significado','Profile':'Perfil','Ranking components':'Componentes de clasificación','Limitations':'Limitaciones',
    'Declared question':'Pregunta declarada','Support and status':'Soporte y estado','Recorded relationships':'Relaciones registradas','Findings':'Hallazgos',
    'No questions configured. Exploratory findings remain available.':'No hay preguntas declaradas. Los hallazgos exploratorios siguen disponibles.',
    'No supported exploratory relationships yet. Explore the recorded evidence or inspect metric health; missing support is not a negative result.':'Aún no hay relaciones exploratorias respaldadas. Explora los datos o la calidad de las métricas; falta de soporte no significa un resultado negativo.',
    'No families declared. Ordinary metrics remain available.':'No hay familias declaradas. Las métricas normales siguen disponibles.',
    'Informative metrics':'Métricas informativas','Invariant metrics':'Métricas invariantes','Low coverage':'Cobertura baja',
    'Configured supported':'Declaradas con soporte','Configured unresolved':'Declaradas sin resolver','Exploratory findings':'Hallazgos exploratorios',
    'No matching informative metrics. Use Show all to inspect missing or constant evidence.':'No hay métricas informativas coincidentes. Usa Mostrar todo para examinar ausentes o constantes.',
    'Optional score by trial':'Score por candidato (opcional)','Displayed metric':'Métrica mostrada',
    'Select at most two candidates to compare. Comparable ranking contains terminal selection evidence only. This ledger retains every attempted candidate; × means performance-pruned and its final selection remains unavailable.':'Selecciona hasta dos candidatos para comparar. La clasificación usa solo evidencia final de selección. La tabla conserva todos los candidatos intentados; × indica poda por rendimiento, sin puntuación final inventada.',
    'State':'Estado','Compare':'Comparar','Trial':'Candidato','Final selection':'Selección final','SE':'Error estándar','Seeds':'Semillas',
    'Censored':'Censurados','Partial best':'Mejor parcial','Prune step':'Paso de poda','Prune reason':'Motivo de poda','P(competitive)':'P(competitivo)',
    'Threshold':'Umbral','Reference':'Referencia','Analyze metric':'Métricas a representar','Add comparison metric…':'Añadir métrica comparativa…',
    'Observed metric by parameter value':'Métricas observadas por valor del parámetro','Persisted adjusted selection response and uncertainty (model evidence)':'Respuesta ajustada e incertidumbre guardadas (evidencia del modelo)',
    'This model response describes the selection objective only. Changing Y metric updates the observed chart and table, not this persisted model.':'Esta respuesta del modelo describe solo el objetivo de selección. Cambiar Y actualiza el gráfico observado y la tabla, no el modelo guardado.',
    'Observed exact-value summary':'Resumen observado por valor exacto','Value':'Valor','Mean metric':'Métrica media','Empirical SD':'Desviación típica empírica',
    'Select metrics':'Seleccionar métricas','No metrics selected. Choose one or more metrics above.':'No hay métricas seleccionadas. Elige una o más arriba.',
    'Parameter response metrics':'Métricas de respuesta del parámetro','Observed metric interactions · not the predictive HPO model':'Interacciones métricas observadas · no el modelo predictivo de HPO',
    'Pair':'Par','View':'Vista','Colour scale':'Escala de color','Heatmap':'Mapa de calor','3D surface':'Superficie 3D','3D scatter':'Dispersión 3D',
    'Reverse':'Invertir','Blue ↔ red':'Azul ↔ rojo','Purple ↔ green':'Morado ↔ verde','Brown ↔ teal':'Marrón ↔ verde azulado','Accessible':'Accesible',
    'Joint predictive response is conditional on persisted model evidence and must not be read as a causal effect.':'La respuesta predictiva conjunta depende del modelo guardado; no es un efecto causal.',
    'Domain coverage':'Cobertura del dominio','Coverage details':'Detalles de cobertura','Kind':'Tipo','Authored domain':'Dominio declarado',
    'Observed marginal coverage':'Cobertura marginal observada','Fraction of authored domain':'Fracción del dominio declarado',
    'Persisted adjusted response':'Respuesta ajustada guardada','Select a parameter to inspect its persisted response':'Selecciona un parámetro para ver su respuesta guardada',
    'Select a persisted parameter pair':'Selecciona un par de parámetros guardado',
    'Observed domain':'Dominio observado','Occupied bins':'Intervalos ocupados','Cost':'Coste','GPU seconds':'Segundos GPU','Duration':'Duración',
    'CPU seconds':'Segundos CPU','Peak VRAM':'Pico VRAM','Peak RAM':'Pico RAM',
    'Per-comparable-Run resource evidence is kept separate from total controller spend.':'Los recursos por ejecución comparable se separan del gasto total del controlador.',
    'Complete reproducible analysis JSON':'JSON completo y reproducible','Open this section to inspect the persisted snapshot.':'Abre esta sección para examinar la instantánea guardada.',
    'Surrogate diagnostics':'Diagnósticos del modelo predictivo','Seed evidence':'Evidencia de semillas','Pruning evidence':'Evidencia de poda','Research methodology':'Metodología de investigación',
    'Chart name':'Nombre del gráfico','My candidate view':'Mi vista de candidatos','Type':'Tipo','Scatter':'Dispersión','Line':'Línea','Bar':'Barras',
    'Parallel coordinates':'Coordenadas paralelas','Observed heatmap':'Mapa de calor observado','Observed 3D surface':'Superficie 3D observada',
    'X axis':'Eje X','Y axis':'Eje Y','Z / cell metric':'Z / métrica de celda','Grouping':'Agrupación','One point per trial':'Un punto por candidato',
    'Mean per exact X value + SD':'Media por valor X exacto + desviación típica','Reverse colours':'Invertir colores',
    'Include partial / pruned observations':'Incluir observaciones parciales / podadas','Additional Y metrics (2D)':'Métricas Y adicionales (2D)',
    'Save chart':'Guardar gráfico','Live preview · Save chart keeps this view for reopening.':'Vista previa inmediata · Guardar conserva la vista para volver a abrirla.',
    'Analyze':'Analizar','Choose metric…':'Elegir métrica…','By · add parameter…':'Por · añadir parámetro…','Compare with…':'Comparar con…',
    'Completed candidates':'Candidatos completos','Include partial / pruned':'Incluir parciales / podados','Explore observations':'Explorar observaciones',
    'Advanced visualization options':'Opciones avanzadas de visualización','Notes (optional)':'Notas (opcional)',
    'Notes / annotation for this saved view':'Notas de esta vista guardada','Saved view notes':'Notas de la vista guardada','Add Y metric…':'Añadir métrica Y…',
    'Visual 0–1 normalization only':'Solo normalización visual 0–1','Export research views':'Exportar vistas','Import research views':'Importar vistas',
    'Select two Trials to compare recorded metrics by category. The first selection is the displayed reference.':'Selecciona dos candidatos para comparar por categoría. El primero es la referencia mostrada.',
    'Filter comparison metrics / categories':'Filtrar métricas / categorías','Filter candidate comparison':'Filtrar comparación','Show all recorded metrics':'Mostrar todas las métricas',
    'Metric':'Métrica','Category':'Categoría','Difference (right − left)':'Diferencia (derecha − izquierda)','Direction':'Dirección',
    'Differences are descriptive. Unknown direction is not improvement; practical equivalence belongs only to the authored objective margin.':'Las diferencias son descriptivas. Una dirección desconocida no significa mejora; la equivalencia práctica depende del margen del objetivo declarado.',
    'Search Trials, states, parameters…':'Buscar candidatos, estados, parámetros…','Search Trials':'Buscar candidatos','All branches':'Todas las ramas',
    'All categories':'Todas las categorías','No matches. Try a shorter name or alias.':'Sin coincidencias. Prueba un nombre o alias más corto.',
    'Further structured details are available in the analysis JSON.':'Hay más detalles estructurados en el JSON de análisis.',
    'No final selection objective is available for this view.':'No hay un objetivo final de selección para esta vista.',
    'How to read parameter responses':'Cómo leer las respuestas de parámetros',
    'Each curve joins means at exact tested parameter values. The remaining parameters were not held constant, so this is an observed association, not a causal experiment.':'Cada curva une medias en valores exactos probados. Los demás parámetros no se mantuvieron constantes: es una asociación observada, no un experimento causal.',
    'Error bars show dispersion across candidates, not uncertainty over seeds. Hover for support counts and exact trial identities. Missing values are never replaced with zero.':'Las barras muestran dispersión entre candidatos, no incertidumbre entre semillas. Pasa el ratón para ver soporte e identidades. Los valores ausentes nunca se sustituyen por cero.',
    'Metric summaries retain their persisted aggregation. Selecting more metrics does not recompute an objective or choose a different best epoch.':'Los resúmenes conservan su agregación guardada. Seleccionar más métricas no recalcula un objetivo ni elige otra mejor época.',
    'Different or unknown units are displayed on labelled independent axes in the same chart. Comparing heights across those axes is not a comparison of scientific value.':'Unidades distintas o desconocidas se muestran en ejes independientes del mismo gráfico. Comparar sus alturas no compara su valor científico.',
    'How to read coverage':'Cómo leer la cobertura',
    'Coverage describes where observations exist, not how likely the optimum is to be there. Conditional parameters must be interpreted only in their active branch.':'La cobertura describe dónde hay observaciones, no la probabilidad de que allí esté el óptimo. Los parámetros condicionales se interpretan solo en su rama activa.',
    'Attempted and censored evidence can cover search locations without resolving complete response evidence. A successful Study can remain scientifically unresolved.':'Los intentos y las podas pueden cubrir posiciones sin resolver respuestas completas. Un estudio completado puede seguir sin resolver científicamente.',
    'Axes use recorded hyperparameter values, not epochs. Metrics are persisted trial/seed summaries, not new metric optima. Exact-value grouping averages trials equally, with other parameters uncontrolled. Heatmap/surface cells are observed means, never model predictions; untested combinations stay blank. Partial values are descriptive only. Saved views stay in this HTML’s browser preferences; they do not refit or control HPO.':'Los ejes usan hiperparámetros registrados, no épocas. Las métricas son resúmenes guardados, no nuevos óptimos. La agrupación pondera candidatos por igual sin controlar otros parámetros. Mapas y superficies muestran medias observadas, no predicciones; lo no probado queda vacío. Los parciales son descriptivos. Las vistas se guardan en este navegador; no reajustan ni controlan el HPO.',
    'No matching observations. See the explanation below.':'No hay observaciones coincidentes. Consulta la explicación debajo.',
    'Observed joint metric':'Métrica conjunta observada','Preview':'Vista previa','Exploratory view':'Vista exploratoria',
    'Candidate evidence':'Evidencia de candidatos','Persisted pairwise predictive response':'Respuesta predictiva conjunta guardada',
    'Model evidence':'Evidencia del modelo','No joint model evidence. Observed interactions remain available below.':'No hay evidencia de modelo conjunto. Las interacciones observadas siguen disponibles.',
    'Unknown units use separate axes, not assumed comparable scales.':'Las unidades desconocidas usan ejes separados, no escalas supuestamente comparables.',
    'Same-chart overlays use separate Y axes for different or unknown units.':'Las curvas comparten gráfico; unidades distintas o desconocidas usan ejes Y separados.',
    'Selected units':'Unidades seleccionadas','Default metric':'Métrica predeterminada',
    'More results':'Más resultados','Page':'Página','Show details':'Ver detalles',
    'Persisted scientific statements and authored names are shown in their original language.':'Las afirmaciones científicas guardadas y los nombres declarados se muestran en su idioma original.'
  };
  const fragments = {
    'Completed evidence only.':'Solo evidencia completa.',
    'Partial/censored evidence is descriptive only, not final selection.':'La evidencia parcial/censurada es descriptiva, no selección final.',
    'Each point is a trial summary; duplicate parameter values are retained.':'Cada punto resume un candidato; se conservan valores de parámetros repetidos.',
    'Equal-weight trial means; error bars are empirical SD across trials, not seed confidence intervals.':'Medias de candidatos con el mismo peso; las barras son desviación típica empírica, no intervalos de confianza de semillas.',
    'Different or unknown units use small multiples with independent Y scales.':'Unidades distintas o desconocidas usan paneles con escalas Y independientes.',
    'Cells average trial summaries; missing pairs stay blank, never zero or predicted.':'Las celdas promedian candidatos; los pares no probados quedan vacíos, nunca son cero ni predicciones.',
    'Use Explore to inspect current/best partial observations without inventing a final score.':'Usa Explorar para ver observaciones actuales/mejores sin inventar un score final.',
    'No trials have a recorded value for the selected X parameter.':'No hay candidatos con el parámetro X registrado.',
    'No recorded values match this metric and evidence filter. Try another metric or enable partial observations.':'No hay valores registrados para esta métrica y filtro. Prueba otra métrica o incluye parciales.',
    'Choose Current observed objective or Best observed objective and enable partial observations when needed.':'Elige el objetivo actual o el mejor observado e incluye parciales si es necesario.',
    '3D surfaces require numeric parameter axes. Use a heatmap or 3D scatter for categorical values.':'Las superficies 3D necesitan parámetros numéricos. Usa mapa de calor o dispersión 3D para categorías.',
    'Too many distinct value pairs for a readable grid. Use 3D scatter instead; no observations were discarded.':'Hay demasiados pares para una cuadrícula legible. Usa dispersión 3D; no se descartaron observaciones.',
    'No parameter conclusion is available yet.':'Aún no hay conclusiones sobre parámetros.',
    'Mean · ':'Media · ','Parameter · ':'Parámetro · ','Metric · ':'Métrica · ','Resource · ':'Recurso · ',
    'Analyze ':'Analizar ','Best observed objective (partial)':'Mejor objetivo observado (parcial)',
    'Best observed ':'Mejor observado ','Current observed ':'Actual observado ',' (partial)':' (parcial)',
    'Current observed objective':'Objetivo observado actual','Selection objective':'Objetivo de selección',
    'Composite selection score':'Score de selección compuesto','Trial ':'Candidato ',
    'trials for the primary metric':'candidatos para la métrica principal','metric(s)':'métrica(s)',
    'observed trials':'candidatos observados','hover a point for its exact values and identity.':'pasa sobre un punto para ver valores e identidad.',
    'observed pairs':'pares observados',' cells.':' celdas.','trials ·':'candidatos ·',
    'recorded metrics · all evidence remains available in Metrics & health.':'métricas registradas · toda la evidencia está disponible en Métricas y calidad.',
    'supported observations / relationships':'observaciones / relaciones con soporte',
    'Configured · ':'Declarada · ','Scientific status: ':'Estado científico: ','seed stability: ':'estabilidad entre semillas: ',
    'joint coverage: ':'cobertura conjunta: ','Categories · ':'Categorías · ',
    'No final selection objective is available for this view.':'No hay objetivo final de selección para esta vista.'
  };
  function t(text) {
    if (typeof text !== 'string') return text;
    if(language!=='es')return englishStatus[text]||text;
    const whitespace = text.match(/^(\s*)([\s\S]*?)(\s*)$/), value = whitespace[2].replace(/\s+/g,' ');
    if (es[value]) return whitespace[1] + es[value] + whitespace[3];
    let result = text;
    for (const [from,to] of Object.entries(fragments)) result = result.split(from).join(to);
    return result;
  }
  const originals = new WeakMap();
  function translateText(node) {
    const previous = originals.get(node), current = node.nodeValue;
    const source = previous && previous.rendered === current ? previous.source : current;
    const rendered = t(source); originals.set(node,{source,rendered});
    if (current !== rendered) node.nodeValue = rendered;
  }
  const attributes = new WeakMap();
  const chartLabels = new WeakMap();
  function localizeCharts(){
    const authored = new Set([services.data.objective_label,...services.data.parameters,
      ...Object.values(services.data.research?.metric_catalog?.metrics||{}).map(m=>m.label),
      ...(services.getPrefs().customCharts||[]).map(c=>c.name)]);
    document.querySelectorAll('.view:not([hidden]) .js-plotly-plot').forEach(chart=>{
      if(!chart.layout)return;const cache=chartLabels.get(chart)||{},patch={};
      for(const [key,current] of [['title.text',chart.layout.title?.text],
        ...Object.entries(chart.layout).filter(([name])=>/^[xy]axis\d*$/.test(name)).map(([name,axis])=>[name+'.title.text',axis.title?.text])]){
        if(!current)continue;const previous=cache[key],source=previous?.rendered===current?previous.source:current;
        const rendered=authored.has(source)?source:t(source);cache[key]={source,rendered};if(rendered!==current)patch[key]=rendered;
      }
      chartLabels.set(chart,cache);if(Object.keys(patch).length)Plotly.relayout(chart,patch);
    });
  }
  function apply(root) {
    if (root.nodeType === Node.TEXT_NODE) {
      if (!root.parentElement?.closest('script,style,pre,svg,.js-plotly-plot,[data-authored]')) translateText(root); return;
    }
    if (!(root instanceof Element) || root.closest('script,style,pre,svg,.js-plotly-plot,[data-authored]')) return;
    // Display translation must not change palette/type identities on text-valued options.
    if(root.tagName==='OPTION'&&!root.hasAttribute('value'))root.setAttribute('value',root.value);
    for (const name of ['placeholder','aria-label','title']) if (root.hasAttribute(name)) {
      const cache = attributes.get(root) || {}, current = root.getAttribute(name), previous = cache[name];
      const source = previous?.rendered === current ? previous.source : current, rendered = t(source);
      cache[name] = {source,rendered}; attributes.set(root,cache);
      if (current !== rendered) root.setAttribute(name,rendered);
    }
    for (const child of root.childNodes) apply(child);
  }
  function mount(s) {
    services = s; language = services.getPrefs().language === 'es' ? 'es' : 'en';
    const control = document.createElement('label'); control.className = 'language-control';
    control.append(document.createTextNode('Language / Idioma ')); const select = document.createElement('select');
    select.id = 'study-language'; select.setAttribute('aria-label','Language / Idioma');
    for (const [value,text] of [['en','English'],['es','Español']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = text; select.append(option);}
    select.value = language; control.append(select); document.querySelector('header').append(control);
    const change = () => {
      language = select.value; document.documentElement.lang = language; services.save({language});
      document.dispatchEvent(new Event('study-language-changed')); apply(document.body);
      // Rerender only visible charts; embedded observations are never modified.
      services.refreshVisible?.();
      requestAnimationFrame(localizeCharts);
    };
    select.onchange = change; change();
    document.addEventListener('study-view-changed',()=>requestAnimationFrame(localizeCharts));
    const pending = new Set(); let frame;
    const observer = new MutationObserver(records => {
      for (const record of records) {
        if (record.target.parentElement?.closest('.js-plotly-plot,svg,pre')) continue;
        if (record.type === 'characterData') pending.add(record.target);
        else for (const node of record.addedNodes) pending.add(node);
      }
      if (!pending.size || frame) return;
      frame = requestAnimationFrame(() => {frame = null; const nodes = [...pending]; pending.clear(); nodes.forEach(apply);});
    });
    observer.observe(document.body,{subtree:true,childList:true,characterData:true});
  }
  return {mount,t};
})();
