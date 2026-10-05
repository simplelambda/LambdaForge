/* Useful entry points into persisted evidence, never another statistical analysis. */
window.LambdaForgeStudyOverview = function (services) {
  const {data, config, baseLayout} = services;
  const el = id => document.getElementById(id);
  const t = text => window.LambdaForgeStudyLocale.t(text);
  const make = (tag, text, cls) => {const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n;};
  const finite = n => typeof n === 'number' && Number.isFinite(n);
  const fmt = n => finite(n) ? n.toLocaleString(document.documentElement.lang, {maximumSignificantDigits:4}) : t('Not recorded');
  const activate = id => document.querySelector(`.tab[data-target="${id}"]`)?.click();
  const condition = rule => Object.entries(rule.when || {}).map(([name, value]) =>
    name + (value && typeof value === 'object' ? ' ∈ ' + JSON.stringify(value.in) : ' = ' + JSON.stringify(value))).join(' ∧ ');
  const metric = name => data.research?.metric_catalog?.metrics?.[name]?.label || name;
  function help(title, paragraphs, detail) {
    el('research-detail-title').textContent = t(title); const body = el('research-detail-body'); body.replaceChildren();
    for (const text of paragraphs) body.append(make('p', t(text), 'study-local-help'));
    if (detail) {const extra = make('details'), summary = make('summary', t('Persisted diagnostics'));
      const pre = make('pre', JSON.stringify(detail, null, 2), 'raw'); extra.append(summary, pre); body.append(extra);}
    el('research-detail').showModal();
  }
  function carousel(target, cards) {
    const area = el(target); area.replaceChildren();
    if (!cards.length) {area.append(make('p', t('No supported conclusion yet. Inspect coverage or the recorded observations.'), 'note')); return;}
    const strip = make('div', undefined, 'card-carousel'), controls = make('div', undefined, 'carousel-controls');
    const previous = make('button', '← ' + t('Previous')), next = make('button', t('Next') + ' →'), counter = make('small');
    strip.tabIndex = 0; strip.setAttribute('aria-label', t('Interpretation')); strip.append(...cards);
    const update = () => {const step = cards[0].offsetWidth + 14;
      counter.textContent = `${Math.min(cards.length, 1 + Math.round(strip.scrollLeft / step))} / ${cards.length}`;
      previous.disabled = strip.scrollLeft < 1; next.disabled = strip.scrollLeft >= strip.scrollWidth - strip.clientWidth - 2;};
    previous.onclick = () => strip.scrollBy({left: -strip.clientWidth, behavior:'smooth'});
    next.onclick = () => strip.scrollBy({left: strip.clientWidth, behavior:'smooth'});
    strip.onscroll = update; controls.append(previous, counter, next); area.append(controls, strip);
    requestAnimationFrame(update);
  }
  function interpretation() {
    const questions = [...data.parameter_questions || []].sort((a,b) =>
      (b.descriptive_stability || 0) - (a.descriptive_stability || 0));
    carousel('overview-interpretation', questions.map(q => {
      const card = make('article', undefined, 'finding');
      card.append(make('span', t(q.conclusion_kind || 'UNRESOLVED'), 'badge'), make('h3', q.parameter));
      const conclusion=q.exact_conclusion||{};
      const descriptions={CONTEXT_DEPENDENT:'The response depends on other parameters.',
        PRACTICALLY_EQUIVALENT:'The supported values are equivalent within the authored practical margin.',
        FLAT:'No stable tendency is visible; this does not establish equivalence.',
        PREFERRED:'The persisted conclusion supports a preference among the displayed values.',
        PREFERRED_REGION:'The persisted conclusion supports a preference among the displayed values.',
        WEAK_PREFERENCE:'A possible preference needs more evidence.',
        UNRESOLVED:'The available evidence does not resolve this question.'};
      card.append(make('p',t(descriptions[q.conclusion_kind]||'Inspect the exact conclusion and its supporting values.')));
      if(conclusion.values?.length)card.append(make('p',t('Conclusion values')+': '+conclusion.values.map(value=>JSON.stringify(value)).join(', ')));
      const direct = q.response_support?.direct || [];
      card.append(make('p', t('Direct evidence at values') + ': ' + (direct.length ? direct.join(', ') : t('Not recorded'))),
        make('p', t('Conclusion stability') + ': ' + fmt(100 * (q.descriptive_stability || 0)) + '%', 'muted'));
      if (q.missing_evidence?.length) card.append(make('p', t('More evidence is needed') + ` (${q.missing_evidence.length})`, 'muted'));
      // Keep the exact persisted statement accessible without repeating technical boilerplate.
      const details = make('button', t('Conclusion details')); details.onclick = () => help(q.parameter,
        [q.summary || 'The question remains unresolved.'], q);
      const view = make('button', t('View response')); view.onclick = () => parameter(q.parameter);
      const actions = make('div', undefined, 'tools'); actions.append(view, details); card.append(actions); return card;
    }));
  }
  function parameter(name, selected) {
    activate('study-parameters'); const input = el('parameter-select'); input.value = name;
    if (selected) {el('parameter-metric').value = selected === 'selection_objective' ? '__selection__' : selected;
      services.save({parameterMetrics:[]});}
    input.dispatchEvent(new Event('change', {bubbles:true}));
    document.dispatchEvent(new Event('research-controls-changed'));
  }
  function summary() {
    const evidence = data.evidence || {}, winner = evidence.winner || {};
    const winning = winner.confirmed_winner || winner.screening_winner;
    const area = el('overview-summary'); area.className = 'overview-summary'; area.replaceChildren();
    const fields = [
      ['Selection metric', data.objective_label, 'Only the authored objective governs selection. Other metrics are diagnostics.'],
      ['Conclusion', t(evidence.scientific_status || 'UNRESOLVED'), 'Execution success does not guarantee scientific certainty.'],
      ['Leading trial', winning?.trial ? 'Trial ' + winning.trial : t('Not recorded'),
        winner.confirmed_winner ? 'Confirmed on fresh seeds.' : 'Screening leader; confirmation may still be pending.'],
      ['Evidence source', t(evidence.source?.status || 'unknown'), 'This report is a snapshot, not a live controller.']
    ];
    for (const [title, value, description] of fields) {const card = make('article', undefined, 'card');
      card.append(make('small', t(title)), make('strong', value), make('p', t(description))); area.append(card);}
  }
  function progress() {
    const states = {Completed:0, Pruned:0, Active:0, Failed:0, Other:0};
    for (const c of data.candidates) {
      const runs = c.runs || [], runStates = runs.map(r => r.state);
      if (c.censored_observations || runStates.includes('pruned')) states.Pruned++;
      else if (finite(c.mean)) states.Completed++;
      else if (runStates.some(s => ['running','staging','preparing','pending','queued'].includes(s))) states.Active++;
      else if (runStates.includes('failed') || c.state === 'failed') states.Failed++;
      else states.Other++;
    }
    const layout = baseLayout(t('Candidate states'), '', t('Candidates')); layout.height = 350;
    layout.margin = {l:55,r:25,t:50,b:50}; layout.yaxis.dtick = Math.max(1, Math.ceil(data.candidates.length / 6));
    Plotly.react(el('overview-states'), [{type:'bar', x:Object.keys(states).map(t), y:Object.values(states),
      text:Object.values(states).map(String),textposition:'auto',cliponaxis:false,
      marker:{color:['#56d364','#ffa657','#58a6ff','#ff7b72','#8b949e']},
      hovertemplate:'%{x}: %{y}<extra></extra>'}], layout, config);
    // A candidate containing censored Runs remains censored, never a fabricated final mean.
    const note = el('overview-states-note') || make('p', undefined, 'note'); note.id = 'overview-states-note';
    note.textContent = t('Counts refer to candidates, not seeds. Pruned means incomplete performance evidence, not a software failure.');
    el('overview-states').after(note);
  }
  function importance() {
    const schema = data.parameter_space || {}, items = [...new Set([...Object.keys(schema),...Object.keys(data.importance || {})])]
      .map(name=>[name,data.importance?.[name] || {}]);
    const unconditional = items.filter(([name,score]) => !schema[name]?.when && finite(score.importance));
    const graph = el('overview-importance')?.querySelector('.js-plotly-plot');
    if (graph) {
      graph.style.height='350px';
      const layout = baseLayout(t('Predictive association · unconditional parameters'), t('Parameter'), t('Model association score'));layout.height=350;
      if(!unconditional.some(([,v])=>finite(v.importance)))layout.annotations=[{text:t('Not recorded'),xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}];
      Plotly.react(graph, [{type:'bar',x:unconditional.map(([name])=>name), y:unconditional.map(([,v])=>v.importance ?? null),
        customdata:unconditional.map(([,v])=>[v.support ?? null,t(v.reliability || 'unrated')]),marker:{color:'#bc8cff'},
        hovertemplate:'%{x}<br>%{y:.4g}<br>'+t('Support')+'=%{customdata[0]}<br>'+t('Reliability')+'=%{customdata[1]}<extra></extra>'}], layout, config);
    }
    const conditional = items.filter(([name]) => schema[name]?.when);
    el('overview-conditional').hidden = !conditional.length; const target = el('overview-conditional-table'); target.replaceChildren();
    const table = make('table'), header = make('tr');
    ['Parameter','Active when','Observed active / total','Model score','Reliability','Open'].forEach(v=>header.append(make('th',t(v))));
    table.append(header);
    for (const [name, score] of conditional) {
      const observed = data.candidates.filter(c=>Object.hasOwn(c.parameters || {},name));
      const complete = observed.filter(c=>finite(c.mean) && !c.censored_observations).length;
      const row = make('tr'); [name,condition(schema[name]),`${observed.length} / ${data.candidates.length} · ${complete} ${t('Completed')}`,
        fmt(score.importance),t(score.reliability || 'unrated')].forEach(v=>row.append(make('td',v)));
      const cell = make('td'), button = make('button',t('View response')); button.onclick = () => parameter(name); cell.append(button); row.append(cell); table.append(row);
    }
    target.append(make('p',t('These scores may mix branch activation with within-branch variation. They do not measure how much a parameter matters inside that branch or justify choosing the branch.'),'overview-warning'),table);
  }
  function suggestions() {
    const target = el('overview-suggestions'); target.className = 'suggestion-grid'; target.replaceChildren();
    const choices = [['Where is evidence missing?', 'Coverage distinguishes tested values from completed response evidence.', () => activate('study-coverage')],
      ['Compare candidates', 'Inspect two candidates and their recorded metrics, not just their ranking.', () => activate('study-trials')]];
    const response = data.parameter_questions?.[0]?.parameter || data.parameters[0];
    if (response) choices.push(['Inspect a parameter response', 'See recorded means, dispersion and support; other parameters are uncontrolled.', () => parameter(response)]);
    const f = (data.research?.inbox_findings || []).find(f=>f.recommended_view);
    if (f) choices.push(['Inspect a supported relationship', f.title, () => {activate('study-custom'); services.customCharts.open(f.recommended_view);}]);
    for (const [title, description, open] of choices) {const card = make('article',undefined,'finding');
      const button = make('button',t('Open suggested view')); button.onclick = open;
      card.append(make('h3',t(title)),make('p',t(description)),button); target.append(card);}
  }
  function render() {summary();progress();importance();interpretation();suggestions();}
  function mount() {
    render(); el('interpretation-help').onclick = () => help('How to interpret this evidence', [
      'Stability measures how often the displayed conclusion survives resampling the available evidence. It is not the probability that the conclusion is true.',
      'Direct support comes from completed observations; censored support comes from pruned Runs; predictive support comes from the persisted model. They are not interchangeable.',
      'Conditional scores may reflect the choice of parent branch, not the effect of changing the parameter within that branch.',
      'Model variation scores need coverage and validated predictive reliability. They are not additive percentages of responsibility.'
    ], {seed_noise:data.scientific?.seed_noise_model, stability:data.parameter_questions?.map(q=>({parameter:q.parameter,...q.stability_diagnostics}))});
    for(const [view,title,paragraphs] of [
      ['#study-parameters','How to read parameter responses',[
        'Each curve joins means at exact tested parameter values. The remaining parameters were not held constant, so this is an observed association, not a causal experiment.',
        'Error bars show dispersion across candidates, not uncertainty over seeds. Hover for support counts and exact trial identities. Missing values are never replaced with zero.',
        'Metric summaries retain their persisted aggregation. Selecting more metrics does not recompute an objective or choose a different best epoch.',
        'Automatic scales group substantially overlapping ranges without assuming scientific equivalence. Known different units stay separate. Choose independent/shared scales and combined/separate panels; do not compare heights across independent axes.']],
      ['#study-coverage','How to read coverage',[
        'Coverage describes where observations exist, not how likely the optimum is to be there. Conditional parameters must be interpreted only in their active branch.',
        'Attempted and censored evidence can cover search locations without resolving complete response evidence. A successful Study can remain scientifically unresolved.']]
    ]){const heading=document.querySelector(view+' h2'),button=make('button','ⓘ','info-button');
      button.setAttribute('aria-label',title);button.onclick=()=>help(title,paragraphs);heading?.append(' ',button);}
    const note=make('p',t('Persisted scientific statements and authored names are shown in their original language.'),'note');
    document.querySelector('.shell').append(note);
    document.addEventListener('study-language-changed', render);
  }
  return {mount,carousel};
};
