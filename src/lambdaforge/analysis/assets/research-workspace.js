/* Offline research presentation. Every finding/profile is computed and persisted in Python. */
window.LambdaForgeResearchWorkspace = function (services) {
  const {data, customCharts, save, getPrefs, config, baseLayout} = services;
  const research = data.research || {}, catalog = research.metric_catalog?.metrics || {};
  const profiles = research.metric_profiles || {}, findings = research.findings || [];
  const el = id => document.getElementById(id);
  const make = (tag, text, cls) => {const n = document.createElement(tag); if(text !== undefined)n.textContent = text;
    if(cls)n.className = cls; return n;};
  const fmt = value => value === null || value === undefined ? 'unknown'
    : typeof value === 'number' ? value.toLocaleString(undefined, {maximumSignificantDigits: 4}) : String(value);
  const key = name => name === 'selection_objective' ? '__selection__' : name;
  const metricName = name => name === '__selection__' ? 'selection_objective' : name;
  const label = name => catalog[metricName(name)]?.label || name;
  const searchable = m => [m.name, m.label, m.description, m.category, ...(m.aliases || []), ...(m.tags || [])].join(' ').toLowerCase();
  // Presentation-only fuzzy matching: words first, then an ordered subsequence.
  const matches = (text, words) => words.every(word => {
    text=text.toLowerCase();if(text.includes(word))return true;
    let index=0;for(const character of text)if(character===word[index])index++;
    return word.length>=3&&index===word.length;
  });
  let picker = null;
  function tab(id) {const button = document.querySelector(`.tab[data-target="${id}"]`); if(button)button.click();}
  function explore(spec) {tab('study-custom'); customCharts.open({kind:'scatter', partial:false, aggregate:'points', palette:'Accessible', ...spec});}
  function recent(name) {save({researchRecent:[name, ...(getPrefs().researchRecent || []).filter(n=>n!==name)].slice(0,12)});}
  function describe(value, depth=0) {
    if(value===null || typeof value!=='object')return make('span',fmt(value));
    if(depth>4)return make('span','Further structured details are available in the analysis JSON.','muted');
    if(Array.isArray(value)){const list=make('ul');for(const item of value.slice(0,50)){const row=make('li');row.append(describe(item,depth+1));list.append(row);}
      if(value.length>50)list.append(make('li',`${value.length-50} more entries in the analysis JSON.`));return list;}
    const list=make('dl');for(const [name,item] of Object.entries(value)){list.append(make('dt',name.replaceAll('_',' ')));const description=make('dd');description.append(describe(item,depth+1));list.append(description);}return list;
  }
  function detail(title, sections, open) {
    el('research-detail-title').textContent = title;const body=el('research-detail-body');body.replaceChildren();
    for(const [heading, values] of sections){body.append(make('h3',heading));
      body.append(describe(values || {}));}
    if(open){const button=make('button','Explore these observations');button.onclick=()=>{el('research-detail').close();explore(open)};body.append(button);}
    el('research-detail').showModal();
  }
  function findingCard(finding) {const card=make('article',undefined,'finding');card.append(make('span',finding.origin+' · '+finding.reliability.status,'badge'),
    make('h3',finding.title),make('p',finding.summary));const button=make('button','Inspect');
    button.onclick=()=>detail(finding.title,[['Evidence',finding.evidence || {support:finding.support}],['Reliability',finding.reliability],
      ['Ranking components',finding.ranking_components || finding.reliability.components || {}],['Limitations',{caveats:finding.caveats}]],finding.recommended_view);card.append(button);return card;}
  function metrics() {const query=el('research-metric-search').value.toLowerCase(), category=el('research-category').value,
    all=el('research-show-all').checked,sort=el('research-sort').value;const body=el('research-metrics-body');body.replaceChildren();
    const names=Object.keys(catalog).filter(name=>(all || ((profiles[name]?.informative ?? true) && catalog[name].visibility!=='hidden'))
      && (!category || catalog[name].category.startsWith(category)) && query.split(/\s+/).every(word=>searchable(catalog[name]).includes(word)));
    names.sort((a,b)=>sort==='name'?label(a).localeCompare(label(b)):sort==='coverage'?(profiles[b]?.coverage||0)-(profiles[a]?.coverage||0):
      sort==='spread'?(profiles[b]?.robust_spread||0)-(profiles[a]?.robust_spread||0):(catalog[b].priority||0)-(catalog[a].priority||0)||a.localeCompare(b));
    for(const name of names){const m=catalog[name],p=profiles[name]||{},row=make('tr'),meaning=make('td');meaning.append(make('strong',m.label),make('div',name,'muted'),make('div',m.description,'muted'));row.append(meaning);
      const fields=[m.category+' · '+m.split,m.unit+' · '+m.direction,`${p.finite_candidates??'?'}/${p.total_candidates??'?'} candidates · ${p.seed_support??'?'} seeds`,
        `${fmt(p.min)} → ${fmt(p.max)} · SD ${fmt(p.sd)}`,`${p.reason||'unprofiled'} · ${m.aggregation} ${p.warnings?.join(' · ')||''}`];
      fields.forEach(v=>row.append(make('td',v)));const action=make('td'),button=make('button','Inspect');button.onclick=()=>detail(m.label,
        [['Meaning',m],['Profile',p]],{x:'trial',y:'metric:'+key(name)});action.append(button);row.append(action);body.append(row);}
    if(!names.length){const row=make('tr'),cell=make('td','No matching informative metrics. Use Show all to inspect missing or constant evidence.');cell.colSpan=7;row.append(cell);body.append(row);}
    save({researchMetricQuery:query,researchShowAll:all,researchCategory:category,researchSort:sort});
  }
  function family(name) {const f=research.families?.[name];if(!f)return;
    const dimensions=Object.keys(f.dimensions||{}),xname=dimensions.find(n=>f.dimensions[n].kind!=='categorical')||dimensions[0],
      other=dimensions.filter(n=>n!==xname),groups=new Map();
    for(const point of f.points||[]){const group=other.map(d=>d+'='+point.dimensions[d]).join(' · ')||name;
      const list=groups.get(group)||[];list.push(point);groups.set(group,list);}
    const order=f.dimensions[xname]?.values||[],traces=[];
    for(const [group,points] of groups){points.sort((a,b)=>order.indexOf(a.dimensions[xname])-order.indexOf(b.dimensions[xname]));
      traces.push({type:'scatter',mode:'lines+markers',name:group,x:points.map(p=>String(p.dimensions[xname])),y:points.map(p=>p.mean),
        customdata:points.map(p=>[p.metric,p.support,p.sd]),connectgaps:false,hovertemplate:'%{customdata[0]}<br>value=%{y}<br>candidates=%{customdata[1]}<br>SD=%{customdata[2]}<extra></extra>'});}
    const layout=baseLayout(f.label||name,xname,'Observed family means');layout.xaxis.type='category';
    Plotly.react(el('research-family-plot'),traces,layout,config);el('research-family-status').textContent=f.interpretation;
    tab('study-metrics');
  }
  function search() {const query=el('research-global-query').value.toLowerCase(),box=el('research-search-results');box.replaceChildren();
    const words=query.split(/\s+/).filter(Boolean),match=text=>matches(text,words);let items=[];
    const favorites=getPrefs().researchFavorites||[],recents=getPrefs().researchRecent||[];
    for(const [name,m] of Object.entries(catalog)){if(match(searchable(m))){items.push({name,title:m.label+' · '+m.category+' · '+(profiles[name]?.reason||m.aggregation),
      description:m.description||name,rank:favorites.includes(name)?2:recents.includes(name)?1:0,action:()=>{recent(name);if(picker){picker(name);picker=null;}else explore({x:'trial',y:'metric:'+key(name)});}});}}
    if(!picker){for(const name of data.parameters||[])if(match(name))items.push({title:'Parameter · '+name,description:'Explore observed metric response',action:()=>explore({x:'param:'+name,y:'metric:__selection__'})});
      for(const [name,f] of Object.entries(research.families||{}))if(match(name+' '+(f.label||'')))items.push({title:'Family · '+(f.label||name),description:f.description||'',action:()=>family(name)});
      for(const f of findings)if(match(f.title+' '+f.summary))items.push({title:'Finding · '+f.title,description:f.summary,action:()=>detail(f.title,[['Evidence',f.evidence],['Reliability',f.reliability],['Limitations',{caveats:f.caveats}]],f.recommended_view)});
      for(const c of data.candidates||[])if(match('trial '+c.trial))items.push({title:'Trial '+c.trial,description:c.state||'Observed candidate',action:()=>{tab('study-trials');const rows=document.querySelectorAll('#study-trials tbody tr');rows[[...data.candidates].indexOf(c)]?.scrollIntoView({block:'center'});}});
      for(const c of getPrefs().customCharts||[])if(match(c.name))items.push({title:'Saved view · '+c.name,description:c.notes||'',action:()=>{tab('study-custom');customCharts.open(c)}});}
    items.sort((a,b)=>(b.rank||0)-(a.rank||0)||a.title.localeCompare(b.title));
    for(const item of items.slice(0,80)){const row=make('div',undefined,'research-search-result'),button=make('button',item.title);button.onclick=()=>{el('research-search').close();item.action();};row.append(button,make('p',item.description,'muted'));
      if(item.name){const star=make('button',favorites.includes(item.name)?'★':'☆');star.title='Toggle favourite';star.onclick=()=>{save({researchFavorites:favorites.includes(item.name)?favorites.filter(n=>n!==item.name):[...favorites,item.name]});search();};row.append(star);}box.append(row);}
    if(!items.length)box.append(make('p','No matches. Try a shorter name or alias.'));
  }
  function choose(callback) {picker=callback;el('research-global-query').value='';search();el('research-search').showModal();el('research-global-query').focus();}
  function enhanceSelector(id,prefix) {const selector=el(id);if(!selector)return;const button=make('button','Search metrics…');
    button.onclick=()=>choose(name=>{const value=prefix+key(name);if(![...selector.options].some(o=>o.value===value)){const o=make('option',label(name));o.value=value;selector.append(o);}
      selector.value=value;selector.dispatchEvent(new Event('change',{bubbles:true}));});selector.after(button);}
  function mount() {
    const styles=make('style');styles.textContent=`dialog{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:14px;width:min(950px,95vw);max-height:85vh;overflow:auto;padding:22px}dialog::backdrop{background:#0009}dialog p{overflow-wrap:anywhere}.research-search-result{padding:10px;border-bottom:1px solid var(--line)}#research-global-query{width:100%;padding:14px;font:inherit;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:8px}.research-search-result p{margin:6px 0}.tools input[type=search]{padding:10px;background:var(--bg);color:var(--text);border:1px solid var(--line)}#research-metrics-body td{white-space:normal;min-width:140px}#research-detail h3{color:var(--accent)}.explore-simple{border-bottom:1px solid var(--line);background:var(--panel2)}.explore-simple button{margin:5px}`;document.head.append(styles);
    const count=Object.keys(profiles).length,informative=Object.values(profiles).filter(p=>p.informative).length,
      warnings=Object.values(profiles).filter(p=>p.warnings?.length).length;
    for(const [title,value] of [['Recorded metrics',count],['Informative',informative],['Health warnings',warnings],['Exploratory findings',findings.length]]){
      const card=make('div',undefined,'card');card.append(make('small',title),make('strong',String(value)));el('research-health').append(card);}
    const status=data.research_status||{};el('research-health').after(make('p',
      `Scientific status: ${status.scientific_status||'unavailable'} · seed stability: ${status.seed_analysis?.status||'unavailable'} · joint coverage: ${status.coverage?.quality||'unavailable'}`,'note'));
    (research.inbox_findings||findings.slice(0,8)).forEach(f=>el('research-inbox').append(findingCard(f)));
    if(!findings.length)el('research-inbox').append(make('p','No supported relationships yet. Explore the recorded evidence or inspect metric health; missing support is not a negative result.','empty'));
    findings.forEach(f=>el('research-all-findings').append(findingCard(f)));
    for(const question of research.questions||[])el('research-questions').append(make('p',question.id+' · '+question.kind+' · '+question.status+' · '+(question.reason||'')));
    const category=el('research-category');[...new Set(Object.values(catalog).map(m=>m.category))].sort().forEach(name=>{const option=make('option',name);option.value=name;category.append(option);});
    const prefs=getPrefs();el('research-metric-search').value=prefs.researchMetricQuery||'';el('research-show-all').checked=!!prefs.researchShowAll;category.value=prefs.researchCategory||'';el('research-sort').value=prefs.researchSort||'priority';
    ['research-metric-search','research-show-all','research-category','research-sort'].forEach(id=>el(id).addEventListener('input',metrics));metrics();
    for(const group of research.redundancy_groups||[]){const details=make('details'),summary=make('summary',`${group.members.length} redundant metrics · representative: ${label(group.representative)}`);
      details.append(summary,make('p',group.method,'note'));for(const name of group.members){const button=make('button',label(name));button.onclick=()=>detail(label(name),[['Meaning',catalog[name]],['Profile',profiles[name]]],{x:'trial',y:'metric:'+key(name)});details.append(button);}el('research-redundancy').append(details);}
    for(const [name,f] of Object.entries(research.families||{})){const button=make('button',f.label||name);button.onclick=()=>family(name);el('research-family-list').append(button);}
    if(!Object.keys(research.families||{}).length){el('research-family-list').append(make('p','No families declared. Ordinary metrics remain available.','muted'));el('research-family-plot').style.display='none';}
    el('research-detail-close').onclick=()=>el('research-detail').close();el('research-search-close').onclick=()=>{el('research-search').close();picker=null};
    el('research-search-open').onclick=()=>{picker=null;choose(null)};el('research-global-query').oninput=search;
    el('research-global-query').onkeydown=event=>{if(event.key==='ArrowDown'){event.preventDefault();el('research-search-results').querySelector('button')?.focus()}};
    document.addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='k'){event.preventDefault();picker=null;choose(null)}});
    ['trial-metric','parameter-metric'].forEach(id=>enhanceSelector(id,''));['study-chart-x','study-chart-y','study-chart-z'].forEach(id=>enhanceSelector(id,'metric:'));
    const addParameter=make('button','Add comparison metric…'),resetParameter=make('button','Single metric');addParameter.onclick=()=>choose(name=>{save({parameterMetrics:[...new Set([...(getPrefs().parameterMetrics||[]),'metric:'+key(name)])]});customCharts.updateParameter();});
    resetParameter.onclick=()=>{save({parameterMetrics:[]});customCharts.updateParameter();};el('parameter-metric').closest('.tools').append(addParameter,resetParameter);
    const add=make('button','Add Y metric…');add.onclick=()=>choose(name=>{const fields=[...el('study-chart-extra-metrics').querySelectorAll('input:checked')].map(i=>i.value);customCharts.open({metrics:[...new Set([...fields,'metric:'+key(name)])]});});el('study-chart-extra-control').append(add);
    const builder=el('study-chart-name').closest('.custom-builder'), advanced=make('details');advanced.append(make('summary','Advanced visualization options'));
    const simple=make('div',undefined,'tools explore-simple');simple.append(make('strong','Analyze'),make('span','Choose metrics and a parameter; changes preview immediately.'));
    const metric=make('button','Choose metric…');metric.onclick=()=>choose(name=>customCharts.open({kind:'scatter',x:el('study-chart-x').value,y:'metric:'+key(name)}));simple.append(metric);
    const notes=make('input');notes.id='research-view-notes';notes.placeholder='Notes / annotation for this saved view';notes.setAttribute('aria-label','Saved view notes');simple.append(notes);
    builder.before(simple);for(const name of ['kind','z-control','aggregate','palette','reverse','partial']){const control=el('study-chart-'+name),wrapper=control?.closest('label');if(wrapper)advanced.append(wrapper)}builder.append(advanced);
    const normalization=make('label'),normalize=make('input');normalize.type='checkbox';normalize.id='study-chart-normalize';normalization.append(normalize,document.createTextNode('Visual 0–1 normalization only'));
    advanced.append(normalization);normalize.onchange=()=>el('study-chart-y').dispatchEvent(new Event('change'));
    const active=(getPrefs().customCharts||[]).find(view=>view.id===getPrefs().customChartId);if(active){normalize.checked=!!active.normalize;notes.value=active.notes||'';customCharts.open(active);}
    const explain=make('p',(research.methodology?.limitations||[]).join(' '),'note');el('study-findings').prepend(explain);
    const exportButton=make('button','Export research views'),importButton=make('button','Import research views'),input=make('input');input.type='file';input.accept='.json,application/json';input.hidden=true;
    const views=el('study-saved-charts');views.before(exportButton,importButton,input);
    exportButton.onclick=()=>{const blob=new Blob([JSON.stringify({views_version:1,views:getPrefs().customCharts||[]},null,2)],{type:'application/json'}),link=make('a');link.href=URL.createObjectURL(blob);link.download='research-views.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)};
    importButton.onclick=()=>input.click();input.onchange=async()=>{try{if(input.files[0].size>1048576)throw Error('View file exceeds 1 MiB');const value=JSON.parse(await input.files[0].text());customCharts.importViews(value);}
      catch(error){detail('Could not import views',[['Error',{message:error.message}]]);}finally{input.value=''}};
    window.lfResearchComparison=(selected,area)=>{area.style.display='block';if(selected.length!==2){area.append(make('p','Select two Trials to compare recorded metrics by category. The first selection is the displayed reference.'));return;}
      const left=data.candidates.find(c=>Number(c.trial)===selected[0]),right=data.candidates.find(c=>Number(c.trial)===selected[1]),
        leftValues=research.rows?.find(r=>r.trial===left.trial)?.values||{},rightValues=research.rows?.find(r=>r.trial===right.trial)?.values||{};
      const filter=make('input');filter.type='search';filter.placeholder='Filter comparison metrics / categories';filter.setAttribute('aria-label','Filter candidate comparison');area.append(filter);
      const wrap=make('div',undefined,'table-wrap'),table=make('table'),head=make('tr');['Metric','Category','Trial '+left.trial,'Trial '+right.trial,'Difference (right − left)','Direction'].forEach(text=>head.append(make('th',text)));table.append(head);wrap.append(table);area.append(wrap);
      const render=()=>{table.querySelectorAll('tr:not(:first-child)').forEach(r=>r.remove());const query=filter.value.toLowerCase();
        Object.entries(catalog).sort(([a,am],[b,bm])=>(bm.priority||0)-(am.priority||0)||am.category.localeCompare(bm.category)||a.localeCompare(b)).forEach(([name,m])=>{if(!searchable(m).includes(query))return;
          const a=leftValues[name],b=rightValues[name],delta=typeof a==='number'&&typeof b==='number'?b-a:null,row=make('tr');
          [m.label,m.category,fmt(a),fmt(b),fmt(delta),m.direction].forEach(value=>row.append(make('td',value)));table.append(row);});};filter.oninput=render;render();
      area.append(make('p','Differences are descriptive. Unknown direction is not improvement; practical equivalence belongs only to the authored objective margin.','note'));
    };
    const trialTools=document.querySelector('#study-trials .tools'),trialSearch=make('input');trialSearch.type='search';trialSearch.placeholder='Search Trials, states, parameters…';trialSearch.setAttribute('aria-label','Search Trials');trialTools.prepend(trialSearch);
    trialSearch.oninput=()=>document.querySelectorAll('#study-trials tbody tr').forEach(row=>row.hidden=!row.textContent.toLowerCase().includes(trialSearch.value.toLowerCase()));
    if(!getPrefs().tab)tab('study-research');
  }
  return {mount};
};
