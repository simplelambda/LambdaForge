/* Offline observed-evidence charts. These views never fit a model or alter HPO. */
window.LambdaForgeStudyCharts = function (services) {
  const {data, node, config, baseLayout, studyField, fieldLabel, metricValue,
    save, getPrefs, palettes} = services;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const present = value => value !== null && value !== undefined;
  const categorical = values => !values.every(finite);
  const label = value => typeof value === 'string' ? value : JSON.stringify(value);
  const compare = (a, b) => finite(a) && finite(b) ? a - b : label(a).localeCompare(label(b));
  const unique = values => [...new Map(values.map(value => [JSON.stringify(value), value])).values()].sort(compare);
  const mean = values => values.reduce((total, value) => total + value, 0) / values.length;
  const sd = values => {if (values.length < 2) return null; const average = mean(values);
    return Math.sqrt(values.reduce((total, value) => total + (value - average) ** 2, 0) / (values.length - 1));};
  const id = name => document.getElementById('study-chart-' + name);
  const status = (message, target = 'study-chart-status') => { document.getElementById(target).textContent = message; };
  const isJoint = kind => ['scatter3d', 'heatmap', 'surface'].includes(kind);
  const isComplete = candidate => finite(candidate.mean) || candidate.n > 0;
  const isPartial = candidate => !isComplete(candidate);
  const domain = (name, observed) => {
    const rule = name.startsWith('param:') ? data.parameter_space?.[name.slice(6)] : null;
    return unique([...(Array.isArray(rule?.values) ? rule.values : []), ...observed]);
  };
  let selectedParameters;

  function comparableRuns(candidate, partial) {
    const runs = (candidate.runs || []).filter(run => run.phase !== 'confirmation'
      && (run.state === 'succeeded' || (partial && ['running', 'pruned', 'paused'].includes(run.state))));
    // Never silently average different fidelity rungs in a descriptive chart.
    const target = Math.max(0, ...runs.map(run => run.fidelity?.target || 0));
    return runs.filter(run => (run.fidelity?.target || 0) === target);
  }

  function observedMetric(candidate, metric, partial) {
    if (metric === '__best_observed__') return candidate.best_observed_objective;
    const runs = comparableRuns(candidate, partial);
    if (metric === '__current_observed__') {
      const values = runs.map(run => run.current_observed_objective).filter(finite);
      return values.length ? mean(values) : null;
    }
    const value = metricValue(candidate, metric);
    if (finite(value)) return value;
    // Missing final selection is NEVER replaced with a censored best observation.
    if (metric === '__selection__') return null;
    const values = runs.map(run => run.metrics?.[metric]).filter(finite);
    return values.length ? mean(values) : null;
  }

  function field(candidate, name, partial) {
    if (name.startsWith('metric:')) return observedMetric(candidate, name.slice(7), partial);
    return studyField(candidate, name);
  }

  function rowsFor(spec, yField) {
    return data.candidates.filter(candidate => isComplete(candidate) || spec.partial)
      .map(candidate => ({candidate, x: field(candidate, spec.x, spec.partial),
        y: field(candidate, yField, spec.partial), z: field(candidate, spec.z || 'metric:__selection__', spec.partial)}))
      .filter(row => present(row.x) && present(row.y)
        && (isJoint(spec.kind) ? finite(row.z) : finite(row.y)));
  }

  function grouped(rows) {
    const groups = new Map();
    for (const row of rows) {
      const key = JSON.stringify(row.x);
      const group = groups.get(key) || {x: row.x, values: [], trials: [], partial: false};
      group.values.push(row.y); group.trials.push(row.candidate.trial);
      group.partial ||= isPartial(row.candidate); groups.set(key, group);
    }
    return [...groups.values()].sort((a, b) => compare(a.x, b.x));
  }

  function axis(values, title) {
    const categories = unique(values), discrete = categorical(values);
    const indices = new Map(categories.map((value, index) => [JSON.stringify(value), index]));
    return {values: values.map(value => discrete ? indices.get(JSON.stringify(value)) : value),
      layout: {title: {text: title}, autorange: true,
        ...(discrete ? {tickmode: 'array', tickvals: categories.map((_, index) => index),
          ticktext: categories.map(label)} : {})}};
  }

  function render(spec, target = 'study-custom-chart', statusTarget = 'study-chart-status') {
    const chart = node(target); if (!chart || !spec) return;
    const colorscale = palettes[spec.palette] || 'Viridis';
    const layout = baseLayout(spec.name, fieldLabel(spec.x), fieldLabel(spec.y));
    const rows = rowsFor(spec, spec.y), traces = [];
    let message = '';
    if(spec.kind==='parallel'){
      const parameters=spec.parameters||data.parameters.slice(0,8), valid=data.candidates.filter(candidate=>(isComplete(candidate)||spec.partial)
        &&parameters.every(name=>present(field(candidate,'param:'+name,spec.partial)))&&finite(field(candidate,spec.y,spec.partial)));
      const dimensions=parameters.map(name=>{const a=axis(valid.map(row=>row.parameters[name]),name);return {label:name,values:a.values,...(a.layout.tickvals?{tickvals:a.layout.tickvals,ticktext:a.layout.ticktext}:{})}});
      dimensions.push({label:fieldLabel(spec.y),values:valid.map(row=>field(row,spec.y,spec.partial))});
      traces.push({type:'parcoords',dimensions,line:{color:valid.map(row=>field(row,spec.y,spec.partial)),colorscale,showscale:true}});
      message=`${valid.length} observed Trials · ${parameters.length} parameters, coloured by recorded metric. Missing coordinates are excluded; categories retain labels.`;
    } else if (isJoint(spec.kind)) {
      if (spec.kind === 'scatter3d') {
        const x = axis(rows.map(row => row.x), fieldLabel(spec.x));
        const y = axis(rows.map(row => row.y), fieldLabel(spec.y));
        traces.push({type: 'scatter3d', mode: 'markers', x: x.values, y: y.values,
          z: rows.map(row => row.z), marker: {size: 6, color: rows.map(row => row.z), colorscale,
            symbol: rows.map(row => isPartial(row.candidate) ? 'x' : 'circle'),
            reversescale: Boolean(spec.reverse), showscale: true, colorbar: {title: {text: fieldLabel(spec.z)}}},
          customdata: rows.map(row => [label(row.x), label(row.y), row.candidate.trial,
            isPartial(row.candidate) ? 'partial / censored' : 'completed']),
          hovertemplate: 'x=%{customdata[0]}<br>y=%{customdata[1]}<br>metric=%{z:.6g}'
            + '<br>trial=%{customdata[2]}<br>%{customdata[3]}<extra></extra>'});
        layout.scene = {xaxis: x.layout, yaxis: y.layout, zaxis: {title: {text: fieldLabel(spec.z)}, autorange: true}};
        message = `${rows.length} observed trials · hover a point for its exact values and identity.`;
      } else {
        const xs = domain(spec.x, rows.map(row => row.x)), ys = domain(spec.y, rows.map(row => row.y));
        if (xs.length * ys.length > 65536) {
          message = 'Too many distinct value pairs for a readable grid. Use 3D scatter instead; no observations were discarded.';
        } else if (spec.kind === 'surface' && (categorical(xs) || categorical(ys))) {
          message = '3D surfaces require numeric parameter axes. Use a heatmap or 3D scatter for categorical values.';
        } else {
          const cells = new Map();
          for (const row of rows) {
            const key = JSON.stringify([row.x, row.y]), cell = cells.get(key) || {values: [], trials: [], partial: false};
            cell.values.push(row.z); cell.trials.push(row.candidate.trial);
            cell.partial ||= isPartial(row.candidate); cells.set(key, cell);
          }
          const matrix = ys.map(y => xs.map(x => cells.get(JSON.stringify([x, y]))));
          const z = matrix.map(row => row.map(cell => cell ? mean(cell.values) : null));
          traces.push({type: spec.kind, x: xs.map(value => categorical(xs) ? label(value) : value),
            y: ys.map(value => categorical(ys) ? label(value) : value), z, colorscale,
            reversescale: Boolean(spec.reverse), colorbar: {title: {text: fieldLabel(spec.z)}},
            connectgaps: false, hoverongaps: false,
            customdata: matrix.map(row => row.map(cell => cell ? [cell.values.length,
              sd(cell.values), cell.trials.join(', '), cell.partial ? 'includes partial / censored' : 'completed'] : [0, null, '', 'unobserved'])),
            hovertemplate: 'x=%{x}<br>y=%{y}<br>mean=%{z:.6g}<br>trials=%{customdata[0]}'
              + '<br>SD across trials=%{customdata[1]}<br>trial IDs=%{customdata[2]}<br>%{customdata[3]}<extra></extra>'});
          if (spec.kind === 'surface') layout.scene = {xaxis: {title: {text: fieldLabel(spec.x)}},
            yaxis: {title: {text: fieldLabel(spec.y)}}, zaxis: {title: {text: fieldLabel(spec.z)}}};
          else {
            if (categorical(xs)) layout.xaxis.type = 'category';
            if (categorical(ys)) layout.yaxis.type = 'category';
          }
          message = `${rows.length} trials · ${cells.size} observed pairs / ${xs.length * ys.length} cells. `
            + 'Cells average trial summaries; missing pairs stay blank, never zero or predicted.';
        }
      }
    } else {
      const fields = [...new Set([spec.y, ...(spec.metrics || [])])];
      const unit = field => data.research?.metric_catalog?.metrics?.[field.replace('metric:', '').replace('__selection__','selection_objective')]?.unit || 'unknown';
      const units = fields.map(unit);
      // Unknown units are not assumed compatible. Preserve independent scales unless normalized explicitly.
      const smallMultiples = fields.length > 1 && !spec.normalize
        && (units.includes('unknown') || new Set(units).size > 1);
      const metricColors = ['#58a6ff', '#56d364', '#bc8cff', '#ffa657', '#39c5cf', '#ff7b72'];
      for (const [index, yField] of fields.entries()) {
        const selected = rowsFor(spec, yField), color = metricColors[index % metricColors.length];
        if(spec.normalize && selected.length){const values=selected.map(row=>row.y),minimum=Math.min(...values),span=Math.max(...values)-minimum;
          selected.forEach(row=>row.y=span?(row.y-minimum)/span:0);}
        if (spec.aggregate === 'mean') {
          const groups = grouped(selected);
          const discrete = categorical(groups.map(item => item.x));
          traces.push({type: spec.kind === 'bar' ? 'bar' : 'scatter',
            mode: spec.kind === 'line' ? 'lines+markers' : 'markers', name: fieldLabel(yField),
            x: groups.map(group => discrete ? label(group.x) : group.x),
            y: groups.map(group => mean(group.values)), marker: {color, size: 9,
              symbol: groups.map(group => group.partial ? 'x' : 'circle')}, line: {color},
            error_y: {type: 'data', array: groups.map(group => sd(group.values) || 0), visible: true},
            customdata: groups.map(group => [group.trials.join(', '), group.values.length]),
            hovertemplate: 'x=%{x}<br>mean=%{y:.6g}<br>trial IDs=%{customdata[0]}'
              + '<br>trials=%{customdata[1]}<extra>%{fullData.name}</extra>'});
        } else {
          // A line sorted by X is descriptive; its segments do not establish causality.
          selected.sort((a, b) => compare(a.x, b.x));
          const discrete = categorical(selected.map(item => item.x));
          traces.push({type: spec.kind === 'bar' ? 'bar' : 'scatter',
            mode: spec.kind === 'line' ? 'lines+markers' : 'markers', name: fieldLabel(yField),
            x: selected.map(row => discrete ? label(row.x) : row.x),
            y: selected.map(row => row.y), line: {color}, marker: {color, size: 9,
              symbol: selected.map(row => isPartial(row.candidate) ? 'x' : 'circle')},
            customdata: selected.map(row => [row.candidate.trial, isPartial(row.candidate) ? 'partial / censored' : 'completed']),
            hovertemplate: 'x=%{x}<br>y=%{y:.6g}<br>trial=%{customdata[0]}<br>%{customdata[1]}<extra>%{fullData.name}</extra>'});
        }
        if(smallMultiples){const suffix=index===0?'':String(index+1),trace=traces[traces.length-1];trace.xaxis='x'+suffix;trace.yaxis='y'+suffix;
          layout['xaxis'+suffix]={title:{text:fieldLabel(spec.x)},autorange:true};
          layout['yaxis'+suffix]={title:{text:fieldLabel(yField)},autorange:true};}
      }
      if(smallMultiples){layout.grid={rows:fields.length,columns:1,pattern:'independent'};layout.height=Math.min(1800,fields.length*280);}
      if(spec.normalize){layout.yaxis.title={text:'Per-metric visual 0–1 normalization (not scientific utility)'};}
      message = `${rows.length} trials for the primary metric · ${fields.length} metric(s). `
        + (spec.aggregate === 'mean' ? 'Equal-weight trial means; error bars are empirical SD across trials, not seed confidence intervals.'
        : 'Each point is a trial summary; duplicate parameter values are retained.')
        + (smallMultiples?' Different or unknown units use small multiples with independent Y scales.':'');
    }
    if (!traces.length || traces.every(trace => trace.type==='parcoords'?!trace.dimensions?.some(d=>d.values?.length):!trace.x?.length)) {
      const parameterRows = data.candidates.filter(candidate => present(field(candidate, spec.x, true)));
      const selection = spec.y === 'metric:__selection__' || (isJoint(spec.kind) && spec.z === 'metric:__selection__');
      const reason = !parameterRows.length ? 'No trials have a recorded value for the selected X parameter.'
        : selection ? 'No final selection objective is available for this view. '
          + (target === 'study-custom-chart' ? 'Choose Current observed objective or Best observed objective and enable partial observations when needed.'
            : 'Use Explore to inspect current/best partial observations without inventing a final score.')
        : 'No recorded values match this metric and evidence filter. Try another metric or enable partial observations.';
      message += ' ' + reason;
      layout.annotations = [{text: 'No matching observations. See the explanation below.', xref: 'paper', yref: 'paper',
        x: 0.5, y: 0.5, showarrow: false}];
    }
    status(message + (spec.partial ? ' Partial/censored evidence is descriptive only, not final selection.' : ' Completed evidence only.'), statusTarget);
    // A new graph specification resets stale zoom/camera rather than reusing another metric's scale.
    layout.uirevision = spec.id;
    layout.autosize = true;
    Plotly.react(chart, traces, layout, config).then(() => {
      // Initial figures are mounted in hidden tabs; fit the now-visible panel after rendering.
      if (chart.clientWidth) requestAnimationFrame(() => Plotly.Plots.resize(chart));
    });
  }

  function form() {
    const joint = isJoint(id('kind').value);
    id('z-control').hidden = !joint;
    id('extra-control').hidden = joint;
    id('aggregate').disabled = joint;
  }

  function show(savedId) {
    clearTimeout(previewTimer);
    const charts = getPrefs().customCharts || [], spec = charts.find(item => item.id === savedId) || charts[0];
    document.querySelectorAll('.saved-study-chart').forEach(button =>
      button.classList.toggle('active', button.dataset.id === spec?.id));
    if (spec) {
      selectedParameters=spec.parameters;
      for (const name of ['kind', 'x', 'y', 'z', 'aggregate', 'palette']) {
        if (spec[name] !== undefined) {
          const select=id(name);
          if(select.options&&![...select.options].some(o=>o.value===spec[name])){const option=document.createElement('option');option.value=spec[name];option.textContent=fieldLabel(spec[name]);select.append(option);}
          select.value = spec[name];
        }
      }
      id('name').value = spec.name;
      id('partial').checked = Boolean(spec.partial); id('reverse').checked = Boolean(spec.reverse);
      if(id('normalize'))id('normalize').checked=!!spec.normalize;
      setAdditional(spec.metrics || []);
      if(document.getElementById('research-view-notes'))document.getElementById('research-view-notes').value=spec.notes||'';
      form(); save({customChartId: spec.id}); render(spec);document.dispatchEvent(new Event('research-controls-changed'));
    }
  }

  function saved() {
    const target = document.getElementById('study-saved-charts'); target.replaceChildren();
    const charts = getPrefs().customCharts || [];
    for (const spec of charts) {
      const button = document.createElement('button'); button.className = 'saved-chart saved-study-chart';
      button.dataset.id = spec.id; button.textContent = spec.name;
      button.addEventListener('click', () => show(spec.id));
      const remove = document.createElement('button'); remove.textContent = '×'; remove.title = 'Delete ' + spec.name;
      remove.addEventListener('click', () => {const remaining = charts.filter(item => item.id !== spec.id);
        save({customCharts: remaining, customChartId: remaining[0]?.id}); saved();});
      const wrap = document.createElement('span'); wrap.append(button, remove); target.appendChild(wrap);
    }
    if (!charts.length) preview(); else show(getPrefs().customChartId);
  }

  function specification() {
    return {name: id('name').value.trim() || 'Preview', kind: id('kind').value,
      x: id('x').value, y: id('y').value, z: id('z').value,
      metrics: [...id('extra-metrics').querySelectorAll('input:checked')].map(input => input.value),
      aggregate: id('aggregate').value, partial: id('partial').checked,
      palette: id('palette').value, reverse: id('reverse').checked,
      notes:document.getElementById('research-view-notes')?.value||'',normalize:!!id('normalize')?.checked,...(selectedParameters?{parameters:selectedParameters}: {})};
  }
  function setAdditional(fields){for(const field of fields){if(![...id('extra-metrics').querySelectorAll('input')].some(input=>input.value===field)){
    const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=field;label.append(input,document.createTextNode(fieldLabel(field)));id('extra-metrics').append(label);}}
    id('extra-metrics').querySelectorAll('input').forEach(input=>input.checked=fields.includes(input.value));}
  function open(spec){if(Object.hasOwn(spec,'parameters'))selectedParameters=spec.parameters?.length?spec.parameters:undefined;else if(Object.hasOwn(spec,'x')||Object.hasOwn(spec,'kind'))selectedParameters=undefined;for(const name of ['kind','x','y','z','aggregate','palette']){if(spec[name]!==undefined){const select=id(name);
    if(select.options && ![...select.options].some(o=>o.value===spec[name])){const option=document.createElement('option');option.value=spec[name];option.textContent=fieldLabel(spec[name]);select.append(option);}select.value=spec[name];}}
    if(spec.metrics)setAdditional(spec.metrics);if(spec.partial!==undefined)id('partial').checked=!!spec.partial;
    if(spec.normalize!==undefined&&id('normalize'))id('normalize').checked=!!spec.normalize;
    id('name').value=spec.name||'Exploratory view';if(document.getElementById('research-view-notes'))document.getElementById('research-view-notes').value=spec.notes||'';form();preview();document.dispatchEvent(new Event('research-controls-changed'));}
  function importViews(document){const fields=new Set(['trial',...data.parameters.map(n=>'param:'+n),...Object.keys(data.research?.metric_catalog?.metrics||{}).map(n=>'metric:'+(n==='selection_objective'?'__selection__':n)),...data.metrics.map(n=>'metric:'+n),
    'metric:__selection__','metric:__best_observed__','metric:__current_observed__',...['gpu_seconds','duration_seconds','cpu_seconds','peak_vram','peak_ram'].map(n=>'resource:'+n)]);
    if(document.views_version!==1||!Array.isArray(document.views)||document.views.length>50)throw Error('Unsupported views document (maximum 50 views).');
    const allowed=new Set(['id','name','kind','x','y','z','metrics','aggregate','partial','palette','reverse','notes','normalize','parameters']);
    for(const spec of document.views){if(!spec||typeof spec.name!=='string'||spec.name.length>200||Object.keys(spec).some(name=>!allowed.has(name))||!['scatter','line','bar','scatter3d','heatmap','surface','parallel'].includes(spec.kind)
      ||!Array.isArray(spec.metrics||[])||(spec.metrics||[]).length>32||![spec.x,spec.y,...(spec.z?[spec.z]:[]),...(spec.metrics||[])].every(field=>fields.has(field))
      ||typeof(spec.notes||'')!=='string'||(spec.notes||'').length>10000||!['points','mean',undefined].includes(spec.aggregate)||![undefined,true,false].includes(spec.partial)||![undefined,true,false].includes(spec.normalize)
      ||![undefined,true,false].includes(spec.reverse)||(spec.palette!==undefined&&!Object.hasOwn(palettes,spec.palette))
      ||(spec.parameters!==undefined&&(!Array.isArray(spec.parameters)||!spec.parameters.length||spec.parameters.length>32||!spec.parameters.every(name=>data.parameters.includes(name)))))throw Error('Invalid saved chart fields.');}
    const views=document.views.map(spec=>({...spec,id:globalThis.crypto?.randomUUID?.()||String(Math.random())}));save({customCharts:views,customChartId:views[0]?.id});saved();}
  let previewTimer;
  function preview() {
    const spec = specification(); spec.id = 'preview:' + JSON.stringify(spec);
    document.querySelectorAll('.saved-study-chart').forEach(button => button.classList.remove('active'));
    render(spec);
  }
  function queuePreview() {clearTimeout(previewTimer); previewTimer = setTimeout(preview, 80);}

  function updateParameter() {
    const name = document.getElementById('parameter-select').value;
    const metric = document.getElementById('parameter-metric').value;
    const spec = {id: 'parameter:' + name + ':' + metric, name: 'Observed metric by parameter value',
      kind: 'line', x: 'param:' + name, y: 'metric:' + metric, metrics:getPrefs().parameterMetrics||[], aggregate: 'mean', partial: false};
    render(spec, 'parameter-observed', 'parameter-observed-status');
    const body = document.querySelector('#parameter-values tbody'); body.replaceChildren();
    document.getElementById('parameter-value-metric').textContent = 'Mean · ' + fieldLabel(spec.y);
    for (const group of grouped(rowsFor(spec, spec.y))) {
      const row = document.createElement('tr');
      for (const value of [label(group.x), group.values.length, mean(group.values), sd(group.values)]) {
        const cell = document.createElement('td'); cell.textContent = value === null ? '—'
          : typeof value === 'number' ? value.toLocaleString(undefined, {maximumSignificantDigits: 6}) : value;
        row.appendChild(cell);
      }
      body.appendChild(row);
    }
    save({parameterMetric: metric});
  }

  function mount() {
    id('kind').addEventListener('change', () => {
      if (isJoint(id('kind').value)) {
        if (id('y').value.startsWith('metric:')) id('y').value = 'param:' + (data.parameters[1] || data.parameters[0] || '');
      } else if (id('y').value.startsWith('param:')) id('y').value = 'metric:__selection__';
      form();
    });
    if (data.parameters.length) id('x').value = 'param:' + data.parameters[0];
    for (const name of ['kind', 'x', 'y', 'z', 'aggregate', 'palette', 'reverse', 'partial', 'extra-metrics']) {
      if(['kind','x','y'].includes(name))id(name).addEventListener('change',()=>{selectedParameters=undefined;document.dispatchEvent(new Event('research-controls-changed'));});
      id(name).addEventListener('change', queuePreview);
    }
    id('name').addEventListener('input', queuePreview);
    document.getElementById('save-study-chart').addEventListener('click', () => {
      clearTimeout(previewTimer);
      const charts = getPrefs().customCharts || [], spec = specification();
      spec.id = globalThis.crypto?.randomUUID?.() || String(Date.now());
      spec.name = id('name').value.trim() || 'Chart ' + (charts.length + 1);
      save({customCharts: [...charts, spec], customChartId: spec.id}); saved();
    });
    form(); saved();
  }
  return {mount, updateParameter, open, importViews, render, specification, preview};
};
