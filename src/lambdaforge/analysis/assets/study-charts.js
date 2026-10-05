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

  // Descriptive rendering only. Unknown units may share a visual range, not a scientific meaning.
  function parameterFigure(spec, layout, traces) {
    const fields = [...new Set([spec.y, ...(spec.metrics || [])])];
    const colors = ['#58a6ff', '#56d364', '#bc8cff', '#ffa657', '#39c5cf', '#ff7b72'];
    const series = fields.map((name, index) => {
      const rows = rowsFor(spec, name);
      if (spec.normalize && rows.length) {
        const values = rows.map(row => row.y), low = Math.min(...values), span = Math.max(...values) - low;
        rows.forEach(row => row.y = span ? (row.y - low) / span : 0);
      }
      const groups = !spec.parameterView && spec.aggregate === 'points' && ['step','area'].includes(spec.kind)
        ? [...rows].sort((a,b) => compare(a.x,b.x)).map(row => ({x:row.x,values:[row.y],trials:[row.candidate.trial],partial:isPartial(row.candidate)})) : grouped(rows);
      const values = rows.map(row => row.y);
      const unit = data.research?.metric_catalog?.metrics?.[name.slice(7).replace('__selection__', 'selection_objective')]?.unit;
      return {name, rows, groups, unit: spec.normalize ? 'visual-normalized' : unit && unit !== 'unknown' ? unit : null,
        color: colors[index % colors.length], low: values.length ? Math.min(...values) : null,
        high: values.length ? Math.max(...values) : null};
    });
    const compatible = (a, b) => {
      if (a.unit && b.unit && a.unit !== b.unit) return false;
      if (a.low === null || b.low === null) return false;
      const overlap = Math.min(a.high, b.high) - Math.max(a.low, b.low);
      const smaller = Math.min(a.high - a.low, b.high - b.low);
      // Contained intervals share naturally; slight overlap must not collapse unrelated ranges.
      return smaller === 0 ? overlap >= 0 : overlap / smaller >= 0.5;
    };
    const scaleGroups = [];
    for (const item of series) {
      let group = spec.scales === 'shared' ? scaleGroups[0]
        : spec.scales === 'independent' ? null : scaleGroups.find(group => group.every(other => compatible(item, other)));
      if (!group) {group = []; scaleGroups.push(group);}
      group.push(item); item.scale = scaleGroups.indexOf(group);
    }
    const t = text => window.LambdaForgeStudyLocale?.t(text) || text;
    const metricNames = items => items.map(item => fieldLabel(item.name)).join(' · ');
    const ranges = scaleGroups.map(group => {
      const valid = group.filter(item => item.low !== null);
      if (!valid.length) return undefined;
      const low = Math.min(...valid.map(item => item.low)), high = Math.max(...valid.map(item => item.high));
      const padding = (high - low || Math.abs(low) || 1) * 0.08;
      return [low - padding, high + padding];
    });
    if (spec.kind === 'matrix') {
      const xs = domain(spec.x, series.flatMap(item => item.groups.map(group => group.x)));
      const groups = series.map(item => new Map(item.groups.map(group => [JSON.stringify(group.x), group])));
      const raw = groups.map(row => xs.map(x => {const group = row.get(JSON.stringify(x));return group ? mean(group.values) : null;}));
      const normalized = raw.map(row => {const values = row.filter(finite);if (!values.length) return row;
        const low = Math.min(...values), span = Math.max(...values) - low;
        return row.map(value => finite(value) ? (span ? (value - low) / span : 0.5) : null);});
      traces.push({type:'heatmap', x:xs.map(label), y:series.map(item => fieldLabel(item.name)), z:normalized,
        customdata:raw, zmin:0, zmax:1, colorscale:'Viridis', hoverongaps:false,
        colorbar:{title:{text:t('Within-metric range')}},
        hovertemplate:'%{y}<br>x=%{x}<br>mean=%{customdata:.6g}<extra></extra>'});
      layout.yaxis = {type:'category', autorange:'reversed'}; layout.xaxis.type = 'category';
      layout.height = Math.max(500, Math.min(1800, series.length * 55 + 180));
      return t('Heatmap colours are normalized within each metric; hover shows the original mean. Blank cells are unobserved, not zero.');
    }
    if (['histogram', 'ecdf', 'hbar'].includes(spec.kind)) {
      const panels = spec.arrangement === 'panels' || scaleGroups.length > 1;
      layout.annotations = []; layout.legend = {orientation:'h', y:-0.18};
      layout.barmode = spec.kind === 'histogram' ? 'overlay' : 'group';
      layout.height = panels ? Math.max(500, series.length * 300) : 500;
      for (const [index, item] of series.entries()) {
        const suffix = panels && index ? String(index + 1) : '';
        const refs = {xaxis:'x' + suffix, yaxis:'y' + suffix};
        const xRange = ranges[item.scale] && [...ranges[item.scale]];
        if (xRange && spec.kind === 'hbar') {
          xRange[0] = Math.min(0, xRange[0]); xRange[1] = Math.max(0, xRange[1]);
          if (spec.dispersion !== 'off') for (const member of scaleGroups[item.scale]) for (const group of member.groups) {
            const deviation = sd(group.values);
            if (deviation !== null) {xRange[0] = Math.min(xRange[0], mean(group.values) - deviation);
              xRange[1] = Math.max(xRange[1], mean(group.values) + deviation);}
          }
        }
        layout['xaxis' + suffix] = {title:{text:panels ? fieldLabel(item.name) : metricNames(series)},
          ...(xRange ? {range:xRange, autorange:false} : {autorange:true}), anchor:'y' + suffix};
        layout['yaxis' + suffix] = {title:{text:spec.kind === 'ecdf' ? t('Cumulative fraction')
          : spec.kind === 'histogram' ? t('Candidate count') : fieldLabel(spec.x)}, anchor:'x' + suffix,
          ...(spec.kind === 'ecdf' ? {range:[0,1], autorange:false} : {autorange:true}),
          ...(spec.kind === 'hbar' ? {type:'category'} : {})};
        if (panels) {
          const bottom = 1 - (index + 1) / series.length;
          layout['yaxis' + suffix].domain = [bottom + 0.17 / series.length, bottom + 0.88 / series.length];
        }
        const values = item.rows.map(row => row.y).sort((a,b) => a-b);
        if (spec.kind === 'histogram') traces.push({type:'histogram', name:fieldLabel(item.name),
          x:values, marker:{color:item.color}, opacity:0.65, ...refs,
          hovertemplate:'value=%{x}<br>candidates=%{y}<extra>%{fullData.name}</extra>'});
        else if (spec.kind === 'ecdf') {
          const xs = unique(values);
          // A descriptive empirical distribution, not an interpolated probability model.
          let cursor = 0;
          const ys = xs.map(x => {while (cursor < values.length && values[cursor] <= x) cursor++;return cursor / values.length;});
          traces.push({type:'scatter', mode:'lines+markers', name:fieldLabel(item.name), x:xs, y:ys,
            line:{color:item.color, shape:'hv'}, marker:{color:item.color}, ...refs,
            hovertemplate:'value≤%{x:.6g}<br>fraction=%{y:.3f}<extra>%{fullData.name}</extra>'});
        } else traces.push({type:'bar', orientation:'h', name:fieldLabel(item.name), ...refs,
          x:item.groups.map(group => mean(group.values)), y:item.groups.map(group => label(group.x)),
          marker:{color:item.color}, offsetgroup:item.name,
          error_x:{type:'data', array:item.groups.map(group => sd(group.values)), visible:spec.dispersion !== 'off'},
          hovertemplate:'parameter=%{y}<br>mean=%{x:.6g}<extra>%{fullData.name}</extra>'});
      }
      return t('Distributions show recorded candidate summaries, not individual seeds.')
        + (spec.kind === 'hbar' ? '' : ' ' + t('Parameter values are pooled in this distribution view; use boxes or violins to compare exact values.'));
    }
    const panels = spec.arrangement === 'panels', discrete = categorical(series.flatMap(item => item.rows.map(row => row.x)));
    layout.barmode = 'group'; layout.legend = {orientation:'h', y:-0.22};
    layout.showlegend = !panels;
    // Independent panels avoid illegible axis stacks, while preserving optional shared ranges.
    if (panels) {layout.height = Math.max(500, series.length * 300);layout.annotations = [];}
    if (!panels && scaleGroups.length > 1) {layout.xaxis.domain = [0, 0.82];layout.margin.r = 100;}
    for (const [index, item] of series.entries()) {
      const axisIndex = panels ? index : item.scale, suffix = axisIndex ? String(axisIndex + 1) : '';
      const refs = {yaxis:'y' + suffix, ...(panels ? {xaxis:'x' + suffix} : {})};
      const axisName = 'yaxis' + suffix;
      const yRange = ranges[item.scale] && [...ranges[item.scale]];
      if (yRange && spec.dispersion !== 'off' && spec.kind !== 'box') {
        for (const member of scaleGroups[item.scale]) for (const group of member.groups) {
          const deviation = sd(group.values);
          if (deviation !== null) {yRange[0] = Math.min(yRange[0], mean(group.values) - deviation);
            yRange[1] = Math.max(yRange[1], mean(group.values) + deviation);}
        }
      }
      if (yRange && spec.kind === 'bar') {yRange[0] = Math.min(0, yRange[0]);yRange[1] = Math.max(0, yRange[1]);}
      const title = panels ? fieldLabel(item.name) : metricNames(scaleGroups[item.scale]);
      layout[axisName] = {title:{text:title, font:{color:scaleGroups[item.scale][0].color}}, automargin:true,
        ...(yRange ? {range:yRange, autorange:false} : {autorange:true}),
        ...(!panels && axisIndex ? {overlaying:'y', side:'right', anchor:'free', autoshift:true, showgrid:false} : {})};
      if (panels) {
        const bottom = 1 - (index + 1) / series.length;
        layout[axisName].domain = [bottom + 0.10 / series.length, bottom + 0.85 / series.length];
        layout[axisName].anchor = 'x' + suffix;
        layout['xaxis' + suffix] = {title:{text:index === series.length - 1 ? fieldLabel(spec.x) : ''}, anchor:'y' + suffix, autorange:true,
          ...(discrete ? {type:'category'} : {})};
        layout.annotations.push({text:fieldLabel(item.name), x:0, y:bottom + 0.94 / series.length,
          xref:'paper', yref:'paper', showarrow:false, xanchor:'left', font:{color:item.color}});
      } else if (discrete) layout.xaxis.type = 'category';
      const xValue = value => discrete ? label(value) : value;
      if (['box', 'violin', 'raw'].includes(spec.kind)) {
        traces.push({type:spec.kind === 'raw' ? 'scatter' : spec.kind, mode:'markers', name:fieldLabel(item.name), x:item.rows.map(row => xValue(row.x)),
          y:item.rows.map(row => row.y), ...(spec.kind === 'violin' ? {points:'all', box:{visible:true}, meanline:{visible:true}, spanmode:'hard'}
            : spec.kind === 'box' ? {boxpoints:'all'} : {}),
          ...(spec.kind === 'raw' ? {} : {jitter:0.25, pointpos:0}),
          marker:{color:item.color, ...(spec.kind === 'raw' ? {symbol:item.rows.map(row => isPartial(row.candidate) ? 'x' : 'circle')} : {})},
          customdata:item.rows.map(row => row.candidate.trial), ...refs,
          hovertemplate:'x=%{x}<br>value=%{y:.6g}<br>trial=%{customdata}<extra>%{fullData.name}</extra>'});
        layout.boxmode = 'group'; layout.violinmode = 'group'; continue;
      }
      const groups = item.groups, means = groups.map(group => mean(group.values)), deviations = groups.map(group => sd(group.values));
      const connected = ['line', 'step', 'area'].includes(spec.kind);
      const band = spec.dispersion === 'band' && connected && !discrete;
      // Isolated SD values cannot form an honest ribbon: retain their whisker rather than hide evidence.
      const whiskers = deviations.map((value, n) => band
        && (deviations[n - 1] !== null && finite(deviations[n - 1]) || finite(deviations[n + 1])) ? null : value);
      if (band) {
        const rgba = item.color.match(/\w\w/g).map(value => parseInt(value, 16)).join(',');
        for (const sign of [-1, 1]) traces.push({type:'scatter', mode:'lines', x:groups.map(group => xValue(group.x)),
          y:means.map((value, n) => deviations[n] === null ? null : value + sign * deviations[n]),
          connectgaps:false, line:{width:0}, showlegend:false, hoverinfo:'skip',
          ...(sign === 1 ? {fill:'tonexty', fillcolor:`rgba(${rgba},0.18)`} : {}), ...refs});
      }
      traces.push({type:spec.kind === 'bar' ? 'bar' : 'scatter', mode:connected ? 'lines+markers' : 'markers',
        name:fieldLabel(item.name), x:groups.map(group => xValue(group.x)), y:means, ...refs,
        ...(spec.kind === 'bar' ? {offsetgroup:item.name, alignmentgroup:'parameter-metrics'} : {}),
        marker:{color:item.color, size:9}, line:{color:item.color, shape:spec.kind === 'step' ? 'hv' : 'linear'},
        ...(spec.kind === 'area' ? {fill:'tozeroy', fillcolor:item.color + '22'} : {}),
        error_y:{type:'data', array:whiskers, visible:spec.dispersion !== 'off' && whiskers.some(finite)},
        customdata:groups.map(group => [group.trials.join(', '), group.values.length, sd(group.values)]),
        hovertemplate:'x=%{x}<br>mean=%{y:.6g}<br>trial IDs=%{customdata[0]}<br>trials=%{customdata[1]}'
          + '<br>SD=%{customdata[2]}<extra>%{fullData.name}</extra>'});
    }
    return t('Equal-weight candidate summaries; SD is across candidates, not a seed confidence interval.')
      + ' ' + scaleGroups.map((group, index) => t('Scale') + ' ' + (index + 1) + ': ' + metricNames(group)).join('; ')
      + (spec.dispersion === 'band' && (discrete || !['line','step','area'].includes(spec.kind)) ? ' ' + t('Shaded bands require numeric X lines; this view uses SD whiskers instead.') : '')
      + (['box','violin'].includes(spec.kind) ? ' ' + t('Distributions show recorded candidate summaries, not individual seeds.') : '')
      + (spec.kind === 'violin' ? ' ' + t('Violin density is descriptive smoothing; small samples do not establish a population distribution.') : '');
  }

  function render(spec, target = 'study-custom-chart', statusTarget = 'study-chart-status') {
    const chart = node(target); if (!chart || !spec) return;
    const colorscale = palettes[spec.palette] || 'Viridis';
    const layout = baseLayout(spec.name, fieldLabel(spec.x), fieldLabel(spec.y));
    const rows = rowsFor(spec, spec.y), traces = [];
    let message = '';
    if(spec.parameterView || ['raw','step','area','box','violin','hbar','histogram','ecdf','matrix'].includes(spec.kind)){
      message = parameterFigure(spec, layout, traces);
    } else if(spec.kind==='parallel'){
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
      const incompatible = fields.length > 1 && !spec.normalize
        && (units.includes('unknown') || new Set(units).size > 1);
      const smallMultiples = incompatible && !spec.overlay;
      const overlayAxes = new Map();
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
        if(incompatible && spec.overlay){
          // Unknown units are independent, even when their magnitudes happen to match.
          const group=units[index]==='unknown'?yField:units[index];
          if(!overlayAxes.has(group))overlayAxes.set(group,overlayAxes.size);
          const axisIndex=overlayAxes.get(group),suffix=axisIndex===0?'':String(axisIndex+1);
          traces[traces.length-1].yaxis='y'+suffix;
          if(!layout['yaxis'+suffix]||axisIndex===0)layout['yaxis'+suffix]={title:{text:fieldLabel(yField),font:{color}},autorange:true,
            ...(axisIndex?{overlaying:'y',side:'right',anchor:'free',autoshift:true,showgrid:false}:{})};
        }
      }
      if(smallMultiples){layout.grid={rows:fields.length,columns:1,pattern:'independent'};layout.height=Math.min(1800,fields.length*280);}
      if(incompatible && spec.overlay)layout.margin.r=160;
      if(spec.normalize){layout.yaxis.title={text:'Per-metric visual 0–1 normalization (not scientific utility)'};}
      message = `${rows.length} trials for the primary metric · ${fields.length} metric(s). `
        + (spec.aggregate === 'mean' ? 'Equal-weight trial means; error bars are empirical SD across trials, not seed confidence intervals.'
        : 'Each point is a trial summary; duplicate parameter values are retained.')
        + (smallMultiples?' Different or unknown units use small multiples with independent Y scales.':'')
        + (incompatible&&spec.overlay?' Same-chart overlays use separate Y axes for different or unknown units.':'');
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
    status(message + (spec.normalize ? ' Per-metric visual 0–1 normalization (not scientific utility).' : '')
      + (spec.partial ? ' Partial/censored evidence is descriptive only, not final selection.' : ' Completed evidence only.'), statusTarget);
    // A new graph specification resets stale zoom/camera rather than reusing another metric's scale.
    layout.uirevision = spec.id;
    layout.autosize = true;
    // Plotly's initial embedded figure has a fixed CSS height; update it for multi-panel figures.
    if (spec.parameterView) {
      chart.style.height = layout.height + 'px';
      for (let wrapper = chart.parentElement; wrapper && wrapper !== document.getElementById(target); wrapper = wrapper.parentElement) {
        if (wrapper.style.height) wrapper.style.height = 'auto';
      }
    }
    Plotly.react(chart, traces, layout, config).then(() => {
      // Initial figures are mounted in hidden tabs; fit the now-visible panel after rendering.
      if (chart.clientWidth) requestAnimationFrame(() => Plotly.Plots.resize(chart));
    });
  }

  function form() {
    const joint = isJoint(id('kind').value);
    id('z-control').hidden = !joint;
    id('extra-control').hidden = joint;
    id('aggregate').disabled = joint || ['raw','box','violin','hbar','histogram','ecdf','matrix'].includes(id('kind').value);
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
    for(const spec of document.views){if(!spec||typeof spec.name!=='string'||spec.name.length>200||Object.keys(spec).some(name=>!allowed.has(name))||!['scatter','line','bar','scatter3d','heatmap','surface','parallel','raw','step','area','box','violin','hbar','histogram','ecdf','matrix'].includes(spec.kind)
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
    if(!metric){const chart=node('parameter-observed');if(chart)Plotly.react(chart,[],baseLayout('No metrics selected. Choose one or more metrics above.'),config);
      status('No metrics selected. Choose one or more metrics above.','parameter-observed-status');document.querySelector('#parameter-values tbody').replaceChildren();return;}
    const extras=getPrefs().parameterMetrics||[];
    const controls = Object.fromEntries(['style','arrangement','scales','dispersion'].map(key =>
      [key, document.getElementById('parameter-' + key).value]));
    save({parameterPlot:controls});
    const spec = {id: 'parameter:' + name + ':' + metric + ':' + JSON.stringify([extras,controls]), name: 'Observed metric by parameter value',
      kind:controls.style, arrangement:controls.arrangement, scales:controls.scales, dispersion:controls.dispersion,
      parameterView:true, x: 'param:' + name, y: 'metric:' + metric, metrics:extras, aggregate: 'mean', partial: false};
    document.getElementById('parameter-arrangement').disabled = controls.style === 'matrix';
    document.getElementById('parameter-scales').disabled = controls.style === 'matrix';
    document.getElementById('parameter-dispersion').disabled = ['matrix','box'].includes(controls.style);
    document.dispatchEvent(new Event('research-controls-changed'));
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
    for (const key of ['style','arrangement','scales','dispersion']) {
      const input = document.getElementById('parameter-' + key), stored = getPrefs().parameterPlot?.[key];
      if ([...input.options].some(option => option.value === stored)) input.value = stored;
      input.addEventListener('change', updateParameter);
    }
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
