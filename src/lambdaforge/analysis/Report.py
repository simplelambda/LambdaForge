"""Optional self-contained Plotly export for Study Analysis."""

# Embedded CSS and JavaScript stay readable as browser-native source.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import math
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.work.atomic import atomic_write_text


def write_html(analysis: Mapping[str, Any], output: str | Path) -> Path:
    """Write the self-contained Study Analysis dashboard."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError as error:
        raise RuntimeError(
            "Install lambdaforge[analysis-report] to export interactive HTML reports."
        ) from error

    return _write_study_dashboard(analysis, output, go=go, plot=plot)


def _legacy_write_html(analysis: Mapping[str, Any], output: str | Path, *, go: Any, plot: Any) -> Path:
    """Build the former linear report; kept as a compact compatibility reference."""

    sections: list[str] = []
    candidates = [value for value in analysis.get("candidates", ()) if isinstance(value, Mapping)]
    if candidates:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=[str(value.get("trial")) for value in candidates],
                    y=[value.get("mean") for value in candidates],
                    error_y={
                        "type": "data",
                        "array": [value.get("standard_error") or 0 for value in candidates],
                    },
                )
            ]
        )
        figure.update_layout(
            title="Candidate ranking", xaxis_title="Trial", yaxis_title="Objective"
        )
        sections.append(plot(figure, include_plotlyjs="inline", output_type="div"))
    importance = analysis.get("parameter_importance", {})
    if isinstance(importance, Mapping) and importance:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=list(importance),
                    y=[value.get("importance", 0) for value in importance.values()],
                )
            ]
        )
        figure.update_layout(title="Global functional parameter importance")
        sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    top = analysis.get("top_region_importance", {})
    if isinstance(top, Mapping) and top:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=list(top),
                    y=[
                        value.get("importance", 0) if isinstance(value, Mapping) else 0
                        for value in top.values()
                    ],
                )
            ]
        )
        figure.update_layout(title="Observed top-region importance")
        sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    responses = analysis.get("response_curves", {})
    if isinstance(responses, Mapping):
        for name, response in responses.items():
            if not isinstance(response, Mapping):
                continue
            points = [value for value in response.get("points", ()) if isinstance(value, Mapping)]
            x = [value.get("x", value.get("category")) for value in points]
            y = [value.get("effect") for value in points]
            if x:
                figure = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines+markers")])
                figure.update_layout(title=f"Response: {name}")
                sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    interactions = analysis.get("interactions", {})
    matrix = interactions.get("matrix", {}) if isinstance(interactions, Mapping) else {}
    if isinstance(matrix, Mapping) and matrix:
        names = list(matrix)
        figure = go.Figure(
            data=[
                go.Heatmap(
                    z=[[matrix[left].get(right, 0) for right in names] for left in names],
                    x=names,
                    y=names,
                )
            ]
        )
        figure.update_layout(title="Interaction matrix")
        sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    surfaces = interactions.get("surfaces", {}) if isinstance(interactions, Mapping) else {}
    if isinstance(surfaces, Mapping):
        for name, surface in list(surfaces.items())[:6]:
            if not isinstance(surface, Mapping):
                continue
            x_values, y_values = list(surface.get("x", ())), list(surface.get("y", ()))
            cells = [value for value in surface.get("cells", ()) if isinstance(value, Mapping)]
            lookup = {(str(value.get("x")), str(value.get("y"))): value for value in cells}
            z_values = [
                [
                    lookup.get((str(x_value), str(y_value)), {}).get("predicted_objective")
                    for x_value in x_values
                ]
                for y_value in y_values
            ]
            if x_values and y_values:
                heatmap = go.Figure(data=[go.Heatmap(z=z_values, x=x_values, y=y_values)])
                heatmap.update_layout(title=f"Pairwise predictive heatmap: {name}")
                sections.append(plot(heatmap, include_plotlyjs=False, output_type="div"))
                if all(isinstance(value, int | float) for value in (*x_values, *y_values)):
                    surface_3d = go.Figure(data=[go.Surface(z=z_values, x=x_values, y=y_values)])
                    surface_3d.update_layout(title=f"Pairwise predictive surface: {name}")
                    sections.append(plot(surface_3d, include_plotlyjs=False, output_type="div"))
    if candidates:
        parameter_names = sorted(
            {str(name) for candidate in candidates for name in candidate.get("parameters", {})}
        )
        dimensions = []
        for name in parameter_names:
            values = [candidate.get("parameters", {}).get(name) for candidate in candidates]
            numeric = all(isinstance(value, int | float) for value in values if value is not None)
            if numeric:
                dimensions.append({"label": name, "values": values})
            else:
                levels = sorted({str(value) for value in values if value is not None})
                encoded = {level: index for index, level in enumerate(levels)}
                dimensions.append(
                    {
                        "label": name,
                        "values": [encoded.get(str(value), -1) for value in values],
                        "tickvals": list(encoded.values()),
                        "ticktext": levels,
                    }
                )
        if dimensions:
            figure = go.Figure(data=[go.Parcoords(dimensions=dimensions)])
            figure.update_layout(title="Candidate parallel coordinates")
            sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    coverage = analysis.get("coverage", {})
    marginal = coverage.get("marginal", {}) if isinstance(coverage, Mapping) else {}
    if isinstance(marginal, Mapping) and marginal:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=list(marginal),
                    y=[
                        value.get("transformed_range_coverage", value.get("active_fraction", 0))
                        if isinstance(value, Mapping)
                        else 0
                        for value in marginal.values()
                    ],
                )
            ]
        )
        figure.update_layout(title="Marginal search-space coverage")
        sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    boundaries = analysis.get("boundaries", {})
    if isinstance(boundaries, Mapping) and boundaries:
        figure = go.Figure(
            data=[
                go.Bar(
                    x=list(boundaries),
                    y=[
                        value.get("boundary_enrichment", 0) if isinstance(value, Mapping) else 0
                        for value in boundaries.values()
                    ],
                )
            ]
        )
        figure.update_layout(title="Boundary enrichment", yaxis_title="top/global")
        sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    resources = analysis.get("resources", {})
    resource_rows = resources.get("targets", ()) if isinstance(resources, Mapping) else ()
    resource_rows = [value for value in resource_rows if isinstance(value, Mapping)]
    for target in ("gpu_seconds", "duration_seconds", "peak_vram"):
        valid = [value for value in resource_rows if isinstance(value.get(target), int | float)]
        if valid:
            figure = go.Figure(
                data=[
                    go.Scatter(
                        x=[value[target] for value in valid],
                        y=[value.get("objective") for value in valid],
                        mode="markers+text",
                        text=[f"trial {value.get('trial')}" for value in valid],
                    )
                ]
            )
            figure.update_layout(title=f"Objective / {target}", xaxis_title=target)
            sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    component_pareto = analysis.get("pareto", {})
    component_names = (
        list(component_pareto.get("objective_components", ()))
        if isinstance(component_pareto, Mapping)
        else []
    )
    if len(component_names) >= 2:
        left, right = component_names[:2]
        valid = [
            value
            for value in candidates
            if left in value.get("objective_components", {})
            and right in value.get("objective_components", {})
        ]
        if valid:
            figure = go.Figure(
                data=[
                    go.Scatter(
                        x=[value["objective_components"][left] for value in valid],
                        y=[value["objective_components"][right] for value in valid],
                        mode="markers+text",
                        text=[f"trial {value.get('trial')}" for value in valid],
                    )
                ]
            )
            figure.update_layout(
                title="Scientific component Pareto", xaxis_title=left, yaxis_title=right
            )
            sections.append(plot(figure, include_plotlyjs=False, output_type="div"))
    findings = analysis.get("findings", ())
    finding_html = "".join(
        "<article><h3>"
        + html.escape(str(value.get("title", "Finding")))
        + "</h3><p>"
        + html.escape(str(value.get("statement", "")))
        + "</p><p><b>Recommendation:</b> "
        + html.escape(str(value.get("recommendation", "")))
        + "</p></article>"
        for value in findings
        if isinstance(value, Mapping)
    )
    raw = html.escape(json.dumps(analysis, indent=2, default=str))
    status = html.escape(str(analysis.get("source", {}).get("status", "unknown")))
    objective_label = html.escape(_objective_label(analysis))
    seed_analysis = analysis.get("seed_analysis", {})
    seed_warning = ""
    if isinstance(seed_analysis, Mapping) and seed_analysis.get("status") == "insufficient":
        seed_warning = (
            '<article class="warning"><h3>Empirical seed stability unavailable</h3><p>'
            + html.escape(
                str(
                    seed_analysis.get(
                        "interpretation",
                        "Leading candidates do not have enough repeated-seed evidence.",
                    )
                )
            )
            + "</p><p>A one-seed bootstrap is not presented as empirical stability.</p></article>"
        )
    diagnostic = html.escape(
        json.dumps(
            {
                "surrogate": analysis.get("surrogate"),
                "pruning": analysis.get("pruning"),
                "candidate_pool_resolution": analysis.get("candidate_pool_resolution"),
                "seed_stability": analysis.get("seed_analysis"),
                "winner": analysis.get("winner"),
            },
            indent=2,
            default=str,
        )
    )
    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>LambdaForge Study Analysis</title><style>"
        "body{font:16px system-ui;max-width:1200px;margin:auto;padding:2rem;color:#20242b}"
        "article{border-left:4px solid #d19a00;padding:.2rem 1rem;margin:1rem 0}"
        "details{margin-top:2rem}"
        "pre{white-space:pre-wrap;background:#f5f6f8;padding:1rem}"
        "</style></head><body><h1>LambdaForge Study Analysis</h1>"
        f"<p>Status: {status} · Objective: {objective_label}</p>"
        f"{seed_warning}{''.join(sections)}<h2>Diagnostics</h2>"
        f"<pre>{diagnostic}</pre><h2>Findings</h2>{finding_html}"
        "<details><summary>Complete reproducible analysis JSON</summary>"
        f"<pre>{raw}</pre></details></body></html>"
    )
    path = Path(output).expanduser().resolve()
    atomic_write_text(path, document)
    return path


def _write_study_dashboard(
    analysis: Mapping[str, Any], output: str | Path, *, go: Any, plot: Any
) -> Path:
    """Render persisted Study Analysis as one navigable offline dashboard."""
    candidates = [dict(value) for value in analysis.get("candidates", ()) if isinstance(value, Mapping)]
    parameter_names = sorted(
        {str(name) for candidate in candidates for name in candidate.get("parameters", {})}
    )
    metric_names = sorted(
        {
            str(name)
            for candidate in candidates
            for name in candidate.get("objective_components", {})
        }
    )
    objective_label = _objective_label(analysis)
    common_layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(13,17,23,.72)",
        "font": {"family": "Inter, ui-sans-serif, system-ui", "color": "#c9d1d9"},
        "height": 500,
        "margin": {"l": 72, "r": 30, "t": 52, "b": 66},
    }
    ranking = go.Figure(
        data=[
            go.Bar(
                x=[str(value.get("trial")) for value in candidates],
                y=[value.get("mean") for value in candidates],
                error_y={
                    "type": "data",
                    "array": [value.get("standard_error") or 0 for value in candidates],
                },
                marker={"color": "#58a6ff"},
                hovertemplate="trial=%{x}<br>selection objective=%{y:.6g}<extra></extra>",
            )
        ]
    )
    ranking.update_layout(
        **common_layout,
        title="Candidate evidence",
        xaxis_title="Trial",
        yaxis_title=objective_label,
    )
    importance = analysis.get("parameter_importance", {})
    importance = importance if isinstance(importance, Mapping) else {}
    importance_figure = go.Figure(
        data=[
            go.Bar(
                x=list(importance),
                y=[
                    value.get("importance", 0) if isinstance(value, Mapping) else 0
                    for value in importance.values()
                ],
                marker={"color": "#bc8cff"},
                hovertemplate="%{x}<br>global importance=%{y:.4f}<extra></extra>",
            )
        ]
    )
    importance_figure.update_layout(
        **common_layout,
        title="Global predictive importance",
        xaxis_title="Parameter",
        yaxis_title="Fraction of surrogate variation",
    )
    response_figure = go.Figure()
    response_figure.update_layout(
        **common_layout,
        title="Select a parameter to inspect its persisted response",
    )
    interactions = analysis.get("interactions", {})
    interactions = interactions if isinstance(interactions, Mapping) else {}
    interaction_figure = go.Figure()
    interaction_figure.update_layout(
        **common_layout,
        title="Select a persisted parameter pair",
    )
    coverage = analysis.get("coverage", {})
    marginal = coverage.get("marginal", {}) if isinstance(coverage, Mapping) else {}
    marginal = marginal if isinstance(marginal, Mapping) else {}
    coverage_figure = go.Figure(
        data=[
            go.Bar(
                x=list(marginal),
                y=[
                    value.get("transformed_range_coverage", value.get("active_fraction", 0))
                    if isinstance(value, Mapping)
                    else 0
                    for value in marginal.values()
                ],
                text=[
                    f"{100 * float(value.get('transformed_range_coverage', value.get('active_fraction', 0))):.0f}%"
                    if isinstance(value, Mapping)
                    else "0%"
                    for value in marginal.values()
                ],
                textposition="auto",
                marker={"color": "#39c5cf"},
            )
        ]
    )
    coverage_figure.update_layout(
        **common_layout,
        title="Observed marginal coverage",
        yaxis_title="Fraction of authored domain",
        yaxis={"range": [0, 1]},
    )
    resources = analysis.get("resources", {})
    resource_rows = resources.get("targets", ()) if isinstance(resources, Mapping) else ()
    resource_rows = [dict(value) for value in resource_rows if isinstance(value, Mapping)]
    resource_figure = go.Figure()
    resource_figure.update_layout(**common_layout, title="Objective and resource cost")
    study_custom_figure = go.Figure()
    study_custom_figure.update_layout(
        **common_layout,
        title="Create or select a saved candidate chart",
    )
    plot_config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    figure_html = {
        "ranking": plot(ranking, include_plotlyjs="inline", output_type="div", config=plot_config),
        "importance": plot(
            importance_figure, include_plotlyjs=False, output_type="div", config=plot_config
        ),
        "response": plot(
            response_figure, include_plotlyjs=False, output_type="div", config=plot_config
        ),
        "interaction": plot(
            interaction_figure, include_plotlyjs=False, output_type="div", config=plot_config
        ),
        "coverage": plot(
            coverage_figure, include_plotlyjs=False, output_type="div", config=plot_config
        ),
        "resources": plot(
            resource_figure, include_plotlyjs=False, output_type="div", config=plot_config
        ),
        "custom": plot(
            study_custom_figure,
            include_plotlyjs=False,
            output_type="div",
            config=plot_config,
        ),
    }
    completed = sum(1 for value in candidates if isinstance(value.get("mean"), int | float))
    censored = sum(int(value.get("censored_observations") or 0) for value in candidates)
    winner = analysis.get("winner", {})
    winner = winner if isinstance(winner, Mapping) else {}
    winning = winner.get("confirmed_winner") or winner.get("screening_winner")
    winning_trial = winning.get("trial") if isinstance(winning, Mapping) else None
    findings = [value for value in analysis.get("findings", ()) if isinstance(value, Mapping)]
    findings_html = "".join(
        '<article class="finding"><div><span class="badge">'
        + html.escape(str(value.get("reliability", "unrated")))
        + " reliability</span><h3>"
        + html.escape(str(value.get("title", "Finding")))
        + "</h3></div><p>"
        + html.escape(str(value.get("statement", "")))
        + '</p><p class="muted"><b>Next:</b> '
        + html.escape(str(value.get("recommendation", "")))
        + "</p></article>"
        for value in findings
    )
    coverage_rows = "".join(
        "<tr><td>"
        + html.escape(str(name))
        + "</td><td>"
        + html.escape(str(value.get("kind", "unknown")))
        + "</td><td>"
        + html.escape(str(value.get("authored_range", "—")))
        + "</td><td>"
        + html.escape(str(value.get("observed_range", "—")))
        + "</td><td>"
        + html.escape(str(value.get("occupied_bins", "—")))
        + " / "
        + html.escape(str(value.get("total_bins", "—")))
        + "</td></tr>"
        for name, value in marginal.items()
        if isinstance(value, Mapping)
    )
    payload = json.dumps(
        {
            "report_id": secrets.token_hex(12),
            "objective_label": objective_label,
            "candidates": candidates,
            "parameters": parameter_names,
            "metrics": metric_names,
            "responses": analysis.get("response_curves", {}),
            "interactions": interactions,
            "resources": resource_rows,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).replace("<", "\\u003c")
    raw = html.escape(json.dumps(analysis, indent=2, ensure_ascii=False, default=str))
    status = html.escape(str(analysis.get("source", {}).get("status", "unknown")))
    style = """
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--line:#30363d;
--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--good:#56d364}*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 82% -15%,#17304f 0,var(--bg) 35%);color:var(--text);
font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.shell{max-width:1800px;
margin:auto;padding:28px}header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px}
h1{margin:0;font-size:clamp(1.6rem,3vw,2.5rem);letter-spacing:-.04em}header p,.muted{color:var(--muted)}
.status{border:1px solid var(--line);border-radius:999px;padding:6px 11px;color:var(--good)}.cards{display:grid;
grid-template-columns:repeat(4,minmax(140px,1fr));gap:12px;margin:20px 0}.card,.panel,.finding{background:
linear-gradient(145deg,var(--panel2),#141920);border:1px solid var(--line);border-radius:13px;box-shadow:
0 14px 38px #0004}.card{padding:15px 18px}.card small{display:block;color:var(--muted);text-transform:
uppercase;font-size:.69rem;letter-spacing:.11em}.card strong{font-size:1.3rem}.tabs{position:sticky;top:0;z-index:5;
display:flex;gap:6px;overflow:auto;padding:10px 0;background:#0d1117e8;backdrop-filter:blur(12px)}button,select{
border:1px solid var(--line);border-radius:7px;background:#21262d;color:var(--text);padding:8px 11px;font:inherit}
button{cursor:pointer}.tab[aria-selected=true]{background:#1f6feb;border-color:#388bfd}.view[hidden]{display:none}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.panel{overflow:hidden;resize:vertical;
min-height:120px}.panel h2{
font-size:1rem;margin:0;padding:15px 18px;border-bottom:1px solid var(--line)}.plot{min-height:500px}.tools{display:flex;
align-items:center;gap:10px;flex-wrap:wrap;padding:12px 16px;border-bottom:1px solid var(--line)}.tools label{color:
var(--muted);font-size:.82rem;display:flex;align-items:center;gap:7px}.wide{grid-column:1/-1}.note{margin:0;padding:
11px 16px;color:var(--muted);font-size:.82rem;border-top:1px solid var(--line)}.table-wrap{overflow:auto;max-height:620px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;border-bottom:
1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{position:sticky;
top:0;background:#21262d;text-transform:uppercase;letter-spacing:.06em;font-size:.72rem}tr:hover{background:#ffffff08}
.compare{padding:16px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.compare-card{border:
1px solid var(--line);border-radius:10px;padding:13px}.compare-card dl{display:grid;grid-template-columns:auto 1fr;gap:5px 12px}
.compare-card dt{color:var(--muted)}.compare-card dd{margin:0;text-align:right;overflow-wrap:anywhere}.finding{padding:17px;
margin:12px 0}.finding h3{margin:.35rem 0}.badge{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;
color:#d2a8ff}.raw{white-space:pre-wrap;overflow:auto;max-height:650px;background:#090c10;padding:16px}.empty{padding:
38px;color:var(--muted);text-align:center}@media(max-width:900px){.shell{padding:15px}.cards,.grid{grid-template-columns:
1fr}.wide{grid-column:auto}.compare{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}}
.panel .js-plotly-plot,.panel .plotly-graph-div{max-width:100%}.custom-builder{display:grid;
grid-template-columns:minmax(170px,1fr) repeat(3,minmax(145px,auto)) auto;gap:10px;align-items:end}
.custom-builder input{width:100%;border:1px solid var(--line);border-radius:7px;background:#0d1117;
color:var(--text);padding:8px 10px}.saved-charts{display:flex;gap:7px;flex-wrap:wrap;padding:11px 16px;
border-bottom:1px solid var(--line)}.saved-chart.active{border-color:var(--accent);color:#79c0ff}
@media(max-width:1000px){.custom-builder{grid-template-columns:1fr 1fr}}
"""
    script = """
(()=>{const data=JSON.parse(document.getElementById('lf-study-data').textContent);
const key='lambdaforge:study-dashboard:'+data.report_id;let prefs={};try{prefs=JSON.parse(localStorage.getItem(key)||'{}')}catch(_e){}
const save=patch=>{prefs={...prefs,...patch};try{localStorage.setItem(key,JSON.stringify(prefs))}catch(_e){}};
const node=id=>document.querySelector('#'+id+' .js-plotly-plot, #'+id+' .plotly-graph-div');
const config={responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['lasso2d','select2d']};
const palettes={'Blue ↔ red':[[0,'#2166ac'],[.5,'#f2f2f2'],[1,'#b2182b']],
'Purple ↔ green':[[0,'#762a83'],[.5,'#f1f1f1'],[1,'#1b7837']],
'Brown ↔ teal':[[0,'#8c510a'],[.5,'#f5f5f5'],[1,'#01665e']],
'Accessible':[[0,'#0072b2'],[.5,'#d8d8d8'],[1,'#d55e00']]};
const fmt=value=>value===null||value===undefined||!Number.isFinite(Number(value))?'—':Number(value).toLocaleString(undefined,{maximumSignificantDigits:6});
const baseLayout=(title,x='',y='')=>({template:'plotly_dark',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(13,17,23,.72)',
font:{family:'Inter, ui-sans-serif, system-ui',color:'#c9d1d9'},height:500,margin:{l:72,r:30,t:52,b:66},title,
xaxis:{title:{text:x},autorange:true},yaxis:{title:{text:y},autorange:true}});
function metricValue(candidate,metric){return metric==='__selection__'?candidate.mean:candidate.objective_components?.[metric]}
function updateRanking(){const metric=document.getElementById('trial-metric').value;const chart=node('study-ranking');if(!chart)return;
 const rows=data.candidates.filter(row=>Number.isFinite(Number(metricValue(row,metric))));
 Plotly.react(chart,[{type:'bar',x:rows.map(row=>String(row.trial)),y:rows.map(row=>metricValue(row,metric)),
  error_y:metric==='__selection__'?{type:'data',array:rows.map(row=>row.standard_error||0)}:undefined,
  marker:{color:'#58a6ff'},customdata:rows.map(row=>row.n||0),hovertemplate:'trial=%{x}<br>value=%{y:.6g}<br>seeds=%{customdata}<extra></extra>'}],
  baseLayout('Candidate evidence','Trial',metric==='__selection__'?data.objective_label:metric),config);save({metric});}
function responsePoints(name){const response=data.responses?.[name]||{};return Array.isArray(response.points)?response.points:[]}
function updateParameter(){const name=document.getElementById('parameter-select').value;const points=responsePoints(name);const chart=node('parameter-response');
 if(chart){const x=points.map(point=>point.x??point.category),y=points.map(point=>point.predicted_objective??point.prediction??point.effect),
  uncertainty=points.map(point=>point.uncertainty??null),traces=[];if(y.length&&uncertainty.every(Number.isFinite)){
  traces.push({type:'scatter',x,y:y.map((v,i)=>v+uncertainty[i]),mode:'lines',line:{width:0},hoverinfo:'skip',showlegend:false});
  traces.push({type:'scatter',x,y:y.map((v,i)=>v-uncertainty[i]),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(88,166,255,.18)',hoverinfo:'skip',name:'uncertainty'});}
  traces.push({type:'scatter',x,y,mode:'lines+markers',name:'persisted response',customdata:points.map(p=>p.support_count??p.support??null),
  hovertemplate:name+'=%{x}<br>response=%{y:.6g}<br>support=%{customdata}<extra></extra>'});
  Plotly.react(chart,traces,baseLayout('Persisted adjusted response',name,data.objective_label),config);}
 const values=new Map();for(const candidate of data.candidates){const value=candidate.parameters?.[name],score=candidate.mean;
  if(value===undefined||!Number.isFinite(Number(score)))continue;const id=JSON.stringify(value),row=values.get(id)||{value,scores:[]};row.scores.push(Number(score));values.set(id,row)}
 const body=document.querySelector('#parameter-values tbody');body.replaceChildren();for(const row of values.values()){
  const mean=row.scores.reduce((a,b)=>a+b,0)/row.scores.length;const variance=row.scores.length>1?row.scores.reduce((a,b)=>a+(b-mean)**2,0)/(row.scores.length-1):null;
  const tr=document.createElement('tr');for(const value of [row.value,row.scores.length,mean,variance===null?null:Math.sqrt(variance)]){const td=document.createElement('td');td.textContent=typeof value==='string'?value:fmt(value);tr.appendChild(td)}body.appendChild(tr)}
 save({parameter:name});}
function updateInteraction(){const pair=document.getElementById('pair-select').value,surface=data.interactions?.surfaces?.[pair]||{};
 const x=surface.x||[],y=surface.y||[],lookup=new Map((surface.cells||[]).map(cell=>[String(cell.x)+'\u0000'+String(cell.y),cell.predicted_objective]));
 const z=y.map(right=>x.map(left=>lookup.get(String(left)+'\u0000'+String(right))??null));const chart=node('interaction-chart');if(!chart)return;
 const mode=document.getElementById('interaction-kind').value,palette=document.getElementById('study-palette').value,reverse=document.getElementById('study-reverse').checked;
 const trace=mode==='surface'?{type:'surface',x,y,z,colorscale:palettes[palette],reversescale:reverse,colorbar:{title:data.objective_label}}:
  {type:'heatmap',x,y,z,colorscale:palettes[palette],reversescale:reverse,colorbar:{title:data.objective_label},hoverongaps:false,
   hovertemplate:'x=%{x}<br>y=%{y}<br>prediction=%{z:.6g}<extra></extra>'};
 Plotly.react(chart,[trace],baseLayout('Persisted pairwise predictive response',pair.split('::')[0]||'',pair.split('::')[1]||''),config);
 save({pair,kind:mode,palette,reverse});}
function updateResources(){const target=document.getElementById('resource-select').value,chart=node('resource-chart');if(!chart)return;
 const rows=data.resources.filter(row=>Number.isFinite(Number(row[target]))&&Number.isFinite(Number(row.objective)));
 Plotly.react(chart,[{type:'scatter',mode:'markers+text',x:rows.map(r=>r[target]),y:rows.map(r=>r.objective),
 text:rows.map(r=>'trial '+r.trial),textposition:'top center',marker:{size:10,color:'#56d364'},hovertemplate:'%{text}<br>cost=%{x:.6g}<br>objective=%{y:.6g}<extra></extra>'}],
 baseLayout('Objective / '+target,target,data.objective_label),config);save({resource:target});}
function renderComparison(){const selected=[...document.querySelectorAll('.trial-compare:checked')].map(box=>Number(box.value)).slice(0,2);
 document.querySelectorAll('.trial-compare').forEach(box=>{if(!selected.includes(Number(box.value))&&selected.length>=2)box.disabled=true;else box.disabled=false});
 const area=document.getElementById('trial-comparison');area.replaceChildren();for(const id of selected){const row=data.candidates.find(item=>Number(item.trial)===id);if(!row)continue;
  const card=document.createElement('section');card.className='compare-card';const title=document.createElement('h3');title.textContent='Trial '+id;card.appendChild(title);
  const dl=document.createElement('dl');const fields=[['Selection objective',row.mean],['Standard error',row.standard_error],['Seeds',row.n],...Object.entries(row.parameters||{})];
  for(const [name,value] of fields){const dt=document.createElement('dt');dt.textContent=name;const dd=document.createElement('dd');dd.textContent=typeof value==='number'?fmt(value):String(value??'—');dl.append(dt,dd)}card.appendChild(dl);area.appendChild(card)}
 save({trials:selected});}
const resourcesByTrial=new Map(data.resources.map(row=>[Number(row.trial),row]));
function studyField(candidate,field){if(field==='trial')return candidate.trial;if(field.startsWith('param:'))return candidate.parameters?.[field.slice(6)];
 if(field.startsWith('metric:')){const name=field.slice(7);return name==='__selection__'?candidate.mean:candidate.objective_components?.[name]}
 if(field.startsWith('resource:'))return resourcesByTrial.get(Number(candidate.trial))?.[field.slice(9)];return null}
function fieldLabel(field){if(field==='trial')return 'Trial';const [kind,name]=field.split(':',2);if(kind==='metric'&&name==='__selection__')return data.objective_label;
 return (kind==='param'?'Parameter · ':kind==='metric'?'Metric · ':'Resource · ')+name.replaceAll('_',' ')}
function renderStudyCustom(spec){const chart=node('study-custom-chart');if(!chart||!spec)return;const rows=data.candidates.map(candidate=>({candidate,x:studyField(candidate,spec.x),y:studyField(candidate,spec.y)}))
 .filter(row=>row.x!==null&&row.x!==undefined&&row.y!==null&&row.y!==undefined);const mode=spec.kind==='line'?'lines+markers':'markers';
 const trace=spec.kind==='bar'?{type:'bar',x:rows.map(row=>row.x),y:rows.map(row=>row.y),customdata:rows.map(row=>row.candidate.trial),marker:{color:'#58a6ff'},
  hovertemplate:'x=%{x}<br>y=%{y}<br>trial=%{customdata}<extra></extra>'}:{type:'scatter',mode,x:rows.map(row=>row.x),y:rows.map(row=>row.y),
  customdata:rows.map(row=>row.candidate.trial),marker:{size:10,color:rows.map(row=>Number(row.candidate.trial)),colorscale:'Viridis',showscale:true,colorbar:{title:'Trial'}},
  hovertemplate:'x=%{x}<br>y=%{y}<br>trial=%{customdata}<extra></extra>'};Plotly.react(chart,[trace],baseLayout(spec.name,fieldLabel(spec.x),fieldLabel(spec.y)),config)}
function showStudyCustom(id){const charts=Array.isArray(prefs.customCharts)?prefs.customCharts:[],spec=charts.find(item=>item.id===id)||charts[0];
 document.querySelectorAll('.saved-study-chart').forEach(button=>button.classList.toggle('active',button.dataset.id===spec?.id));if(spec){save({customChartId:spec.id});renderStudyCustom(spec)}}
function renderStudyCharts(){const target=document.getElementById('study-saved-charts');target.replaceChildren();const charts=Array.isArray(prefs.customCharts)?prefs.customCharts:[];
 for(const spec of charts){const button=document.createElement('button');button.className='saved-chart saved-study-chart';button.dataset.id=spec.id;button.textContent=spec.name;button.addEventListener('click',()=>showStudyCustom(spec.id));
  const remove=document.createElement('button');remove.textContent='×';remove.title='Delete '+spec.name;remove.addEventListener('click',()=>{const remaining=charts.filter(item=>item.id!==spec.id);save({customCharts:remaining,customChartId:remaining[0]?.id});renderStudyCharts()});
  const wrap=document.createElement('span');wrap.append(button,remove);target.appendChild(wrap)}if(!charts.length){const empty=document.createElement('span');empty.className='muted';empty.textContent='No saved charts yet.';target.appendChild(empty)}showStudyCustom(prefs.customChartId)}
const panelHeights=prefs.panelHeights||{};document.querySelectorAll('.panel').forEach((panel,index)=>{const view=panel.closest('.view'),id=(view?.id||'root')+':'+index;panel.dataset.resizeKey=id;
 if(Number.isFinite(Number(panelHeights[id])))panel.style.height=panelHeights[id]+'px'});
const studyResizeObserver=new ResizeObserver(entries=>{const heights={...(prefs.panelHeights||{})};for(const entry of entries){if(entry.target.closest('.view')?.hidden)continue;
 heights[entry.target.dataset.resizeKey]=Math.round(entry.contentRect.height);entry.target.querySelectorAll('.js-plotly-plot,.plotly-graph-div').forEach(chart=>Plotly.Plots.resize(chart))}save({panelHeights:heights})});
document.querySelectorAll('.panel').forEach(panel=>studyResizeObserver.observe(panel));renderStudyCharts();
document.querySelectorAll('.trial-compare').forEach(box=>box.addEventListener('change',renderComparison));
if(Array.isArray(prefs.trials))document.querySelectorAll('.trial-compare').forEach(box=>box.checked=prefs.trials.includes(Number(box.value)));
for(const [id,value] of [['trial-metric',prefs.metric],['parameter-select',prefs.parameter],['pair-select',prefs.pair],['interaction-kind',prefs.kind],['study-palette',prefs.palette],['resource-select',prefs.resource]])if(value!==undefined&&[...document.getElementById(id).options].some(o=>o.value===value))document.getElementById(id).value=value;
document.getElementById('study-reverse').checked=Boolean(prefs.reverse);
document.getElementById('trial-metric').addEventListener('change',updateRanking);document.getElementById('parameter-select').addEventListener('change',updateParameter);
for(const id of ['pair-select','interaction-kind','study-palette','study-reverse'])document.getElementById(id).addEventListener('change',updateInteraction);
document.getElementById('resource-select').addEventListener('change',updateResources);
document.getElementById('save-study-chart').addEventListener('click',()=>{const charts=Array.isArray(prefs.customCharts)?prefs.customCharts:[],input=document.getElementById('study-chart-name');
 const spec={id:(globalThis.crypto?.randomUUID?.()||String(Date.now())),name:input.value.trim()||'Chart '+(charts.length+1),kind:document.getElementById('study-chart-kind').value,
  x:document.getElementById('study-chart-x').value,y:document.getElementById('study-chart-y').value};save({customCharts:[...charts,spec],customChartId:spec.id});input.value='';renderStudyCharts()});
function activate(target){document.querySelectorAll('.tab').forEach(tab=>tab.setAttribute('aria-selected',String(tab.dataset.target===target)));
 document.querySelectorAll('.view').forEach(view=>view.hidden=view.id!==target);save({tab:target});document.querySelectorAll('#'+target+' .js-plotly-plot').forEach(chart=>requestAnimationFrame(()=>Plotly.Plots.resize(chart)))}
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>activate(tab.dataset.target)));
updateRanking();updateParameter();updateInteraction();updateResources();renderComparison();if(prefs.tab&&document.getElementById(prefs.tab))activate(prefs.tab);
})();
"""
    metric_options = '<option value="__selection__">Selection objective</option>' + "".join(
        f'<option value="{html.escape(name, quote=True)}">{html.escape(name)}</option>'
        for name in metric_names
    )
    parameter_options = "".join(
        f'<option value="{html.escape(name, quote=True)}">{html.escape(name)}</option>'
        for name in parameter_names
    )
    pair_names = list(interactions.get("surfaces", {})) if isinstance(interactions.get("surfaces"), Mapping) else []
    pair_options = "".join(
        f'<option value="{html.escape(str(name), quote=True)}">{html.escape(str(name))}</option>'
        for name in pair_names
    )
    trial_rows = "".join(
        "<tr><td><input class=\"trial-compare\" type=\"checkbox\" value=\""
        + html.escape(str(value.get("trial")), quote=True)
        + "\"></td><td>"
        + html.escape(str(value.get("trial")))
        + "</td><td>"
        + html.escape(str(value.get("mean", "—")))
        + "</td><td>"
        + html.escape(str(value.get("standard_error", "—")))
        + "</td><td>"
        + html.escape(str(value.get("n", 0)))
        + "</td><td>"
        + html.escape(str(value.get("censored_observations", 0)))
        + "</td><td>"
        + html.escape(json.dumps(value.get("parameters", {}), ensure_ascii=False, default=str))
        + "</td></tr>"
        for value in candidates
    )
    study_field_options = (
        '<option value="trial">Trial</option>'
        + "".join(
            f'<option value="param:{html.escape(name, quote=True)}">Parameter · {html.escape(name)}</option>'
            for name in parameter_names
        )
        + '<option value="metric:__selection__">Metric · selection objective</option>'
        + "".join(
            f'<option value="metric:{html.escape(name, quote=True)}">Metric · {html.escape(name)}</option>'
            for name in metric_names
        )
        + "".join(
            f'<option value="resource:{name}">Resource · {label}</option>'
            for name, label in (
                ("gpu_seconds", "GPU seconds"),
                ("duration_seconds", "duration"),
                ("cpu_seconds", "CPU seconds"),
                ("peak_vram", "peak VRAM"),
                ("peak_ram", "peak RAM"),
            )
        )
    )
    study_y_field_options = study_field_options.replace(
        'value="metric:__selection__"', 'value="metric:__selection__" selected', 1
    )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>LambdaForge Study Analysis</title><style>'
        + style
        + '</style></head><body><div class="shell"><header><div><h1>Study Analysis dashboard</h1>'
        '<p>Explore persisted evidence; every association remains descriptive or predictive.</p></div>'
        f'<span class="status">{status}</span></header><div class="cards"><div class="card"><small>Candidates</small><strong>{len(candidates)}</strong></div>'
        f'<div class="card"><small>Complete candidates</small><strong>{completed}</strong></div><div class="card"><small>Censored Runs</small><strong>{censored}</strong></div>'
        f'<div class="card"><small>Leading trial</small><strong>{html.escape(str(winning_trial or "—"))}</strong></div></div>'
        '<nav class="tabs"><button class="tab" data-target="study-overview" aria-selected="true">Overview</button>'
        '<button class="tab" data-target="study-trials" aria-selected="false">Trials</button><button class="tab" data-target="study-parameters" aria-selected="false">Parameters</button>'
        '<button class="tab" data-target="study-interactions" aria-selected="false">Interactions</button><button class="tab" data-target="study-coverage" aria-selected="false">Coverage</button>'
        '<button class="tab" data-target="study-resources" aria-selected="false">Resources</button><button class="tab" data-target="study-findings" aria-selected="false">Findings & evidence</button>'
        '<button class="tab" data-target="study-custom" aria-selected="false">My charts</button></nav>'
        '<section class="view" id="study-overview"><div class="grid"><article class="panel wide"><h2>Candidate ranking</h2><div class="tools"><label>Displayed metric <select id="trial-metric">'
        + metric_options
        + '</select></label></div><div class="plot" id="study-ranking">'
        + figure_html["ranking"]
        + '</div></article><article class="panel"><h2>Predictive importance</h2><div class="plot">'
        + figure_html["importance"]
        + '</div></article><article class="panel"><h2>Interpretation</h2><div class="empty">Importance describes the fitted surrogate, not causality. Open Parameters and Interactions before narrowing a search space.</div></article></div></section>'
        '<section class="view" id="study-trials" hidden><div class="panel"><div class="tools"><span class="muted">Select at most two candidates to compare.</span></div><div class="table-wrap"><table><thead><tr><th>Compare</th><th>Trial</th><th>Selection</th><th>SE</th><th>Seeds</th><th>Censored</th><th>Parameters</th></tr></thead><tbody>'
        + trial_rows
        + '</tbody></table></div><div class="compare" id="trial-comparison"></div></div></section>'
        '<section class="view" id="study-parameters" hidden><div class="tools"><label>Parameter <select id="parameter-select">'
        + parameter_options
        + '</select></label></div><div class="grid"><article class="panel wide"><h2>Adjusted response and uncertainty</h2><div class="plot" id="parameter-response">'
        + figure_html["response"]
        + '</div><p class="note">This is the response persisted by Study Analysis; it is not fitted again in the browser.</p></article><article class="panel wide"><h2>Observed exact-value summary</h2><div class="table-wrap"><table id="parameter-values"><thead><tr><th>Value</th><th>Complete candidates</th><th>Mean selection</th><th>Empirical SD</th></tr></thead><tbody></tbody></table></div></article></div></section>'
        '<section class="view" id="study-interactions" hidden><div class="panel"><div class="tools"><label>Pair <select id="pair-select">'
        + pair_options
        + '</select></label><label>View <select id="interaction-kind"><option value="heatmap">Heatmap</option><option value="surface">3D surface</option></select></label><label>Colour scale <select id="study-palette"><option>Blue ↔ red</option><option>Purple ↔ green</option><option>Brown ↔ teal</option><option>Accessible</option></select></label><label><input id="study-reverse" type="checkbox"> Reverse</label></div><div class="plot" id="interaction-chart">'
        + figure_html["interaction"]
        + '</div><p class="note">Joint predictive response is conditional on persisted model evidence and must not be read as a causal effect.</p></div></section>'
        '<section class="view" id="study-coverage" hidden><div class="grid"><article class="panel wide"><h2>Domain coverage</h2><div class="plot">'
        + figure_html["coverage"]
        + '</div></article><article class="panel wide"><h2>Coverage details</h2><div class="table-wrap"><table><thead><tr><th>Parameter</th><th>Kind</th><th>Authored domain</th><th>Observed domain</th><th>Occupied bins</th></tr></thead><tbody>'
        + coverage_rows
        + '</tbody></table></div></article></div></section><section class="view" id="study-resources" hidden><div class="panel"><div class="tools"><label>Cost <select id="resource-select"><option value="gpu_seconds">GPU seconds</option><option value="duration_seconds">Duration</option><option value="cpu_seconds">CPU seconds</option><option value="peak_vram">Peak VRAM</option><option value="peak_ram">Peak RAM</option></select></label></div><div class="plot" id="resource-chart">'
        + figure_html["resources"]
        + '</div><p class="note">Per-comparable-Run resource evidence is kept separate from total controller spend.</p></div></section>'
        '<section class="view" id="study-findings" hidden>'
        + (findings_html or '<p class="empty">No findings were persisted.</p>')
        + '<details class="panel"><summary class="tools">Complete reproducible analysis JSON</summary><pre class="raw">'
        + raw
        + '</pre></details></section><section class="view" id="study-custom" hidden><div class="panel">'
        '<div class="tools custom-builder"><label>Chart name <input id="study-chart-name" '
        'placeholder="My candidate view"></label><label>Type <select id="study-chart-kind">'
        '<option value="scatter">Scatter</option><option value="line">Line</option>'
        '<option value="bar">Bar</option></select></label><label>X axis <select id="study-chart-x">'
        + study_field_options
        + '</select></label><label>Y axis <select id="study-chart-y">'
        + study_y_field_options
        + '</select></label><button id="save-study-chart">Save chart</button></div>'
        '<div class="saved-charts" id="study-saved-charts"></div><div class="plot" '
        'id="study-custom-chart">'
        + figure_html["custom"]
        + '</div><p class="note">Saved views are local presentation preferences over persisted '
        'candidate evidence. They do not refit or control HPO.</p></div></section></div>'
        '<script id="lf-study-data" type="application/json">'
        + payload
        + "</script><script>"
        + script
        + "</script></body></html>"
    )
    path = Path(output).expanduser().resolve()
    atomic_write_text(path, document)
    return path


def write_metric_html(
    curves: Mapping[str, Any],
    names: Sequence[str],
    output: str | Path,
    *,
    display_names: Mapping[str, Any] | None = None,
    preferred_names: Sequence[str] = (),
    selected_step: int | None = None,
    best_step: int | None = None,
) -> Path:
    """Write a self-contained interactive metric dashboard for one Run."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError as error:
        raise RuntimeError(
            "Install lambdaforge[analysis-report] to export interactive HTML reports."
        ) from error

    aliases = display_names or {}
    palette = (
        "#58a6ff",
        "#f2cc60",
        "#56d364",
        "#ff7b72",
        "#bc8cff",
        "#39c5cf",
        "#ffa657",
        "#db61a2",
        "#79c0ff",
        "#7ee787",
    )
    series: list[dict[str, Any]] = []
    for name in dict.fromkeys(str(value) for value in names):
        points = _metric_points(curves.get(name))
        if not points:
            continue
        values = [value for _step, value in points]
        low, high = min(values), max(values)
        normalized = (
            [(value - low) / (high - low) for value in values]
            if high > low
            else [0.5 for _value in values]
        )
        label = _metric_label(name, aliases)
        series.append(
            {
                "name": name,
                "label": label,
                "group": _metric_group(name),
                "color": palette[len(series) % len(palette)],
                "x": [step for step, _value in points],
                "y": values,
                "normalized": normalized,
                "stats": {
                    "count": len(points),
                    "first_step": points[0][0],
                    "latest_step": points[-1][0],
                    "first": values[0],
                    "latest": values[-1],
                    "minimum": low,
                    "maximum": high,
                    "delta": values[-1] - values[0],
                },
            }
        )

    available = {value["name"] for value in series}
    defaults = [str(value) for value in preferred_names if str(value) in available]
    defaults.extend(
        value["name"]
        for value in series
        if value["name"] not in defaults and value["group"] == "Validation"
    )
    defaults.extend(value["name"] for value in series if value["name"] not in defaults)
    defaults = defaults[:4]
    default_set = set(defaults)

    raw_figure = go.Figure()
    normalized_figure = go.Figure()
    for value in series:
        common = {
            "x": value["x"],
            "mode": "lines+markers",
            "name": value["label"],
            "visible": value["name"] in default_set,
            "line": {"color": value["color"], "width": 2.5},
            "marker": {"color": value["color"], "size": 5},
        }
        raw_figure.add_trace(
            go.Scatter(
                **common,
                y=value["y"],
                hovertemplate=(
                    f"epoch=%{{x}}<br>{html.escape(value['label'])}=%{{y:.6g}}<extra></extra>"
                ),
            )
        )
        normalized_figure.add_trace(
            go.Scatter(
                **common,
                y=value["normalized"],
                customdata=value["y"],
                hovertemplate=(
                    "epoch=%{x}<br>relative position=%{y:.3f}"
                    "<br>original=%{customdata:.6g}<extra></extra>"
                ),
            )
        )

    reference_shapes, reference_annotations = _metric_step_references(best_step, selected_step)
    common_layout = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(13,17,23,.72)",
        "font": {"family": "Inter, ui-sans-serif, system-ui", "color": "#c9d1d9"},
        "hovermode": "x unified",
        "showlegend": False,
        "height": 520,
        "margin": {"l": 72, "r": 28, "t": 52, "b": 62},
    }
    curve_layout = {
        **common_layout,
        "shapes": reference_shapes,
        "annotations": reference_annotations,
    }
    raw_figure.update_layout(
        **curve_layout,
        title="Observed learning curves",
        xaxis_title="Epoch",
        yaxis_title="Observed value",
    )
    normalized_figure.update_layout(
        **curve_layout,
        title="Relative trajectory comparison",
        xaxis_title="Epoch",
        yaxis_title="Position within each metric's observed range",
        yaxis={"range": [-0.05, 1.05]},
    )

    selected_series = [value for value in series if value["name"] in default_set]
    snapshot = go.Figure(
        data=[
            go.Bar(
                x=[value["stats"]["latest"] for value in selected_series],
                y=[value["label"] for value in selected_series],
                customdata=[value["stats"]["delta"] for value in selected_series],
                orientation="h",
                marker={"color": [value["color"] for value in selected_series]},
                hovertemplate="%{y}<br>latest=%{x:.6g}<br>delta=%{customdata:+.6g}<extra></extra>",
            )
        ]
    )
    snapshot.update_layout(
        **common_layout,
        title="Latest reported values",
        xaxis_title="Value",
        yaxis_title="",
    )

    correlations = _metric_correlations(series)
    initial_correlation = [
        [correlations.get(left["name"], {}).get(right["name"]) for right in selected_series]
        for left in selected_series
    ]
    correlation = go.Figure(
        data=[
            go.Heatmap(
                z=initial_correlation,
                x=[value["label"] for value in selected_series],
                y=[value["label"] for value in selected_series],
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                hovertemplate="%{y} / %{x}<br>r=%{z:.3f}<extra></extra>",
            )
        ]
    )
    correlation.update_layout(
        **common_layout,
        title="Same-epoch Pearson correlation",
        xaxis_title="Metric",
        yaxis_title="Metric",
    )

    relationship = go.Figure()
    if len(selected_series) >= 2:
        left, right = selected_series[:2]
        left_by_step = dict(zip(left["x"], left["y"], strict=True))
        right_by_step = dict(zip(right["x"], right["y"], strict=True))
        shared_steps = sorted(set(left_by_step) & set(right_by_step))
        relationship.add_trace(
            go.Scatter(
                x=[left_by_step[step] for step in shared_steps],
                y=[right_by_step[step] for step in shared_steps],
                customdata=shared_steps,
                mode="markers+lines",
                marker={
                    "color": shared_steps,
                    "colorscale": "Viridis",
                    "size": 9,
                    "showscale": True,
                    "colorbar": {"title": "Epoch"},
                },
                hovertemplate=(
                    f"{html.escape(left['label'])}=%{{x:.6g}}"
                    f"<br>{html.escape(right['label'])}=%{{y:.6g}}"
                    "<br>epoch=%{customdata}<extra></extra>"
                ),
            )
        )
    relationship.update_layout(
        **common_layout,
        title="Metric relationship over shared epochs",
        xaxis_title=selected_series[0]["label"] if selected_series else "Metric X",
        yaxis_title=selected_series[1]["label"] if len(selected_series) >= 2 else "Metric Y",
    )
    custom_figure = go.Figure()
    custom_figure.update_layout(
        **common_layout,
        title="Create or select a saved chart",
        xaxis_title="Epoch",
    )

    plot_config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
    figures = {
        "curves": plot(
            raw_figure,
            include_plotlyjs="inline",
            output_type="div",
            config=plot_config,
        ),
        "normalized": plot(
            normalized_figure,
            include_plotlyjs=False,
            output_type="div",
            config=plot_config,
        ),
        "snapshot": plot(
            snapshot,
            include_plotlyjs=False,
            output_type="div",
            config=plot_config,
        ),
        "correlation": plot(
            correlation,
            include_plotlyjs=False,
            output_type="div",
            config=plot_config,
        ),
        "relationship": plot(
            relationship,
            include_plotlyjs=False,
            output_type="div",
            config=plot_config,
        ),
        "custom": plot(
            custom_figure,
            include_plotlyjs=False,
            output_type="div",
            config=plot_config,
        ),
    }
    return _write_metric_dashboard(
        output,
        series,
        defaults,
        correlations,
        figures,
        best_step=best_step,
        selected_step=selected_step,
    )


def _metric_points(value: Any) -> list[tuple[int | float, float]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    points: list[tuple[int | float, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        step, observed = item.get("step"), item.get("value")
        if (
            isinstance(step, int | float)
            and not isinstance(step, bool)
            and math.isfinite(float(step))
            and isinstance(observed, int | float)
            and not isinstance(observed, bool)
            and math.isfinite(float(observed))
        ):
            points.append((step, float(observed)))
    return sorted(points, key=lambda point: float(point[0]))


def _metric_label(name: str, aliases: Mapping[str, Any]) -> str:
    alias = aliases.get(name)
    return str(alias) if alias else name.replace("_", " ").title()


def _metric_group(name: str) -> str:
    lowered = name.lower()
    if any(
        token in lowered
        for token in ("gpu", "cpu", "memory", "ram", "duration", "time", "seconds")
    ):
        return "Resources & timing"
    if lowered.startswith(("val_", "validation_")):
        return "Validation"
    if lowered.startswith(("train_", "training_")):
        return "Training"
    return "Other"


def _metric_step_references(
    best_step: int | None, selected_step: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for step, label, color in (
        (best_step, "Best epoch", "#56d364"),
        (selected_step, "Selected epoch", "#ff7b72"),
    ):
        if step is None or any(value.get("x0") == step for value in shapes):
            continue
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": step,
                "x1": step,
                "y0": 0,
                "y1": 1,
                "line": {"color": color, "width": 1.5, "dash": "dot"},
            }
        )
        annotations.append(
            {
                "x": step,
                "y": 1,
                "xref": "x",
                "yref": "paper",
                "text": label,
                "showarrow": False,
                "font": {"color": color, "size": 11},
                "yshift": 12,
            }
        )
    return shapes, annotations


def _metric_correlations(series: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
    by_name = {
        str(value["name"]): {
            str(step): float(observed)
            for step, observed in zip(value["x"], value["y"], strict=True)
        }
        for value in series
    }
    result: dict[str, dict[str, float | None]] = {}
    for left, left_values in by_name.items():
        result[left] = {}
        for right, right_values in by_name.items():
            shared = sorted(set(left_values) & set(right_values))
            if len(shared) < 2:
                result[left][right] = None
                continue
            x = [left_values[step] for step in shared]
            y = [right_values[step] for step in shared]
            x_mean, y_mean = sum(x) / len(x), sum(y) / len(y)
            numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
            x_scale = math.sqrt(sum((value - x_mean) ** 2 for value in x))
            y_scale = math.sqrt(sum((value - y_mean) ** 2 for value in y))
            result[left][right] = (
                numerator / (x_scale * y_scale) if x_scale > 0 and y_scale > 0 else None
            )
    return result


def _write_metric_dashboard(
    output: str | Path,
    series: Sequence[Mapping[str, Any]],
    defaults: Sequence[str],
    correlations: Mapping[str, Any],
    figures: Mapping[str, str],
    *,
    best_step: int | None,
    selected_step: int | None,
) -> Path:
    groups = ("Validation", "Training", "Resources & timing", "Other")
    controls: list[str] = []
    default_set = set(defaults)
    for group in groups:
        members = [value for value in series if value.get("group") == group]
        if not members:
            continue
        options = "".join(
            '<label class="metric-option" data-metric="'
            + html.escape(str(value["name"]), quote=True)
            + '" data-default-group="'
            + html.escape(group, quote=True)
            + '" title="'
            + html.escape(f"{value['label']} · {value['name']}", quote=True)
            + '" data-search="'
            + html.escape(f"{value['label']} {value['name']}", quote=True).lower()
            + '"><input type="checkbox" value="'
            + html.escape(str(value["name"]), quote=True)
            + '"'
            + (" checked" if value["name"] in default_set else "")
            + '><span class="swatch" style="--metric-colour:'
            + html.escape(str(value["color"]), quote=True)
            + '"></span><span><strong>'
            + html.escape(str(value["label"]))
            + '</strong><small>'
            + html.escape(str(value["name"]))
            + "</small></span></label>"
            for value in members
        )
        controls.append(
            '<details class="metric-group" data-group="'
            + html.escape(group, quote=True)
            + '" open><summary><span>'
            + html.escape(group)
            + f'</span><small>{len(members)}</small></summary>{options}</details>'
        )

    all_steps = [float(step) for value in series for step in value.get("x", ())]
    epoch_summary = (
        f"{min(all_steps):g}–{max(all_steps):g}" if all_steps else "No observations"
    )
    context = " · ".join(
        value
        for value in (
            f"Best epoch {best_step}" if best_step is not None else "",
            f"Selected epoch {selected_step}" if selected_step is not None else "",
        )
        if value
    )
    payload = json.dumps(
        {
            "report_id": secrets.token_hex(12),
            "series": list(series),
            "defaults": list(defaults),
            "correlations": correlations,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    style = """
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--line:#30363d;
--text:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--focus:#1f6feb;--good:#56d364;
--sidebar-width:300px}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% -20%,#17304f 0,
var(--bg) 35%);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,
BlinkMacSystemFont,"Segoe UI",sans-serif}.shell{max-width:1800px;margin:auto;padding:28px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:24px;margin-bottom:20px}
h1{font-size:clamp(1.5rem,3vw,2.35rem);margin:0;letter-spacing:-.035em}header p{margin:.35rem 0 0;
color:var(--muted)}.context{color:#c9d1d9;text-align:right}.cards{display:grid;
grid-template-columns:repeat(3,minmax(150px,1fr));gap:12px;margin:0 0 18px}.card{background:
linear-gradient(145deg,#1c2128,#141920);border:1px solid var(--line);border-radius:12px;
padding:15px 18px;
box-shadow:0 10px 30px #0003}.card small{display:block;color:var(--muted);text-transform:uppercase;
font-size:.7rem;letter-spacing:.11em}.card strong{display:block;font-size:1.35rem;margin-top:3px}
.dashboard{display:grid;grid-template-columns:var(--sidebar-width) 7px minmax(0,1fr);gap:8px;align-items:start}
.sidebar{position:sticky;top:16px;max-height:calc(100vh - 32px);overflow:hidden;display:flex;
flex-direction:column;background:color-mix(in srgb,var(--panel) 94%,transparent);
border:1px solid var(--line);border-radius:14px;
box-shadow:0 16px 45px #0005}.sidebar-head{padding:16px;border-bottom:1px solid var(--line)}
.splitter{height:calc(100vh - 32px);position:sticky;top:16px;cursor:col-resize;border-radius:8px;
background:linear-gradient(90deg,transparent 3px,var(--line) 3px,var(--line) 4px,transparent 4px);
touch-action:none}.splitter:hover,.splitter.dragging{background:var(--focus)}
.sidebar-head h2{font-size:1rem;margin:0 0 10px}.search{width:100%;border:1px solid var(--line);
border-radius:8px;background:#0d1117;color:var(--text);padding:9px 11px;outline:none}.search:focus{
border-color:var(--focus);box-shadow:0 0 0 3px #1f6feb33}
.quick{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap}
button{border:1px solid var(--line);border-radius:7px;background:#21262d;color:var(--text);
padding:7px 10px;cursor:pointer;font:inherit}button:hover{border-color:#6e7681;background:#292f36}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.metric-list{overflow:auto;padding:10px 12px 18px}
.metric-group{margin-top:8px}.metric-group summary{display:flex;align-items:center;justify-content:
space-between;gap:8px;padding:8px 5px;color:var(--muted);font-size:.7rem;font-weight:700;
letter-spacing:.12em;text-transform:uppercase;cursor:pointer;list-style:none}.metric-group summary::-webkit-details-marker{
display:none}.metric-group summary:before{content:'▾';color:var(--accent);font-size:.75rem}.metric-group:not([open])
summary:before{content:'▸'}.metric-group summary small{font-size:.68rem;letter-spacing:0;border:1px solid var(--line);
border-radius:999px;padding:0 6px}.metric-group[hidden],.metric-option[hidden]{display:none!important}.metric-option{display:grid;
grid-template-columns:auto auto minmax(0,1fr);align-items:center;
gap:9px;padding:8px;border-radius:8px;cursor:pointer}.metric-option:hover{background:#ffffff0a}
.metric-option input{accent-color:var(--accent)}
.metric-option strong,.metric-option small{display:block;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}.metric-option strong{font-size:.86rem}.metric-option small{
font-size:.7rem;color:var(--muted)}.swatch{width:9px;height:9px;border-radius:50%;background:var(--metric-colour);
box-shadow:0 0 10px var(--metric-colour)}.main{min-width:0}.tabs{display:flex;gap:6px;overflow:auto;
padding:4px;margin-bottom:10px}.tab[aria-selected="true"]{background:var(--focus);border-color:#388bfd}
.view{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;
box-shadow:0 16px 45px #0004}.view[hidden]{display:none}.plot-wrap{min-height:520px}.note{margin:0;
padding:12px 18px;color:var(--muted);border-top:1px solid var(--line);font-size:.82rem}
.view-tools{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 16px;
border-bottom:1px solid var(--line);background:#0d111766}.view-tools label{display:flex;align-items:center;
gap:7px;color:var(--muted);font-size:.8rem}.view-tools select{border:1px solid var(--line);border-radius:7px;
background:#0d1117;color:var(--text);padding:7px 10px;max-width:260px}
.resizable{resize:vertical;min-height:260px;overflow:auto}.resizable .plot-wrap{min-height:240px;height:calc(100% - 48px)}
.resizable .js-plotly-plot,.resizable .plotly-graph-div{height:100%!important}
.chart-builder{display:grid;grid-template-columns:minmax(180px,1fr) repeat(2,minmax(150px,auto)) auto;
gap:10px;align-items:end}.chart-builder input{width:100%;border:1px solid var(--line);border-radius:7px;
background:#0d1117;color:var(--text);padding:8px 10px}.saved-charts{display:flex;gap:7px;flex-wrap:wrap;
padding:10px 16px;border-bottom:1px solid var(--line)}.saved-chart.active{border-color:var(--accent);color:#79c0ff}
dialog{width:min(760px,calc(100vw - 32px));max-height:85vh;background:var(--panel);color:var(--text);
border:1px solid var(--line);border-radius:14px;padding:0;box-shadow:0 28px 90px #000b}dialog::backdrop{
background:#010409bb}.dialog-head,.dialog-foot{display:flex;align-items:center;justify-content:space-between;
gap:12px;padding:15px 18px;border-bottom:1px solid var(--line)}.dialog-foot{border-top:1px solid var(--line);
border-bottom:0;justify-content:flex-end}.dialog-body{padding:16px;overflow:auto;max-height:65vh}.category-create{
display:flex;gap:8px}.category-create input{flex:1;border:1px solid var(--line);border-radius:7px;background:#0d1117;
color:var(--text);padding:8px 10px}.category-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(150px,.6fr);
gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line)}
.table-wrap{overflow:auto;max-height:650px}table{width:100%;border-collapse:collapse;font-variant-numeric:
tabular-nums}th,td{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;
white-space:nowrap}th{position:sticky;top:0;background:#21262d;color:#c9d1d9;font-size:.75rem;
text-transform:uppercase;letter-spacing:.06em}th:first-child,td:first-child{text-align:left}
tbody tr:hover{
background:#ffffff08}.empty{padding:48px;text-align:center;color:var(--muted)}
@media(max-width:900px){.shell{padding:16px}.dashboard{grid-template-columns:1fr}.splitter{display:none}.sidebar{position:relative;
top:auto;max-height:420px}.cards{grid-template-columns:1fr}.chart-builder{grid-template-columns:1fr}header{align-items:flex-start;flex-direction:column}
.context{text-align:left}}
"""
    script = """
(()=>{
const payload=JSON.parse(document.getElementById('lf-metric-data').textContent);
const byName=Object.fromEntries(payload.series.map(item=>[item.name,item]));
const order=payload.series.map(item=>item.name);
const boxes=[...document.querySelectorAll('.metric-option input')];
let groups=[...document.querySelectorAll('.metric-group')];
const storageKey='lambdaforge:metric-dashboard:'+payload.report_id;
const palettes={
 'Blue ↔ red':[[0,'#2166ac'],[.5,'#f2f2f2'],[1,'#b2182b']],
 'Purple ↔ green':[[0,'#762a83'],[.5,'#f1f1f1'],[1,'#1b7837']],
 'Brown ↔ teal':[[0,'#8c510a'],[.5,'#f5f5f5'],[1,'#01665e']],
 'Accessible':[[0,'#0072b2'],[.5,'#d8d8d8'],[1,'#d55e00']]
};
const load=()=>{try{return JSON.parse(localStorage.getItem(storageKey)||'{}');}catch(_error){return {};}};
let preferences=load();
const save=patch=>{preferences={...preferences,...patch};try{localStorage.setItem(storageKey,
 JSON.stringify(preferences));}catch(_error){/* file:// storage may be disabled */}};
const plotNode=view=>document.querySelector('#'+view+' .js-plotly-plot, #'+view+' .plotly-graph-div');
const format=value=>{if(value===null||value===undefined||!Number.isFinite(Number(value)))return '—';
 return Number(value).toLocaleString(undefined,{maximumSignificantDigits:6});};
const selected=()=>boxes.filter(box=>box.checked).map(box=>box.value);
const layoutOf=view=>{const node=plotNode(view);return node&&node.layout?{...node.layout}:{};};
window.lfPlotConfig={responsive:true,displaylogo:false,scrollZoom:true,
 modeBarButtonsToRemove:['lasso2d','select2d']};
const layouts={snapshot:layoutOf('view-snapshot'),correlation:layoutOf('view-correlation'),
 relationship:layoutOf('view-relationship')};

const defaultGroups=['Validation','Training','Resources & timing','Other'];
const groupContainer=document.getElementById('metric-groups');
const safeGroups=()=>[...document.querySelectorAll('.metric-group')];
function ensureGroup(name){let group=safeGroups().find(value=>value.dataset.group===name);if(group)return group;
 group=document.createElement('details');group.className='metric-group';group.dataset.group=name;group.open=true;
 const summary=document.createElement('summary'),title=document.createElement('span'),count=document.createElement('small');
 title.textContent=name;count.textContent='0';summary.append(title,count);group.appendChild(summary);groupContainer.appendChild(group);return group;}
function saveGroupState(){groups=safeGroups();save({groups:Object.fromEntries(groups.map(value=>[value.dataset.group,value.open]))});}
function bindGroups(){groups=safeGroups();for(const group of groups){if(group.dataset.bound)return;group.dataset.bound='1';
 group.addEventListener('toggle',saveGroupState);}}
function rebuildCategories(){const custom=Array.isArray(preferences.customGroups)?preferences.customGroups:[];
 for(const name of custom)ensureGroup(name);const assignments=preferences.assignments||{};
 document.querySelectorAll('.metric-option').forEach(option=>{const desired=assignments[option.dataset.metric]||option.dataset.defaultGroup;
  ensureGroup(defaultGroups.includes(desired)||custom.includes(desired)?desired:option.dataset.defaultGroup).appendChild(option)});
 for(const group of safeGroups()){const count=group.querySelectorAll('.metric-option').length;group.querySelector('summary small').textContent=String(count);
  if(preferences.groups&&group.dataset.group in preferences.groups)group.open=Boolean(preferences.groups[group.dataset.group]);}
 bindGroups();}
function renderCategoryEditor(){const target=document.getElementById('category-assignments');target.replaceChildren();
 const custom=Array.isArray(preferences.customGroups)?preferences.customGroups:[];
 if(custom.length){const heading=document.createElement('p');heading.className='muted';heading.textContent='Custom categories';target.appendChild(heading);
  const tools=document.createElement('div');tools.className='quick';for(const name of custom){const button=document.createElement('button');
   button.textContent='Delete '+name;button.addEventListener('click',()=>{const assignments={...(preferences.assignments||{})};
    for(const [metric,group] of Object.entries(assignments))if(group===name)delete assignments[metric];
    save({customGroups:custom.filter(value=>value!==name),assignments});safeGroups().find(value=>value.dataset.group===name)?.remove();
    rebuildCategories();renderCategoryEditor();});tools.appendChild(button)}target.appendChild(tools)}
 for(const item of payload.series){const row=document.createElement('label');row.className='category-row';const title=document.createElement('span');
  title.textContent=item.label;const select=document.createElement('select');for(const name of [...defaultGroups,...custom]){const option=document.createElement('option');
   option.value=name;option.textContent=name;select.appendChild(option)}select.value=preferences.assignments?.[item.name]||item.group;
  select.addEventListener('change',()=>{const assignments={...(preferences.assignments||{}),[item.name]:select.value};save({assignments});rebuildCategories()});
  row.append(title,select);target.appendChild(row)}}
rebuildCategories();

const splitter=document.getElementById('sidebar-splitter');
if(Number.isFinite(Number(preferences.sidebarWidth)))document.documentElement.style.setProperty('--sidebar-width',preferences.sidebarWidth+'px');
let splitting=false;splitter.addEventListener('pointerdown',event=>{splitting=true;splitter.classList.add('dragging');splitter.setPointerCapture(event.pointerId)});
splitter.addEventListener('pointermove',event=>{if(!splitting)return;const shell=document.querySelector('.shell').getBoundingClientRect();
 const width=Math.max(220,Math.min(620,event.clientX-shell.left));document.documentElement.style.setProperty('--sidebar-width',width+'px');save({sidebarWidth:Math.round(width)})});
splitter.addEventListener('pointerup',()=>{splitting=false;splitter.classList.remove('dragging')});
splitter.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight'].includes(event.key))return;const current=parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--sidebar-width'))||300;
 const width=Math.max(220,Math.min(620,current+(event.key==='ArrowRight'?20:-20)));document.documentElement.style.setProperty('--sidebar-width',width+'px');save({sidebarWidth:width})});
const resizeObserver=new ResizeObserver(entries=>{const heights={...(preferences.panelHeights||{})};for(const entry of entries){if(entry.target.hidden)continue;
 heights[entry.target.id]=Math.round(entry.contentRect.height);entry.target.querySelectorAll('.js-plotly-plot,.plotly-graph-div').forEach(chart=>Plotly.Plots.resize(chart))}save({panelHeights:heights})});
document.querySelectorAll('.resizable').forEach(panel=>{const height=preferences.panelHeights?.[panel.id];if(Number.isFinite(Number(height)))panel.style.height=height+'px';resizeObserver.observe(panel)});

function customLayout(title,x='',y=''){return {template:'plotly_dark',paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(13,17,23,.72)',
 font:{family:'Inter, ui-sans-serif, system-ui',color:'#c9d1d9'},height:Math.max(280,document.getElementById('view-custom').clientHeight-145),
 margin:{l:72,r:28,t:52,b:62},title,xaxis:{title:{text:x},autorange:true},yaxis:{title:{text:y},autorange:true}}}
function renderCustomChart(spec){const chart=plotNode('custom-chart');if(!chart)return;const items=(spec?.metrics||[]).map(name=>byName[name]).filter(Boolean);
 let traces=[],xLabel='Epoch',yLabel='Observed value';if(spec?.kind==='latest'){traces=[{type:'bar',orientation:'h',x:items.map(item=>item.stats.latest),
  y:items.map(item=>item.label),marker:{color:items.map(item=>item.color)},hovertemplate:'%{y}<br>latest=%{x:.6g}<extra></extra>'}];xLabel='Latest value';yLabel='';}
 else if(spec?.kind==='relationship'&&items.length>=2){const left=items[0],right=items[1],lx=Object.fromEntries(left.x.map((step,index)=>[String(step),left.y[index]])),
  ry=Object.fromEntries(right.x.map((step,index)=>[String(step),right.y[index]])),steps=Object.keys(lx).filter(step=>step in ry).sort((a,b)=>Number(a)-Number(b));
  traces=[{type:'scatter',mode:'markers+lines',x:steps.map(step=>lx[step]),y:steps.map(step=>ry[step]),customdata:steps,
  marker:{color:steps.map(Number),colorscale:'Viridis',size:9,showscale:true,colorbar:{title:'Epoch'}},hovertemplate:'%{x:.6g} / %{y:.6g}<br>epoch=%{customdata}<extra></extra>'}];xLabel=left.label;yLabel=right.label;}
 else{const normalized=spec?.kind==='normalized';traces=items.map(item=>({type:'scatter',mode:'lines+markers',x:item.x,y:normalized?item.normalized:item.y,
  name:item.label,line:{color:item.color,width:2.5},marker:{color:item.color,size:5},hovertemplate:'epoch=%{x}<br>value=%{y:.6g}<extra></extra>'}));
  yLabel=normalized?'Position within observed range':'Observed value'}
 Plotly.react(chart,traces,customLayout(spec?.name||'Saved chart',xLabel,yLabel),window.lfPlotConfig)}
function showCustomChart(id){const charts=Array.isArray(preferences.customCharts)?preferences.customCharts:[],spec=charts.find(item=>item.id===id)||charts[0];
 document.querySelectorAll('.saved-chart').forEach(button=>button.classList.toggle('active',button.dataset.id===spec?.id));if(spec){save({customChartId:spec.id});renderCustomChart(spec)}}
function renderSavedCharts(){const target=document.getElementById('saved-charts');target.replaceChildren();const charts=Array.isArray(preferences.customCharts)?preferences.customCharts:[];
 for(const spec of charts){const button=document.createElement('button');button.className='saved-chart';button.dataset.id=spec.id;button.textContent=spec.name;button.addEventListener('click',()=>showCustomChart(spec.id));
  const remove=document.createElement('button');remove.textContent='×';remove.title='Delete '+spec.name;remove.addEventListener('click',()=>{const remaining=charts.filter(item=>item.id!==spec.id);save({customCharts:remaining,customChartId:remaining[0]?.id});renderSavedCharts();showCustomChart(remaining[0]?.id)});
  const wrap=document.createElement('span');wrap.append(button,remove);target.appendChild(wrap)}if(!charts.length){const empty=document.createElement('span');empty.className='muted';empty.textContent='No saved charts yet.';target.appendChild(empty)}
 showCustomChart(preferences.customChartId)}
renderSavedCharts();

function updateRelationship(){
 const left=byName[document.getElementById('relationship-x').value];
 const right=byName[document.getElementById('relationship-y').value];
 const target=plotNode('view-relationship'); if(!left||!right||!target)return;
 const lx=Object.fromEntries(left.x.map((step,index)=>[String(step),left.y[index]]));
 const ry=Object.fromEntries(right.x.map((step,index)=>[String(step),right.y[index]]));
 const steps=Object.keys(lx).filter(step=>step in ry).sort((a,b)=>Number(a)-Number(b));
 const layout={...layouts.relationship,title:'Metric relationship over shared epochs',
  xaxis:{...(layouts.relationship.xaxis||{}),title:{text:left.label},autorange:true},
  yaxis:{...(layouts.relationship.yaxis||{}),title:{text:right.label},autorange:true}};
 Plotly.react(target,[{type:'scatter',mode:'markers+lines',x:steps.map(step=>lx[step]),
  y:steps.map(step=>ry[step]),customdata:steps,marker:{color:steps.map(Number),colorscale:'Viridis',
  size:9,showscale:true,colorbar:{title:'Epoch'}},hovertemplate:left.label+'=%{x:.6g}<br>'+right.label+
  '=%{y:.6g}<br>epoch=%{customdata}<extra></extra>'}],layout,window.lfPlotConfig);
 save({relationshipX:left.name,relationshipY:right.name});
}
function update(){
 const keys=selected(); const chosen=keys.map(key=>byName[key]).filter(Boolean);
 document.getElementById('selected-count').textContent=String(keys.length);
 const visible=order.map(key=>keys.includes(key));
 const curves=plotNode('view-curves'), normalized=plotNode('view-normalized');
 if(curves)Plotly.restyle(curves,{visible}); if(normalized)Plotly.restyle(normalized,{visible});
 const labels=chosen.map(item=>item.label), colors=chosen.map(item=>item.color);
 const snapshot=plotNode('view-snapshot');
 if(snapshot)Plotly.react(snapshot,[{type:'bar',orientation:'h',x:chosen.map(item=>item.stats.latest),
  y:labels,customdata:chosen.map(item=>item.stats.delta),marker:{color:colors},
  hovertemplate:'%{y}<br>latest=%{x:.6g}<br>delta=%{customdata:+.6g}<extra></extra>'}],
  layouts.snapshot,window.lfPlotConfig);
 const correlation=plotNode('view-correlation');
 const z=keys.map(left=>keys.map(right=>payload.correlations[left]?.[right]??null));
 if(correlation)Plotly.react(correlation,[{type:'heatmap',z,x:labels,y:labels,zmin:-1,zmax:1,
  colorscale:palettes[document.getElementById('correlation-palette').value],
  reversescale:document.getElementById('correlation-reverse').checked,
  colorbar:{title:'Pearson r',tickvals:[-1,-.5,0,.5,1]},
  hovertemplate:'%{y} / %{x}<br>r=%{z:.3f}<extra></extra>'}],layouts.correlation,
  window.lfPlotConfig);
 const body=document.querySelector('#metric-summary tbody'); body.replaceChildren();
 for(const item of chosen){const row=document.createElement('tr');const stats=item.stats;
  for(const value of [item.label,stats.latest_step,stats.latest,stats.delta,stats.minimum,
   stats.maximum,stats.count]){const cell=document.createElement('td');
   cell.textContent=typeof value==='string'?value:format(value);row.appendChild(cell);}body.appendChild(row);}
 document.getElementById('summary-empty').hidden=chosen.length>0;
 save({metrics:keys,palette:document.getElementById('correlation-palette').value,
  reverse:document.getElementById('correlation-reverse').checked});
}

const storedMetrics=Array.isArray(preferences.metrics)?new Set(preferences.metrics):null;
if(storedMetrics)boxes.forEach(box=>box.checked=storedMetrics.has(box.value));
groups=safeGroups();
const palette=document.getElementById('correlation-palette');
if(preferences.palette in palettes)palette.value=preferences.palette;
document.getElementById('correlation-reverse').checked=Boolean(preferences.reverse);
for(const select of document.querySelectorAll('.relationship-select')){
 for(const item of payload.series){const option=document.createElement('option');option.value=item.name;
  option.textContent=item.label;select.appendChild(option);}}
document.getElementById('relationship-x').value=preferences.relationshipX in byName?
 preferences.relationshipX:(payload.defaults[0]||order[0]||'');
document.getElementById('relationship-y').value=preferences.relationshipY in byName?
 preferences.relationshipY:(payload.defaults[1]||order[1]||order[0]||'');
boxes.forEach(box=>box.addEventListener('change',update));
palette.addEventListener('change',update);
document.getElementById('correlation-reverse').addEventListener('change',update);
document.querySelectorAll('.relationship-select').forEach(select=>select.addEventListener('change',
 updateRelationship));
document.getElementById('metric-search').addEventListener('input',event=>{
 const query=event.target.value.trim().toLowerCase();
 groups.forEach(group=>{let matches=0;group.querySelectorAll('.metric-option').forEach(option=>{
  const match=!query||option.dataset.search.includes(query);option.hidden=!match;if(match)matches+=1;});
  group.hidden=Boolean(query)&&matches===0;if(query&&matches)group.open=true;
  else if(!query&&preferences.groups&&group.dataset.group in preferences.groups)
   group.open=Boolean(preferences.groups[group.dataset.group]);});});
document.querySelector('[data-choice="recommended"]').addEventListener('click',()=>{boxes.forEach(box=>
 box.checked=payload.defaults.includes(box.value));update();});
document.querySelector('[data-choice="all"]').addEventListener('click',()=>{boxes.forEach(box=>box.checked=true);
 update();});
document.querySelector('[data-choice="none"]').addEventListener('click',()=>{boxes.forEach(box=>box.checked=false);
 update();});
const categoryDialog=document.getElementById('category-dialog');
document.getElementById('organize-metrics').addEventListener('click',()=>{renderCategoryEditor();categoryDialog.showModal()});
document.querySelectorAll('[data-close-category]').forEach(button=>button.addEventListener('click',()=>categoryDialog.close()));
document.getElementById('add-category').addEventListener('click',()=>{const input=document.getElementById('new-category-name'),name=input.value.trim();
 if(!name)return;const existing=[...defaultGroups,...(preferences.customGroups||[])];if(existing.some(value=>value.toLowerCase()===name.toLowerCase())){input.setCustomValidity('This category already exists.');input.reportValidity();return}
 input.setCustomValidity('');save({customGroups:[...(preferences.customGroups||[]),name]});input.value='';rebuildCategories();renderCategoryEditor()});
document.getElementById('save-custom-chart').addEventListener('click',()=>{const metrics=selected(),kind=document.getElementById('custom-chart-kind').value;
 if(!metrics.length||(kind==='relationship'&&metrics.length<2))return;const input=document.getElementById('custom-chart-name'),charts=Array.isArray(preferences.customCharts)?preferences.customCharts:[];
 const spec={id:(globalThis.crypto?.randomUUID?.()||String(Date.now())),name:input.value.trim()||'Chart '+(charts.length+1),kind,metrics};
 save({customCharts:[...charts,spec],customChartId:spec.id});input.value='';renderSavedCharts();activate('view-custom')});
function activate(target){document.querySelectorAll('.tab').forEach(value=>
 value.setAttribute('aria-selected',String(value.dataset.target===target)));
 document.querySelectorAll('.view').forEach(view=>view.hidden=view.id!==target);save({tab:target});
 const chart=plotNode(target);if(chart)requestAnimationFrame(()=>Plotly.Plots.resize(chart));}
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>activate(tab.dataset.target)));
update();updateRelationship();
if(preferences.tab&&document.getElementById(preferences.tab))activate(preferences.tab);
})();
"""
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>LambdaForge metric dashboard</title><style>"
        + style
        + "</style></head><body><div class=\"shell\"><header><div>"
        "<h1>Run metric dashboard</h1><p>Explore exact persisted scalar observations without "
        "changing the scientific result.</p></div><div class=\"context\">"
        + html.escape(context or "Interactive offline report")
        + "</div></header><div class=\"cards\"><div class=\"card\"><small>Available metrics</small>"
        f"<strong>{len(series)}</strong></div><div class=\"card\"><small>Observed epochs</small>"
        f"<strong>{html.escape(epoch_summary)}</strong></div><div class=\"card\">"
        "<small>Selected metrics</small>"
        f'<strong id="selected-count">{len(defaults)}</strong></div></div>'
        '<div class="dashboard"><aside class="sidebar"><div class="sidebar-head"><h2>Metrics</h2>'
        '<input id="metric-search" class="search" type="search" placeholder="Search metrics…" '
        'aria-label="Search metrics"><div class="quick">'
        '<button data-choice="recommended">Recommended</button>'
        '<button data-choice="all">Select all</button>'
        '<button data-choice="none">Clear</button>'
        '<button id="organize-metrics">Organize</button></div></div>'
        '<div class="metric-list" id="metric-groups">'
        + ("".join(controls) or '<p class="empty">No scalar observations are available.</p>')
        + '</div></aside><div class="splitter" id="sidebar-splitter" role="separator" '
        'aria-label="Resize metric sidebar" aria-orientation="vertical" tabindex="0"></div>'
        '<main class="main"><nav class="tabs" aria-label="Dashboard views">'
        '<button class="tab" data-target="view-curves" aria-selected="true">Curves</button>'
        '<button class="tab" data-target="view-normalized" aria-selected="false">'
        "Compare trends</button>"
        '<button class="tab" data-target="view-snapshot" aria-selected="false">'
        "Latest values</button>"
        '<button class="tab" data-target="view-correlation" aria-selected="false">'
        "Correlations</button>"
        '<button class="tab" data-target="view-relationship" aria-selected="false">'
        "Relationship</button>"
        '<button class="tab" data-target="view-summary" aria-selected="false">'
        'Statistics</button><button class="tab" data-target="view-custom" aria-selected="false">'
        "My charts</button></nav>"
        '<section class="view resizable" id="view-curves"><div class="plot-wrap">'
        + figures["curves"]
        + '</div><p class="note">Raw values retain their original units. Scroll to zoom, '
        "drag to pan and "
        "double-click to restore automatic limits.</p></section>"
        '<section class="view resizable" id="view-normalized" hidden><div class="plot-wrap">'
        + figures["normalized"]
        + "</div><p class=\"note\">Each selected metric is independently mapped to 0–1 "
        "over its observed "
        "range. This compares trajectory shape, not absolute scientific magnitude.</p></section>"
        '<section class="view resizable" id="view-snapshot" hidden><div class="plot-wrap">'
        + figures["snapshot"]
        + "</div><p class=\"note\">Latest persisted value and change from the first observation. "
        "Metrics may "
        "use different units, so compare magnitudes carefully.</p></section>"
        '<section class="view resizable" id="view-correlation" hidden><div class="view-tools">'
        '<label>Colour scale <select id="correlation-palette"><option>Blue ↔ red</option>'
        '<option>Purple ↔ green</option><option>Brown ↔ teal</option><option>Accessible</option>'
        '</select></label><label><input id="correlation-reverse" type="checkbox"> Reverse</label>'
        '</div><div class="plot-wrap">'
        + figures["correlation"]
        + "</div><p class=\"note\">Pearson correlation uses only epochs shared by each pair. "
        "Association is "
        "descriptive and does not imply causality.</p></section>"
        '<section class="view resizable" id="view-relationship" hidden><div class="view-tools">'
        '<label>X metric <select id="relationship-x" class="relationship-select"></select></label>'
        '<label>Y metric <select id="relationship-y" class="relationship-select"></select></label>'
        '</div><div class="plot-wrap">'
        + figures["relationship"]
        + '</div><p class="note">Each point is one shared epoch; colour shows training order. '
        "This is a descriptive relationship, not an independent-sample test or a causal effect."
        "</p></section>"
        '<section class="view" id="view-summary" hidden><div class="table-wrap">'
        '<table id="metric-summary">'
        "<thead><tr><th>Metric</th><th>Latest epoch</th><th>Latest</th><th>Change</th>"
        "<th>Minimum</th>"
        '<th>Maximum</th><th>Points</th></tr></thead><tbody></tbody></table><p id="summary-empty" '
        'class="empty" hidden>Select at least one metric.</p></div></section>'
        '<section class="view resizable" id="view-custom" hidden><div class="view-tools chart-builder">'
        '<label>Chart name <input id="custom-chart-name" placeholder="My comparison"></label>'
        '<label>Chart type <select id="custom-chart-kind"><option value="curves">Curves</option>'
        '<option value="normalized">Normalized trends</option><option value="latest">Latest values</option>'
        '<option value="relationship">X/Y relationship</option></select></label>'
        '<span class="muted">Uses the currently selected metrics</span>'
        '<button id="save-custom-chart">Save chart</button></div>'
        '<div class="saved-charts" id="saved-charts"></div><div class="plot-wrap" id="custom-chart">'
        + figures["custom"]
        + '</div><p class="note">Saved charts are presentation preferences in this HTML only. '
        'They never alter persisted observations.</p></section></main></div></div>'
        '<dialog id="category-dialog"><div class="dialog-head">'
        '<strong>Organize metric categories</strong><button data-close-category>Close</button></div>'
        '<div class="dialog-body"><p class="muted">Create presentation-only categories and assign '
        'any metric. Original names and evidence remain unchanged.</p><div class="category-create">'
        '<input id="new-category-name" placeholder="New category"><button id="add-category">Add</button>'
        '</div><div id="category-assignments"></div></div><div class="dialog-foot">'
        '<button data-close-category>Done</button></div></dialog>'
        f'<script id="lf-metric-data" type="application/json">{payload}</script>'
        f"<script>{script}</script>"
        "</body></html>"
    )
    path = Path(output).expanduser().resolve()
    atomic_write_text(path, document)
    return path


def write_parameter_html(analysis: Mapping[str, Any], parameter: str, output: str | Path) -> Path:
    """Write one parameter response plus its persisted pairwise surfaces."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError as error:
        raise RuntimeError(
            "Install lambdaforge[analysis-report] to export interactive HTML reports."
        ) from error

    responses = analysis.get("response_curves", {})
    response = responses.get(parameter, {}) if isinstance(responses, Mapping) else {}
    points = (
        [value for value in response.get("points", ()) if isinstance(value, Mapping)]
        if isinstance(response, Mapping)
        else []
    )
    if not points:
        live = analysis.get("live_hpo", {})
        parameters = live.get("parameters", ()) if isinstance(live, Mapping) else ()
        detail = next(
            (
                value
                for value in parameters
                if isinstance(value, Mapping) and value.get("parameter") == parameter
            ),
            {},
        )
        live_response = detail.get("response", {}) if isinstance(detail, Mapping) else {}
        points = (
            [
                {
                    "x": value.get("parameter", value.get("label")),
                    "predicted_objective": value.get("objective"),
                    "support_count": value.get("samples", 0),
                }
                for value in live_response.get("points", ())
                if isinstance(value, Mapping)
            ]
            if isinstance(live_response, Mapping)
            else []
        )

    x = [value.get("x", value.get("category")) for value in points]
    y = [
        value.get("predicted_objective", value.get("prediction", value.get("effect")))
        for value in points
    ]
    uncertainty = [value.get("uncertainty") for value in points]
    response_figure = go.Figure()
    if (
        points
        and all(isinstance(value, int | float) for value in y)
        and all(isinstance(value, int | float) for value in uncertainty)
    ):
        numeric_y = [float(value) for value in y if isinstance(value, int | float)]
        numeric_uncertainty = [
            float(value) for value in uncertainty if isinstance(value, int | float)
        ]
        upper = [
            value + spread for value, spread in zip(numeric_y, numeric_uncertainty, strict=True)
        ]
        lower = [
            value - spread for value, spread in zip(numeric_y, numeric_uncertainty, strict=True)
        ]
        response_figure.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        response_figure.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(0, 160, 210, 0.18)",
                name="uncertainty",
                hoverinfo="skip",
            )
        )
    response_figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=(
                "lines+markers"
                if all(
                    isinstance(value, int | float) and not isinstance(value, bool) for value in x
                )
                else "markers"
            ),
            name="response",
            customdata=[value.get("support_count", value.get("support", 0)) for value in points],
            hovertemplate=(
                f"{html.escape(parameter)}=%{{x}}<br>objective=%{{y:.6g}}"
                "<br>support=%{customdata}<extra></extra>"
            ),
        )
    )
    objective_label = _objective_label(analysis)
    response_figure.update_layout(
        title=f"Response: {parameter} → {objective_label}",
        xaxis_title=parameter,
        yaxis_title=objective_label,
        template="plotly_white",
    )
    sections = [plot(response_figure, include_plotlyjs="inline", output_type="div")]

    interactions = analysis.get("interactions", {})
    surfaces = interactions.get("surfaces", {}) if isinstance(interactions, Mapping) else {}
    if isinstance(surfaces, Mapping):
        for pair_name, surface in surfaces.items():
            if parameter not in str(pair_name).split("::") or not isinstance(surface, Mapping):
                continue
            x_values, y_values = list(surface.get("x", ())), list(surface.get("y", ()))
            cells = [value for value in surface.get("cells", ()) if isinstance(value, Mapping)]
            lookup = {(str(value.get("x")), str(value.get("y"))): value for value in cells}
            z_values = [
                [
                    lookup.get((str(left), str(right)), {}).get("predicted_objective")
                    for left in x_values
                ]
                for right in y_values
            ]
            heatmap = go.Figure(
                data=[go.Heatmap(z=z_values, x=x_values, y=y_values, hoverongaps=False)]
            )
            heatmap.update_layout(title=f"Pairwise response: {pair_name}", template="plotly_white")
            sections.append(plot(heatmap, include_plotlyjs=False, output_type="div"))
            if all(
                isinstance(value, int | float) and not isinstance(value, bool)
                for value in (*x_values, *y_values)
            ):
                surface_3d = go.Figure(data=[go.Surface(z=z_values, x=x_values, y=y_values)])
                surface_3d.update_layout(
                    title=f"3D pairwise response: {pair_name}", template="plotly_white"
                )
                sections.append(plot(surface_3d, include_plotlyjs=False, output_type="div"))
    return _write_small_report(
        output,
        f"LambdaForge HPO · {parameter}",
        "".join(sections),
    )


def write_resource_html(cluster: str, series: Mapping[str, Any], output: str | Path) -> Path:
    """Write hoverable CPU/RAM/GPU histories from the console's bounded samples."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError as error:
        raise RuntimeError(
            "Install lambdaforge[analysis-report] to export interactive HTML reports."
        ) from error

    sections: list[str] = []
    for metric in ("cpu", "ram", "gpu"):
        figure = go.Figure()
        for owner, label in (("total", "Cluster total"), ("mine", "My LambdaForge jobs")):
            raw = series.get(f"{owner}_{metric}", ())
            points = (
                [value for value in raw if isinstance(value, Sequence) and len(value) >= 2]
                if isinstance(raw, Sequence) and not isinstance(raw, str | bytes)
                else []
            )
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[value[0] for value in points],
                    y=[value[1] for value in points],
                    mode="lines+markers",
                    name=label,
                    hovertemplate=(
                        "seconds before latest=%{x:.0f}<br>usage=%{y:.2f}%<extra>"
                        + html.escape(label)
                        + "</extra>"
                    ),
                )
            )
        figure.update_layout(
            title=("GPU memory" if metric == "gpu" else metric.upper()),
            xaxis_title="Seconds before latest sample",
            yaxis_title="Usage (%)",
            yaxis={"range": [0, 100]},
            hovermode="x unified",
            template="plotly_white",
        )
        sections.append(
            plot(
                figure,
                include_plotlyjs="inline" if not sections else False,
                output_type="div",
            )
        )
    return _write_small_report(
        output,
        f"LambdaForge resources · {cluster}",
        "".join(sections),
    )


def _write_small_report(output: str | Path, title: str, body: str) -> Path:
    report_id = secrets.token_hex(12)
    document = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>"
        "body{font:16px system-ui;max-width:1400px;margin:auto;padding:2rem;color:#20242b}"
        ".resizable-report-panel{resize:vertical;overflow:hidden;min-height:280px;"
        "border:1px solid #d0d7de;border-radius:10px;margin:1rem 0}.resizable-report-panel "
        ".plotly-graph-div{height:100%!important}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>{body}<script>(()=>{{const key='lambdaforge:report:"
        + report_id
        + "';let sizes={};try{sizes=JSON.parse(localStorage.getItem(key)||'{}')}catch(_e){};"
        "document.querySelectorAll('.plotly-graph-div').forEach((chart,index)=>{const parent="
        "chart.parentElement;if(!parent)return;const panel=document.createElement('div');"
        "panel.className='resizable-report-panel';panel.dataset.index=String(index);"
        "if(Number.isFinite(Number(sizes[index])))panel.style.height=sizes[index]+'px';"
        "parent.insertBefore(panel,chart);panel.appendChild(chart);new ResizeObserver(entries=>{"
        "for(const entry of entries){sizes[index]=Math.round(entry.contentRect.height);try{"
        "localStorage.setItem(key,JSON.stringify(sizes))}catch(_e){};Plotly.Plots.resize(chart)}})"
        ".observe(panel)});})();</script></body></html>"
    )
    path = Path(output).expanduser().resolve()
    atomic_write_text(path, document)
    return path


def _objective_label(analysis: Mapping[str, Any]) -> str:
    objective = analysis.get("objective", {})
    if not isinstance(objective, Mapping):
        return "Selection objective"
    metrics = objective.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        return "Composite selection score"
    metric = str(objective.get("metric", "Selection objective"))
    if metric == "__lambdaforge_utility__":
        return "Composite selection score"
    return metric.replace("_", " ").title()


__all__ = ["write_html", "write_metric_html", "write_parameter_html", "write_resource_html"]
