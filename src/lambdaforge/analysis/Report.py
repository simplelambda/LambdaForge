"""Optional self-contained Plotly export for Study Analysis."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lambdaforge.work.atomic import atomic_write_text


def write_html(analysis: Mapping[str, Any], output: str | Path) -> Path:
    """Write an offline report; Plotly remains isolated in the optional extra."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError as error:
        raise RuntimeError(
            "Install lambdaforge[analysis-report] to export interactive HTML reports."
        ) from error

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
        figure.update_layout(title="Top-region importance")
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
                    surface_3d = go.Figure(
                        data=[go.Surface(z=z_values, x=x_values, y=y_values)]
                    )
                    surface_3d.update_layout(title=f"Pairwise predictive surface: {name}")
                    sections.append(plot(surface_3d, include_plotlyjs=False, output_type="div"))
    if candidates:
        parameter_names = sorted(
            {
                str(name)
                for candidate in candidates
                for name in candidate.get("parameters", {})
            }
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
                        value.get("boundary_enrichment", 0)
                        if isinstance(value, Mapping)
                        else 0
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
        f"<p>Status: {status}</p>{''.join(sections)}<h2>Diagnostics</h2>"
        f"<pre>{diagnostic}</pre><h2>Findings</h2>{finding_html}"
        "<details><summary>Complete reproducible analysis JSON</summary>"
        f"<pre>{raw}</pre></details></body></html>"
    )
    path = Path(output).expanduser().resolve()
    atomic_write_text(path, document)
    return path


__all__ = ["write_html"]
