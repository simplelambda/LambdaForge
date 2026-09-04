"""Optional self-contained Plotly export for Study Analysis."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
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


def write_metric_html(
    curves: Mapping[str, Any],
    names: Sequence[str],
    output: str | Path,
    *,
    display_names: Mapping[str, Any] | None = None,
) -> Path:
    """Write exact interactive learning curves for one Run."""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError as error:
        raise RuntimeError(
            "Install lambdaforge[analysis-report] to export interactive HTML reports."
        ) from error

    figure = go.Figure()
    aliases = display_names or {}
    for name in names:
        raw = curves.get(name, ())
        points = (
            [value for value in raw if isinstance(value, Mapping)]
            if isinstance(raw, Sequence)
            else []
        )
        x = [value.get("step") for value in points]
        y = [value.get("value") for value in points]
        if not x:
            continue
        label = str(aliases.get(name, name)).replace("_", " ").title()
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                hovertemplate=f"epoch=%{{x}}<br>{html.escape(label)}=%{{y:.6g}}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Learning curves",
        xaxis_title="Epoch",
        yaxis_title="Metric value",
        hovermode="x unified",
        template="plotly_white",
    )
    body = plot(figure, include_plotlyjs="inline", output_type="div")
    return _write_small_report(output, "LambdaForge learning curves", body)


def write_parameter_html(
    analysis: Mapping[str, Any], parameter: str, output: str | Path
) -> Path:
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
        points = [
            {
                "x": value.get("parameter", value.get("label")),
                "predicted_objective": value.get("objective"),
                "support_count": value.get("samples", 0),
            }
            for value in live_response.get("points", ())
            if isinstance(value, Mapping)
        ] if isinstance(live_response, Mapping) else []

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
            value + spread
            for value, spread in zip(numeric_y, numeric_uncertainty, strict=True)
        ]
        lower = [
            value - spread
            for value, spread in zip(numeric_y, numeric_uncertainty, strict=True)
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
                    isinstance(value, int | float) and not isinstance(value, bool)
                    for value in x
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


def write_resource_html(
    cluster: str, series: Mapping[str, Any], output: str | Path
) -> Path:
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
    document = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>"
        "body{font:16px system-ui;max-width:1400px;margin:auto;padding:2rem;color:#20242b}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>{body}</body></html>"
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
