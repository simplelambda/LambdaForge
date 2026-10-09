"""Offline reports of ordinary Work evidence, without Study inference or consumer code."""

# Embedded browser-native source follows the existing Report renderer's formatting policy.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lambdaforge.analysis.Report import _project_sections, write_metric_html
from lambdaforge.work.atomic import atomic_write_text


def _evidence(value: Any) -> str:
    """Escape every authored value; structured evidence is readable, not executable markup."""
    if isinstance(value, Mapping):
        return (
            "<dl>"
            + "".join(
                f"<dt>{html.escape(str(key).replace('_', ' '))}</dt><dd>{_evidence(item)}</dd>"
                for key, item in value.items()
            )
            + "</dl>"
        )
    if isinstance(value, list | tuple):
        return "<ul>" + "".join(f"<li>{_evidence(item)}</li>" for item in value) + "</ul>"
    return html.escape("—" if value is None else str(value))


def write_work_html(
    overview: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    sections: Sequence[Mapping[str, str]] = (),
) -> Path:
    """Render exact Attempt evidence, with the mature metric dashboard and HTML sandbox.

    Nothing is recomputed scientifically. This explicit report includes bounded log tails and
    downsampled curves; the native export retains the complete original scalar/log files.
    """
    documents = list(sections)
    evidence = []
    all_curves: dict[str, Any] = {}
    names: dict[str, str] = {}
    groups: dict[str, str] = {}
    preferred: list[str] = []
    for detail in attempts:
        selected = detail["selected"]
        label = f"{selected['run_id']} / {selected['attempt_id']}"
        if detail.get("curves"):
            aliases = detail.get("chart_filter", {}).get("display_names", {})
            for metric, points in detail["curves"].items():
                # Namespacing is presentation only: no shared mean, best or final values are
                # invented across Attempts. The complete raw streams remain in native exports.
                key = json.dumps([selected["run_id"], selected["attempt_id"], metric])
                all_curves[key] = points
                names[key] = f"{label} · {aliases.get(metric, metric)}"
                groups[key] = label
                if len(preferred) < 4 and (not preferred or groups[preferred[0]] == label):
                    preferred.append(key)
        evidence.append(
            f"<details open><summary>{html.escape(label)} · {html.escape(str(selected.get('status')))}</summary>"
            + "<h3>Final metrics</h3>"
            + _evidence(selected.get("metrics", {}))
            + "<h3>Primary result</h3>"
            + _evidence(detail.get("result"))
            + "<h3>Registered outputs</h3>"
            + _evidence(detail.get("outputs", {}))
            + "<h3>Artifacts</h3>"
            + _evidence(detail.get("artifacts", ()))
            + "<h3>Datasets</h3>"
            + _evidence(detail.get("datasets", {}))
            + "<h3>Current logical Run checkpoints (shared across Attempts)</h3>"
            + _evidence(detail.get("checkpoints", ()))
            + "<h3>Failure</h3>"
            + _evidence(selected.get("failure"))
            + "<h3>Provenance</h3>"
            + _evidence(detail.get("provenance", {}))
            + "<h3>Persisted log tail (bounded)</h3><pre>"
            + html.escape(str(detail.get("log", "")))
            + "</pre></details>"
        )
    if all_curves:
        with tempfile.TemporaryDirectory(prefix="lambdaforge-work-metrics-") as temporary:
            try:
                path = write_metric_html(
                    all_curves,
                    tuple(all_curves),
                    Path(temporary) / "metrics.html",
                    display_names=names,
                    preferred_names=preferred,
                    series_groups=groups,
                    step_label="Step / observation",
                )
                documents.append(
                    {
                        "name": next(
                            name
                            for index in range(len(documents) + 1)
                            if (name := f"work-metrics-{index}")
                            not in {section["name"] for section in documents}
                        ),
                        "title": "Metrics",
                        "label": "Exact Run / Attempt series",
                        "html": path.read_text(encoding="utf-8"),
                    }
                )
            except RuntimeError:
                # Reports without the optional Plotly extra still expose exact final scalars.
                pass
    buttons, panels, script = _project_sections(documents)
    metadata = json.dumps({"report_id": overview["execution_id"]}, ensure_ascii=True).replace(
        "<", "\\u003c"
    )
    title = html.escape(str(overview.get("name", "Work")))
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="frame-src 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} · Work results</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#0d1117;color:#e6edf3;font:16px system-ui}}
header,main{{padding:24px;max-width:1400px;margin:auto}}nav{{display:flex;gap:10px;flex-wrap:wrap}}
button{{padding:12px 18px;border:1px solid #30363d;border-radius:8px;background:#21262d;color:inherit;cursor:pointer}}
button[aria-selected=true]{{border-color:#58a6ff}}.panel,details{{padding:20px;background:#161b22;border:1px solid #30363d;border-radius:12px;margin:16px 0}}
summary{{cursor:pointer;font-weight:600}}dl{{display:grid;grid-template-columns:minmax(150px,25%) 1fr;gap:8px 20px}}
dt{{color:#8b949e}}dd{{margin:0;overflow-wrap:anywhere}}dd dl{{display:block}}dd dt{{margin-top:8px}}
pre{{overflow:auto;max-height:500px;white-space:pre-wrap}}h1{{margin-bottom:4px}}.note{{color:#8b949e}}
[hidden]{{display:none!important}}</style></head><body><header><h1>{title}</h1>
<p>{html.escape(str(overview.get("status", "unknown")).upper())} · {html.escape(str(overview["execution_id"]))}</p>
<p class="note">Persisted Work evidence · Attempts are kept separate · no statistical Study conclusions are inferred.</p>
<nav><button class="tab" data-target="overview" aria-selected="true">Overview</button>
<button class="tab" data-target="attempts" aria-selected="false">Runs &amp; Attempts</button>{buttons}</nav></header>
<main><section class="view" id="overview"><article class="panel">{_evidence(overview)}</article></section>
<section class="view" id="attempts" hidden>{"".join(evidence) or "<p>No finalized Attempt evidence yet.</p>"}</section>{panels}</main>
<script id="lf-study-data" type="application/json">{metadata}</script>
<script>document.querySelectorAll('button.tab').forEach(b=>b.addEventListener('click',()=>{{
document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==b.dataset.target);
document.querySelectorAll('button.tab').forEach(t=>t.setAttribute('aria-selected',String(t===b)));}}));</script>{script}</body></html>"""
    target = Path(output).expanduser().absolute()
    atomic_write_text(target, document)
    return target
