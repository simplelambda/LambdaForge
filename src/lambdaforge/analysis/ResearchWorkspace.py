"""Small semantic shell for the offline research workspace; no scientific computation."""

# HTML stays readable as native source.
# ruff: noqa: E501


def workspace_html() -> str:
    """Return static controls. Catalog options are searched dynamically, not repeated in HTML."""
    return """
<section class="view" id="study-research" hidden>
 <div class="tools"><h2>Research inbox</h2><button id="research-search-open">Search everything · Ctrl / ⌘ K</button></div>
 <div class="cards" id="research-health"></div>
 <p class="note">Retrospective associations, not causal effects. Inspect a finding to see support,
 uncertainty, multiplicity, limitations and the underlying observations.</p>
 <h2>Configured research questions</h2><div id="research-questions" class="question-grid"></div>
 <h2>Exploratory / integrity / resource signals</h2><div id="research-inbox"></div>
 <details class="panel"><summary class="tools">All exploratory findings</summary>
 <div id="research-all-findings"></div></details>
</section>
<section class="view" id="study-metrics" hidden><div class="panel">
 <div class="tools"><h2>Metrics & health</h2><label>Search <input id="research-metric-search" type="search" placeholder="Name, alias, description, category…"></label>
 <label><input id="research-show-all" type="checkbox">Show constants / missing / hidden</label>
 <input type="hidden" id="research-category"><details id="research-category-tree"><summary>Categories · all</summary><div id="research-category-branches"></div></details>
 <label>Sort <select id="research-sort"><option value="priority">Priority</option><option value="coverage">Coverage</option><option value="name">Name</option><option value="spread">Spread</option></select></label></div>
 <div class="table-wrap"><table><thead><tr><th>Metric / meaning</th><th>Category · split</th><th>Unit / direction</th>
 <th>Evidence</th><th>Spread</th><th>Health / aggregation</th><th>Open</th></tr></thead><tbody id="research-metrics-body"></tbody></table></div>
 <p class="note">Show all preserves constant evidence. Spread uses authored range/practical scale, or relative magnitude;
 unknown units and aggregation remain explicitly unknown. Redundancy never deletes recorded metrics.</p>
 <details><summary class="tools">Redundancy groups</summary><div id="research-redundancy"></div></details>
</div><div class="panel"><h2>Metric families</h2><div id="research-family-list" class="tools"></div>
 <div id="research-family-plot" class="plot"></div><p id="research-family-status" class="note"></p></div></section>
<dialog id="research-detail"><div class="tools"><h2 id="research-detail-title"></h2><button id="research-detail-close">Close</button></div><div id="research-detail-body"></div></dialog>
<dialog id="research-search"><div class="tools"><h2>Search / metric picker</h2><button id="research-search-close">Close</button></div>
 <input id="research-global-query" type="search" placeholder="Metrics, aliases, families, parameters, findings, Trials, saved views…" aria-label="Search research workspace">
 <div id="research-search-results"></div></dialog>
"""
