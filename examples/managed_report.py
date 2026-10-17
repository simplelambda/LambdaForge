"""Rebuild only presentation from an exact retained artifact, without model/inference code.

Bind ``previous`` as a native ResultInput selecting one original Run/Attempt.
The producer must explicitly retain a JSON catalogue named ``saved-predictions``.
This minimal inspector preserves every consumer record; extend its JS renderer in your project.
"""

import shutil

from lambdaforge import Work
from lambdaforge.work import ResultInput


class RetainedPredictionReport(Work):
    def run(self, previous: ResultInput) -> str:
        retained = previous.artifact("saved-predictions")
        bundle = self.outputs.html_bundle("inspector", section="predictions", title="Predictions")
        bundle.asset("catalogue", filename="catalogue.json", media_type="application/json").build(
            lambda destination: shutil.copyfile(retained, destination)
        )
        bundle.entry.write_text("""<h1>Retained predictions</h1>
<p id="status">Loading saved catalogue…</p><select id="records"></select><pre id="detail"></pre>
<script>
(async () => {
  const status = document.querySelector('#status'), select = document.querySelector('#records');
  try {
    const bytes = await lambdaforgeReport.load('catalogue');
    const rows = JSON.parse(new TextDecoder().decode(bytes));
    rows.forEach((row, index) => {
      const option = document.createElement('option'); option.value = index;
      option.textContent = row.id ?? String(index); select.append(option);
    });
    function show() {
      document.querySelector('#detail').textContent = JSON.stringify(rows[select.value], null, 2);
    }
    select.onchange = show; show();
    status.textContent = rows.length + ' retained records, none omitted';
  } catch (error) { status.textContent = 'Presentation error: ' + error.message; }
})();
</script>""")
        return "Presentation rebuilt from exact retained evidence; no inference executed."
