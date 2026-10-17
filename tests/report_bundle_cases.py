"""Disposable native Work fixtures: retained presentation bytes, never training."""

from lambdaforge import Work


class ManagedReportWork(Work):
    def run(self, *, broken: bool = False) -> float:
        self.outputs.value("score", 0.7)
        bundle = self.outputs.html_bundle("inspector", section="proteins", title="Proteins")
        bundle.entry.write_text("<h1>Retained-only inspector</h1>")
        asset = bundle.asset("coordinates", filename="surface.f32")
        if not broken:
            asset.write_bytes(b"exact retained coordinates")
        return 0.7
