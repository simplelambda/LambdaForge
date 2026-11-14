"""Tiny public Work fixture; scientific formats remain the consumer's responsibility."""

from lambdaforge import Work


class IsolatedProducer(Work):
    def run(self):
        tool = self.tools.runtime("example")
        tool.preflight()
        source = self.temp_dir / "input.txt"
        source.write_text("fixture")
        output = self.checkpoints.file(
            "produced.txt",
            build=lambda path: tool.run(["python", tool.entrypoint("extract"), source, path]),
            validate=lambda path: path.read_text() == "FIXTURE",
        )
        return {"content": output.read_text()}
