"""Disposable native Work fixture for matrix-block interruption and recovery tests."""

import numpy as np

from lambdaforge import Work
from lambdaforge.distances import (
    FGWMatrixBuilder,
    FusedGromovWassersteinDistance,
    StructuredMeasure,
)


class BlockMatrixWork(Work):
    """Fail one POT comparison once; all checkpoint I/O uses native Work services."""

    def run(self, error_policy: str = "raise") -> dict:
        objects = [
            StructuredMeasure.from_points(np.arange(n)[:, None], features=np.arange(n)[:, None] / n)
            for n in (2, 3, 4)
        ]
        distance = FusedGromovWassersteinDistance()
        original = distance.solve

        def solve(left, right):
            calls = (
                self.checkpoints.load_json("calls.json")
                if self.checkpoints.exists("calls.json")
                else 0
            )
            self.checkpoints.save_json("calls.json", calls + 1)
            if calls == 2 and not self.checkpoints.exists("interrupted.json"):
                self.checkpoints.save_json("interrupted.json", True)
                raise RuntimeError("deliberate block interruption")
            return original(left, right)

        distance.solve = solve
        result = FGWMatrixBuilder(distance, block_size=1, error_policy=error_policy).build(
            objects, symmetric=True, work=self
        )
        if not result.complete:
            raise RuntimeError("explicit partial matrix; refuse scientific success")
        return {"values": result.values.tolist(), "calls": self.checkpoints.load_json("calls.json")}
