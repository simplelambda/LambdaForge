"""Real POT: weighted unequal objects, bounded matrices and a relational prototype.

Install the optional group, then run: python examples/fgw_barycenter.py
No domain data, records, files, Jobs or clusters are used.
"""

import numpy as np

from lambdaforge.distances import (
    FGWBarycenter,
    FGWMatrixBuilder,
    FusedGromovWassersteinDistance,
    StructuredMeasure,
)


def main() -> None:
    objects = [
        StructuredMeasure.from_points([[0.0], [1.0]], features=[[0.0], [1.0]], weights=[1.0, 3.0]),
        StructuredMeasure.from_points(
            [[0.0], [0.5], [1.0]], features=[[0.0], [0.6], [1.0]], weights=[1.0, 2.0, 1.0]
        ),
    ]
    distance = FusedGromovWassersteinDistance(alpha=0.5)
    builder = FGWMatrixBuilder(distance, block_size=2)
    matrix = builder.build(objects, symmetric=True)
    print("Complete matrix (not a convergence claim):", matrix.complete)
    print(matrix.values)
    barycenter = FGWBarycenter(2, seed=42, n_init=2, max_iter=20).fit(
        objects, object_weights=[1.0, 2.0], support_weights=[1.0, 1.0]
    )
    print("Final weighted postcheck objective:", barycenter.objective)
    print("Both change criteria + postcheck convergence:", barycenter.converged)
    print("Relational diagonal, kept intact:", np.diag(barycenter.prototype.structure))
    print("Restart diagnostics:", barycenter.restarts)
    print("Warnings:", barycenter.warnings)
    print("Rectangular comparison to prototype:")
    print(builder.build(objects, [barycenter.prototype]).values)


if __name__ == "__main__":
    main()
