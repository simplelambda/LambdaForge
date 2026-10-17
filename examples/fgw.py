"""Compare weighted attributed objects; install lambdaforge[optimal-transport]."""

import numpy as np

from lambdaforge.distances import FusedGromovWassersteinDistance, StructuredMeasure


def main() -> None:
    # Geometry, attributes and relative mass are separate, explicit inputs.
    source = StructuredMeasure.from_points(
        [[0.0, 0.0], [1.0, 0.0]],
        features=[[0.0, 1.0], [1.0, 0.0]],
        weights=[1.0, 3.0],
    )
    target = StructuredMeasure.from_points(
        [[0.0], [1.5], [3.0]],
        features=[[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]],
        weights=[1.0, 1.0, 2.0],
    )
    result = FusedGromovWassersteinDistance(alpha=0.5).solve(source, target)
    np.testing.assert_allclose(result.transport.sum(axis=1), source.weights)
    np.testing.assert_allclose(result.transport.sum(axis=0), target.weights)
    print(f"FGW q=2 objective: {result.value:.6g}; locally converged: {result.converged}")
    print("Node coupling:\n", result.transport)


if __name__ == "__main__":
    main()
