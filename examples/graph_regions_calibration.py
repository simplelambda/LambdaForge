"""Generic graph regions and independent negative-object calibration (CPU, synthetic).

Run: python examples/graph_regions_calibration.py
No scientific domain is assumed and no records, files or Jobs are created.
This is the topology/calibration example, NOT the pending full FGW workflow.
"""

import numpy as np

from lambdaforge.stats import EmpiricalNullCalibrator, bootstrap_quantile
from lambdaforge.topology import StableRegionSelector, SuperlevelMergeTree


def main() -> None:
    field = np.array([5.0, 2.0, 4.0, 1.0, 3.0])
    edges = np.column_stack([np.arange(4), np.arange(1, 5)])
    tree = SuperlevelMergeTree().fit(
        field, edges, vertex_weights=np.array([1.0, 2.0, 1.0, 2.0, 1.0])
    )
    selected = StableRegionSelector(min_width=1, min_iou=0.5, overlap="disjoint").select(tree)
    print("Birth/death pairs (None = essential):", tree.persistence_diagram())
    for region in selected.regions:
        print(
            "Region",
            region.branch_id,
            "interval",
            region.interval,
            "membership",
            region.membership,
            "weighted mass",
            region.mass,
        )

    # The candidate/score construction below is fixed and IDENTICAL for controls
    # and the new object. Candidates inside an object are not independent controls.
    rng = np.random.default_rng(42)
    controls = rng.normal(size=(199, 4))
    null = EmpiricalNullCalibrator(tail="upper", alpha=0.05).fit_groups(
        controls.ravel(), np.repeat(np.arange(len(controls)), controls.shape[1])
    )
    result = null.evaluate_group(np.array([1.0, 0.5, 2.5, 0.0]))
    print(
        "New object's maximum:",
        result.score,
        "p-value:",
        result.p_value,
        "reject:",
        result.reject,
        "resolution:",
        result.resolution,
    )
    quantile = bootstrap_quantile(controls.max(axis=1), 0.95, seed=42, n_resamples=500)
    print(
        "Control-maximum 95% quantile:",
        quantile.estimate,
        "approximate bootstrap CI:",
        quantile.interval,
    )


if __name__ == "__main__":
    main()
