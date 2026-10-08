"""CPU-only calibration regression; this simulation is not a mathematical proof.

Run ``python benchmarks/sequential_sweep_audit.py``. Inspect every prefix, with
project-stream seeds, three treatments and a shared reference under a known null.
The repeated-look error event is *any* excluded true paired mean in the family.
"""

from __future__ import annotations

import json
import random
from time import perf_counter

from lambdaforge.hpo.SequentialSweep import PairedSweepSequentialAnalyzer
from lambdaforge.reproducibility.SeedProvider import ProjectSeedStream


def calibration(*, paths: int = 200, blocks: int = 96) -> dict[str, object]:
    rng = random.Random(82731)
    order = ProjectSeedStream("sweep-audit").values(blocks)
    results: dict[str, object] = {}
    for name, noise in (("degenerate", 0.0), ("low_variance", 0.02), ("high_variance", 1.0)):
        started = perf_counter()
        violations = 0
        analyzer = PairedSweepSequentialAnalyzer(
            mode="max", bounds=(0, 1), practical_margin=0.02, reference=0
        )
        for _ in range(paths):
            evidence: dict[int, dict[int, float]] = {trial: {} for trial in range(4)}
            for seed in order:
                shared = rng.random()
                for trial in evidence:
                    evidence[trial][seed] = (1 - noise) * shared + noise * rng.random()
                decision = analyzer.evaluate(evidence, seed_order=order)
                if any(relation.lower > 0 or relation.upper < 0 for relation in decision.relations):
                    violations += 1
                    break
        results[name] = {
            "paths": paths,
            "maximum_looks": blocks,
            "primary_comparisons": 3,
            "family_alpha": 0.05,
            "paths_excluding_true_mean_at_any_look": violations,
            "empirical_family_error": violations / paths,
            "cpu_wall_seconds": perf_counter() - started,
        }
    return {
        "policy": "paired-pm-eb-cs-v2",
        "seed_order_is_non_numeric": order != tuple(sorted(order)),
        "interpretation": "fixed-seed regression, not proof or real-GPU performance",
        "scenarios": results,
    }


if __name__ == "__main__":
    print(json.dumps(calibration(), indent=2))
