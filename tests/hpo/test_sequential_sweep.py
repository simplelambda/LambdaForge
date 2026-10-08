from __future__ import annotations

import random

import pytest

from lambdaforge.hpo.SequentialSweep import PairedSweepSequentialAnalyzer


def _constant_evidence(effects: list[float], blocks: int) -> dict[int, dict[int, float]]:
    return {
        trial: {seed: 0.5 + effect for seed in range(blocks)}
        for trial, effect in enumerate(effects)
    }


def test_reference_sweep_resolves_only_authored_primary_family() -> None:
    analyzer = PairedSweepSequentialAnalyzer(
        mode="max", bounds=(0.0, 1.0), practical_margin=0.05, reference=0
    )
    decision = analyzer.evaluate(
        _constant_evidence([0.0, 0.1, -0.1, 0.01], 500), seed_order=range(500)
    )

    assert decision.stop
    assert decision.conclusion == "PARTIALLY_ORDERED"
    assert decision.primary_comparisons == 3
    assert [value.relation for value in decision.relations] == [
        "MATERIALLY_BETTER",
        "MATERIALLY_WORSE",
        "PRACTICALLY_EQUIVALENT",
    ]


def test_non_reference_sweep_can_resolve_a_practical_top_set() -> None:
    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0), practical_margin=0.05)
    decision = analyzer.evaluate(
        _constant_evidence([0.0, -0.01, -0.2, -0.3], 600), seed_order=range(600)
    )

    assert decision.stop
    assert decision.conclusion == "PRACTICAL_TOP_SET"
    assert decision.practical_top_set == (0, 1)
    assert decision.inferior_set == (2, 3)
    evidence = decision.to_dict()["formal_sequential_evidence"]
    assert evidence["simultaneous_coverage_level"] == 0.95


def test_missing_practical_margin_never_claims_equivalence() -> None:
    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0))
    decision = analyzer.evaluate(_constant_evidence([0.0, 0.0], 500), seed_order=range(500))

    assert not decision.stop
    assert decision.conclusion == "UNRESOLVED"
    assert "No practical_margin" in decision.reason


def test_pm_eb_is_tighter_than_historical_hoeffding_for_low_variance_pairs() -> None:
    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0))
    differences = [0.2 + ((index % 3) - 1) * 0.005 for index in range(150)]
    lower, upper = analyzer._pm_eb_interval(differences, 0.05)  # noqa: SLF001
    old_lower, old_upper = analyzer.hoeffding_interval(differences, objective_span=1.0, alpha=0.05)

    assert upper - lower < old_upper - old_lower


def test_pm_eb_controls_false_discovery_under_repeated_peeking() -> None:
    """Deterministic Monte Carlo regression for the anytime-valid null path."""
    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0))
    generator = random.Random(1729)
    false_discoveries = 0
    paths = 200
    for _path in range(paths):
        differences: list[float] = []
        for _look in range(96):
            differences.append(float((generator.random() < 0.5) - (generator.random() < 0.5)))
            if len(differences) < 2:
                continue
            lower, upper = analyzer._pm_eb_interval(differences, 0.05)  # noqa: SLF001
            if lower > 0.0 or upper < 0.0:
                false_discoveries += 1
                break

    # This is a fixed-seed regression envelope, not a proof or a replacement for the formal CS.
    assert false_discoveries / paths <= 0.05


def test_pm_eb_stops_in_tens_of_low_variance_blocks_for_a_large_effect() -> None:
    analyzer = PairedSweepSequentialAnalyzer(
        mode="max", bounds=(0.0, 1.0), practical_margin=0.02, reference=0
    )
    values: dict[int, dict[int, float]] = {0: {}, 1: {}}
    decision = None
    for block in range(1, 65):
        values[0][block] = 0.5
        values[1][block] = 0.8 + ((block % 5) - 2) * 0.002
        decision = analyzer.evaluate(values, seed_order=range(1, block + 1))
        if decision.stop:
            break

    assert decision is not None
    assert decision.stop
    assert decision.blocks <= 64
    assert decision.conclusion == "BETTER_THAN_REFERENCE"


def test_nineteen_treatment_reference_family_uses_only_eighteen_primary_tests() -> None:
    analyzer = PairedSweepSequentialAnalyzer(
        mode="max", bounds=(0.0, 1.0), practical_margin=0.05, reference=0
    )
    effects = [0.0] + [(-0.3, 0.0, 0.3)[index % 3] for index in range(18)]

    decision = analyzer.evaluate(_constant_evidence(effects, 1000), seed_order=range(1000))

    assert decision.stop
    assert decision.conclusion == "PARTIALLY_ORDERED"
    assert decision.primary_comparisons == 18
    assert len(decision.relations) == 18
    assert {value.relation for value in decision.relations} == {
        "MATERIALLY_BETTER",
        "MATERIALLY_WORSE",
        "PRACTICALLY_EQUIVALENT",
    }


def test_project_stream_evidence_preserves_append_only_acquisition_order() -> None:
    from lambdaforge.reproducibility.SeedProvider import ProjectSeedStream

    order = ProjectSeedStream("ordered-regression").values(80)
    assert tuple(sorted(order)) != order
    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0))
    values: dict[int, dict[int, float]] = {0: {}, 1: {}}
    prefix: tuple[int, ...] = ()
    for ordinal, seed in enumerate(order):
        values[0][seed] = 0.2 + 0.003 * ordinal
        values[1][seed] = 0.8 - 0.004 * ordinal
        decision = analyzer.evaluate(values, seed_order=order)
        assert decision.evidence_seeds[: len(prefix)] == prefix
        prefix = decision.evidence_seeds
    expected = analyzer._pm_eb_interval(  # noqa: SLF001
        [values[0][seed] - values[1][seed] for seed in order], 0.05
    )
    assert decision.relations[0].lower == expected[0]
    assert decision.relations[0].upper == expected[1]
    assert decision.to_dict()["evidence_seeds"] == list(order)
    # Re-loading JSON mappings or reversing worker completion order cannot change inference.
    reversed_values = {trial: dict(reversed(list(rows.items()))) for trial, rows in values.items()}
    assert analyzer.evaluate(reversed_values, seed_order=order) == decision


def test_incomplete_block_seals_prefix_even_if_lookahead_completed() -> None:
    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0))
    values = {0: {54: 0.2, 4: 0.2, 7: 0.2}, 1: {54: 0.7, 7: 0.7}}
    assert analyzer.evaluate(values, seed_order=[54, 4, 7]).evidence_seeds == (54,)
    values[1][4] = 0.7
    assert analyzer.evaluate(values, seed_order=[54, 4, 7]).evidence_seeds == (54, 4, 7)


@pytest.mark.parametrize("order", [None, [4, 4], [True], [7]])
def test_missing_or_conflicting_order_fails_closed(order: list[int] | None) -> None:
    analyzer = PairedSweepSequentialAnalyzer(mode="max", bounds=(0.0, 1.0))
    with pytest.raises(ValueError):
        analyzer.evaluate({0: {4: 0.5}, 1: {4: 0.6}}, seed_order=order)
