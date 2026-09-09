from benchmarks.hpo_scientific_design import run_benchmark


def test_scientific_design_preserves_optimization_quality_at_equal_budget() -> None:
    result = run_benchmark(budget=6)

    assert result["summary"]["optimization_quality_preserved"] is True
    assert result["summary"]["scientific_design_mean_regret"] <= result["summary"][
        "optimization_only_mean_regret"
    ] + result["summary"]["practical_regret_tolerance"] + 1e-12
    for problem in ("smooth", "interaction", "flat-feature"):
        assert result[problem]["scientific_design"]["runs"] == 6
