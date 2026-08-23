"""Focused deterministic random-search tests."""

from lambdaforge.hpo import RandomSearch


def test_random_search_is_reproducible_unique_and_work_parameter_shaped() -> None:
    search = RandomSearch(
        {
            "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "width": {"type": "choice", "values": [8, 16]},
        },
        seed=9,
    )
    assert search.trials(4) == RandomSearch(search.space, seed=9).trials(4)
    trials = search.trials(2)
    assert trials[0].seed == 9
    assert trials[0].parameters["learning_rate"] > 0
    assert trials[0].parameters["width"] in {8, 16}
