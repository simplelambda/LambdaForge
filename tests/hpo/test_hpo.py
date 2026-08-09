"""Focused deterministic random-search tests."""

from lambdaforge.hpo import RandomSearch


def test_random_search_is_reproducible_unique_and_materialized() -> None:
    search = RandomSearch(
        {
            "optimizer.params.lr": {"type": "loguniform", "low": 1e-4, "high": 1e-2},
            "model.params.width": {"type": "choice", "values": [8, 16]},
        },
        seed=9,
    )
    assert search.trials(4) == RandomSearch(search.space, seed=9).trials(4)
    materialized = search.materialize({"optimizer": {"params": {}}, "model": {"params": {}}}, 2)
    assert materialized[0]["extensions"]["hpo_trial"]["seed"] == 9
    assert materialized[0]["optimizer"]["params"]["lr"] > 0
