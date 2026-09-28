from __future__ import annotations

from lambdaforge.work.design import StudyConvergenceState


def test_optimize_goal_does_not_override_useful_scientific_action() -> None:
    state = StudyConvergenceState(
        goal="optimize",
        optimization_stable=True,
        contenders_stable=True,
        questions_resolved=False,
        useful_action_available=True,
        confirmation_complete=False,
        evidence_events=12,
    )

    assert state.screening_stable is True
    assert state.converged is False
    assert state.to_dict()["confirmation_complete"] is False


def test_immaterial_unresolved_questions_can_still_allow_convergence() -> None:
    state = StudyConvergenceState(
        goal="understand",
        optimization_stable=True,
        contenders_stable=True,
        questions_resolved=False,
        useful_action_available=False,
        confirmation_complete=True,
        evidence_events=30,
    )

    assert state.converged is True
