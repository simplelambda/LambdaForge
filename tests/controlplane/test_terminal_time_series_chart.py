"""Focused tests for dependency-light time-series rendering."""

from io import StringIO

from lambdaforge.cli.TerminalTheme import TerminalTheme
from lambdaforge.cli.TerminalTimeSeriesChart import TerminalTimeSeriesChart


def test_chart_has_axes_current_value_and_a_visible_curve() -> None:
    rendered = TerminalTimeSeriesChart.render(
        "GPU utilization",
        (0.0, 10.0, 35.0, 90.0),
        width=64,
        rows=4,
        minimum=0,
        maximum=100,
        suffix="%",
        x_label="← 60s history · now",
    )

    assert rendered[0].startswith("GPU utilization  current=90%")
    assert any(any("\u2801" <= char <= "\u28ff" for char in line) for line in rendered)
    assert rendered[-1].endswith("┘")
    assert max(map(len, rendered)) <= 64


def test_chart_tolerates_missing_samples_and_constant_values() -> None:
    rendered = TerminalTimeSeriesChart.render(
        "epoch_time_s",
        (None, 2.5, None, 2.5),
        width=40,
        rows=3,
        suffix="s",
    )

    assert "current=2.5s" in rendered[0]
    assert len(rendered) == 5


def test_chart_marks_a_selected_observation_without_changing_width() -> None:
    rendered = TerminalTimeSeriesChart.render(
        "val_loss",
        (0.9, 0.6, 0.4, 0.3),
        width=48,
        selected_index=1,
    )

    assert sum(line.count("●") for line in rendered) == 1
    assert max(map(len, rendered)) <= 48


def test_chart_distinguishes_selected_and_best_objective_epochs() -> None:
    rendered = TerminalTimeSeriesChart.render(
        "val_auprc",
        (0.5, 0.8, 0.6, 0.7),
        width=48,
        selected_index=3,
        best_index=1,
    )

    assert sum(line.count("●") for line in rendered) == 1
    assert sum(line.count("◆") for line in rendered) == 1


def test_terminal_theme_adds_semantic_colour_only_when_enabled() -> None:
    screen = "LambdaForge Run · running\nEPOCH METRICS (2)\n▶ 2  0.8\n⠇●\nhelp"

    assert TerminalTheme.apply(screen, enabled=False) == screen
    styled = TerminalTheme.apply(screen, enabled=True)

    assert "\x1b[1;36mLambdaForge" in styled
    assert "\x1b[30;46m▶" in styled
    assert "\x1b[1;31m●" in styled
    assert styled.endswith("\x1b[0m")


def test_terminal_theme_capability_respects_non_tty_stream() -> None:
    assert not TerminalTheme.enabled(StringIO())
