"""End-to-end YAML customization and public extension contract tests."""

from types import SimpleNamespace

from lambdaforge.training.callbacks import LogKeyFilter
from tests.fixtures.UserCallback import UserCallback


class TestCustomization:
    """Verify that external objects and logging choices require no framework edits."""

    def test_log_key_filter_uses_include_and_exclude_patterns(self) -> None:
        selector = LogKeyFilter(include=["val_*", "epoch_time_s"], exclude=["*_loss_*"])
        assert selector.accepts("val_accuracy")
        assert selector.accepts("epoch_time_s")
        assert not selector.accepts("val_loss_aux")
        assert not selector.accepts("train_accuracy")

    def test_project_callback_artifact_is_rank_zero_only(self, tmp_path) -> None:
        marker = tmp_path / "rank-zero.txt"
        callback = UserCallback(str(marker))
        callback.on_fit_end(SimpleNamespace(is_global_zero=False), object())
        assert not marker.exists()
        callback.on_fit_end(SimpleNamespace(is_global_zero=True), object())
        assert marker.read_text(encoding="utf-8") == "callback invoked"
