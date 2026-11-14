"""Native Dataset replication selection/confirmation/progress in the Research Console."""

import asyncio
import threading

import pytest
from textual.widgets import Input, Select

from lambdaforge.tui.App import LambdaForgeApp
from lambdaforge.tui.screens.DatasetReplication import DatasetReplication, ReplicationPassword
from lambdaforge.tui.screens.Workspace import (
    DatasetWorkspace,
    ExactConfirmation,
    ExplanationDialog,
)
from tests.tui.test_research_console_014 import FakeServices


def test_one_use_password_is_masked_not_in_confirmation_and_forwarded_to_native_service():
    secret = "fixture-transient-secret"
    calls = []

    class Services(FakeServices):
        def replicate_dataset(self, selector, **kwargs):
            assert kwargs["site_password"] == secret
            calls.append(bool(kwargs.get("apply")))
            return {
                "action": "REPLICATE",
                "source_cluster": "gpu12",
                "target_cluster": "gpu16",
                "transfer_route": "direct-ssh",
                "execution_host": "gpu12",
                "requires_controller_online": False,
                "job_id": "job-password" if kwargs.get("apply") else None,
            }

    async def exercise():
        app = LambdaForgeApp(Services())
        async with app.run_test(size=(120, 46)) as pilot:
            workspace = DatasetWorkspace({"name": "example", "version": "7"}, app.services)
            await app.push_screen(workspace)
            await pilot.pause()
            workspace._preview_replication(("gpu12", "gpu16", "password"))
            await pilot.pause()
            assert isinstance(app.screen, ReplicationPassword)
            field = app.screen.query_one("#replication-password", Input)
            assert field.password
            field.value = secret
            await pilot.click("#password-accept")
            await pilot.pause(0.2)
            assert isinstance(app.screen, ExactConfirmation)
            assert secret not in str(app.screen.query_one("#exact-preview").render())
            assert calls == [False]
            await pilot.click("#exact-apply")
            await pilot.pause(0.2)
            assert calls == [False, True]
            assert workspace._replication_password is None

    asyncio.run(exercise())


def test_console_preview_confirmation_background_feedback_and_no_duplicate_apply():
    release = threading.Event()
    dataset = {
        "name": "example",
        "version": "7",
        "dataset_id": "sha256:" + "a" * 64,
        "placements": [{"cluster": "local", "root": "/owned/example"}],
    }

    class Services(FakeServices):
        def replicate_dataset(self, selector, **kwargs):
            self.calls.append(
                ("replicate", (selector, kwargs["apply"] if "apply" in kwargs else False))
            )
            assert kwargs["expected_content_id"] == dataset["dataset_id"]
            plan = {
                "dataset": selector,
                "target_cluster": kwargs["destination"],
                "source_cluster": kwargs["source"],
                "action": "REPLICATE",
                "transfer_route": "controller-stream",
                "compression": "tar-gzip-3",
                "relay_consent_required": True,
                "data_path": ["local", "controller", "gpu12"],
                "route_diagnostics": [
                    {
                        "direction": "direct-ssh",
                        "category": "host-key",
                        "message": "Host key verification failed",
                    }
                ],
            }
            if kwargs.get("apply"):
                kwargs["progress"](
                    {"message": "Streaming compressed bytes", "compressed_bytes": 500}
                )
                release.wait(timeout=4)
                plan["placements"] = [*dataset["placements"], {"cluster": "gpu12", "root": "/copy"}]
            return plan

    services = Services()

    async def exercise():
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 42)) as pilot:
            workspace = DatasetWorkspace(dataset, services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            await pilot.click("#dataset-replicate")
            assert isinstance(app.screen, DatasetReplication)
            app.screen.query_one("#replicate-destination", Select).value = "gpu12"
            await pilot.click("#replicate-preview")
            await pilot.pause(0.2)
            assert isinstance(app.screen, ExactConfirmation)
            assert ("replicate", ("example@7", True)) not in services.calls
            assert "tar-gzip" in str(app.screen.query_one("#exact-preview").render())
            preview_text = str(app.screen.query_one("#exact-preview").render())
            assert "relay" in preview_text and "Host key" in preview_text
            await pilot.click("#exact-apply")
            await pilot.pause(0.1)
            assert workspace.query_one("#dataset-replicate").disabled
            assert workspace.query_one("#dataset-delete").disabled
            assert "compressed" in str(workspace.query_one("#dataset-operation-status").render())
            workspace._choose_replication()
            assert app.screen is workspace
            release.set()
            await pilot.pause(0.2)
            assert not workspace.query_one("#dataset-replicate").disabled
            assert "Verified replica registered" in str(
                workspace.query_one("#dataset-operation-status").render()
            )
            assert "gpu12" in str(workspace.query_one("#dataset-locations Static").render())
            assert services.calls.count(("replicate", ("example@7", True))) == 1

    asyncio.run(exercise())


def test_preview_failure_is_visible_and_unlocks_controls():
    class Services(FakeServices):
        def replicate_dataset(self, *_args, **_kwargs):
            raise RuntimeError("destination content conflicts")

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            workspace = DatasetWorkspace(
                {"name": "example", "version": "7", "placements": []}, services
            )
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._preview_replication(("local", "gpu12", "auto"))
            await pilot.pause(0.2)
            assert "destination content conflicts" in str(
                workspace.query_one("#dataset-operation-status").render()
            )
            assert isinstance(app.screen, ExplanationDialog)
            assert not workspace.query_one("#dataset-replicate").disabled
            assert not workspace.query_one("#dataset-delete").disabled

    asyncio.run(exercise())


def test_leaving_workspace_does_not_abort_native_transfer():
    release, completed = threading.Event(), threading.Event()
    errors = []

    class Services(FakeServices):
        def replicate_dataset(self, selector, **kwargs):
            release.wait(timeout=4)
            try:
                kwargs["progress"]({"message": "Checking destination"})
            except Exception as error:
                errors.append(error)
                raise
            completed.set()
            return {"target_cluster": "gpu12", "applied": True}

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test() as pilot:
            workspace = DatasetWorkspace({"name": "example", "version": "7"}, services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._apply_replication({"source": "local", "destination": "gpu12"}, True)
            app.pop_screen()
            await pilot.pause(0.1)
            release.set()
            await pilot.pause(0.2)
            assert completed.is_set() and not errors

    asyncio.run(exercise())


def test_durable_job_submission_is_not_presented_as_verified_completion():
    from textual.widgets import DataTable, TabbedContent

    class Services(FakeServices):
        def replicate_dataset(self, selector, **kwargs):
            return {"dataset": selector, "job_id": "job-transfer", "target_cluster": "gpu16"}

        def dataset_replications(self, _selector):
            return [
                {
                    "job_id": "job-transfer",
                    "source": "gpu12",
                    "destination": "gpu16",
                    "state": "running",
                    "host": "local",
                    "route": "controller-stream",
                    "progress": {"phase": "transfer"},
                }
            ]

        def dataset_replication_logs(self, job_id):
            assert job_id == "job-transfer"
            return "Copying exact compressed bytes"

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(120, 42)) as pilot:
            workspace = DatasetWorkspace({"name": "example", "version": "7"}, services)
            app.push_screen(workspace)
            await pilot.pause(0.2)
            workspace._apply_replication({"source": "gpu12", "destination": "gpu16"}, True)
            await pilot.pause(0.3)
            assert workspace.query_one("#dataset-tabs", TabbedContent).active == "dataset-transfers"
            status = str(workspace.query_one("#dataset-operation-status").render())
            assert "submitted" in status and "Verified replica" not in status
            assert workspace.query_one("#dataset-transfer-table", DataTable).row_count == 1
            values = workspace.query_one("#dataset-transfer-table", DataTable).get_row_at(0)
            assert "via this PC" in values and "local" in values
            await pilot.click("#dataset-transfer-logs")
            await pilot.pause(0.2)
            assert "Copying exact compressed bytes" in str(
                workspace.query_one("#dataset-transfer-details").render()
            )

    asyncio.run(exercise())


def test_endpoint_dialog_controls_fit_standard_terminal():
    async def exercise():
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(
                DatasetReplication("example@7", ("gpu12",), ("local", "gpu12", "gpu16"))
            )
            await pilot.pause(0.1)
            app.screen.query_one("#replicate-destination", Select).value = "gpu16"
            assert await pilot.click("#replicate-preview")

    asyncio.run(exercise())


@pytest.mark.parametrize("route", ["direct", "auto", "relay"])
def test_remote_endpoint_dialog_exposes_route_and_defaults_to_no_laptop(route):
    async def exercise():
        selected = []
        app = LambdaForgeApp(FakeServices())
        async with app.run_test(size=(80, 24)) as pilot:
            dialog = DatasetReplication("example@7", ("gpu16",), ("gpu12", "gpu16"))
            app.push_screen(dialog, selected.append)
            await pilot.pause(0.1)
            assert dialog.query_one("#replicate-route", Select).value == "direct"
            dialog.query_one("#replicate-destination", Select).value = "gpu12"
            dialog.query_one("#replicate-route", Select).value = route
            await pilot.pause(0.1)
            assert await pilot.click("#replicate-preview")
            assert selected == [("gpu16", "gpu12", route)]

    asyncio.run(exercise())


def test_direct_only_preview_failure_is_visible_and_never_enqueues_relay():
    class Services(FakeServices):
        def replicate_dataset(self, _selector, **kwargs):
            assert kwargs["route"] == "direct" and not kwargs.get("apply")
            raise ValueError("push: host-key [untrusted]; pull: authentication denied")

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(80, 24)) as pilot:
            workspace = DatasetWorkspace({"name": "example", "version": "7"}, services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._preview_replication(("gpu16", "gpu12", "direct"))
            await pilot.pause(0.2)
            assert isinstance(app.screen, ExplanationDialog)
            assert "host-key [untrusted]" in str(app.screen.help_text)
            assert "authentication denied" in str(app.screen.help_text)
            assert not workspace._replication_busy
            await pilot.click("#explanation-close")
            assert app.screen is workspace

    asyncio.run(exercise())


def test_direct_policy_cannot_confirm_a_relay_plan():
    async def exercise():
        app = LambdaForgeApp(FakeServices())
        async with app.run_test() as pilot:
            workspace = DatasetWorkspace({"name": "example", "version": "7"}, app.services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._replication_app = app
            workspace._confirm_replication(
                {"source": "gpu16", "destination": "gpu12", "route": "direct"},
                {"action": "REPLICATE", "transfer_route": "controller-stream"},
            )
            await pilot.pause(0.1)
            assert isinstance(app.screen, ExplanationDialog)
            assert "cannot approve a controller relay" in str(app.screen.help_text)

    asyncio.run(exercise())


@pytest.mark.parametrize("confirm", [False, True])
def test_host_fingerprint_modal_requires_approval_and_repreviews_before_copy(confirm):
    from lambdaforge.data.errors import DirectTransferUnavailableError

    class Services(FakeServices):
        trusted = False

        def replicate_dataset(self, _selector, **kwargs):
            self.calls.append(("copy-preview", kwargs))
            assert not kwargs.get("apply")
            if not self.trusted:
                raise DirectTransferUnavailableError(
                    "Unknown host key",
                    (
                        {
                            "category": "host-key",
                            "execution_host": "gpu16",
                            "peer": "gpu12",
                        },
                    ),
                )
            return {
                "action": "REPLICATE",
                "transfer_route": "direct-ssh",
                "target_cluster": "gpu12",
                "execution_host": "gpu16",
            }

        def replication_ssh_trust(self, initiator, peer, **kwargs):
            assert (initiator, peer) == ("gpu16", "gpu12")
            self.calls.append(("trust", kwargs))
            if kwargs.get("apply"):
                assert kwargs["fingerprints"] == ["SHA256:reviewed"]
                assert kwargs["expected_file"] == "/remote/.ssh/known_hosts"
                self.trusted = True
            return {
                "state": "unknown",
                "can_add": True,
                "initiator": initiator,
                "peer": peer,
                "host": "peer.invalid",
                "port": 22,
                "known_hosts_file": "/remote/.ssh/known_hosts",
                "keys": [{"type": "ssh-ed25519", "fingerprint": "SHA256:reviewed"}],
                "notice": "Verify independently; not authenticated by keyscan.",
            }

    async def exercise():
        services = Services()
        app = LambdaForgeApp(services)
        async with app.run_test(size=(100, 36)) as pilot:
            workspace = DatasetWorkspace(
                {"name": "example", "version": "7", "dataset_id": "sha256:exact"}, services
            )
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._preview_replication(("gpu16", "gpu12", "direct"))
            await pilot.pause(0.2)
            assert isinstance(app.screen, ExactConfirmation)
            text = str(app.screen.query_one("#exact-preview").render())
            assert "SHA256:reviewed" in text and "Verify independently" in text
            assert not services.trusted
            await pilot.click("#exact-apply" if confirm else "#exact-cancel")
            await pilot.pause(0.3)
            assert services.trusted is confirm
            if confirm:
                assert isinstance(app.screen, ExactConfirmation)
                assert "Replicate" in app.screen.title_text
                assert sum(name == "copy-preview" for name, _ in services.calls) == 2
            else:
                assert app.screen is workspace
                assert "cancelled" in str(workspace.query_one("#dataset-operation-status").render())
            assert not workspace._replication_busy

    asyncio.run(exercise())


def test_changed_existing_host_key_never_offers_trust_confirmation():
    async def exercise():
        app = LambdaForgeApp(FakeServices())
        async with app.run_test() as pilot:
            workspace = DatasetWorkspace({"name": "example", "version": "7"}, app.services)
            app.push_screen(workspace)
            await pilot.pause(0.1)
            workspace._replication_app = app
            workspace._confirm_replication_trust({}, {"state": "changed", "can_add": False})
            await pilot.pause(0.1)
            assert isinstance(app.screen, ExplanationDialog)
            assert "will not replace" in str(app.screen.help_text)

    asyncio.run(exercise())
