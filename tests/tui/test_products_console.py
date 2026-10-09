"""Product/import widgets use native services, explicit mutation and visible worker feedback."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("textual")
from textual.widgets import Button, Input, Static  # noqa: E402

from lambdaforge.tui.App import LambdaForgeApp  # noqa: E402
from lambdaforge.tui.screens.ProductScreen import (  # noqa: E402
    ProductWorkspace,
    StudyImportWorkspace,
)
from lambdaforge.tui.screens.Workspace import ExactConfirmation  # noqa: E402
from lambdaforge.tui.services import ConsoleServices  # noqa: E402
from tests.products.test_product_registry import model, snapshot  # noqa: E402
from tests.tui.test_research_console_014 import FakeServices  # noqa: E402


class ProductServices(FakeServices):
    def __init__(self, product: dict[str, Any]) -> None:
        super().__init__()
        self.product = product

    def product_rows(self, *, offset: int = 0) -> list[dict[str, Any]]:
        self.calls.append(("product_rows", offset))
        return (
            [
                {
                    key: self.product[key]
                    for key in (
                        "name",
                        "kind",
                        "contract",
                        "content_id",
                        "scientific_id",
                    )
                }
                | {
                    "artifact_count": len(self.product["artifacts"]),
                    "size_bytes": sum(item["size_bytes"] for item in self.product["artifacts"]),
                }
            ]
            if offset == 0
            else []
        )

    def product_detail(self, selector: str, *, offset: int = 0) -> dict[str, Any]:
        self.calls.append(("product_detail", (selector, offset)))
        return {
            "product": self.product,
            "offset": offset,
            "provenance": [{"original": True}],
            "consumers": [],
        }

    def verify_product(self, selector: str) -> dict[str, Any]:
        self.calls.append(("verify_product", selector))
        return {"verified": True}

    def import_study(self, source: Path, *, apply: bool = False) -> dict[str, Any]:
        self.calls.append(("import_study", (source, apply)))
        return {
            "name": "portable",
            "execution_id": "execution-1",
            "applied": apply,
            "will_execute": False,
            "status": "imported" if apply else "verified",
            "captured_state": "failed",
            "products": [],
            "path": str(source),
        }


def test_product_browser_is_lazy_and_import_and_verify_are_explicit(tmp_path: Path) -> None:
    product, _ = model(tmp_path)
    services = ProductServices(product.to_dict())

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 42)) as pilot:
            await pilot.pause()
            assert not any(call[0].startswith("product") for call in services.calls)
            app.show_screen("products")
            await pilot.pause()
            assert ("product_rows", 0) in services.calls
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ProductWorkspace)
            assert ("product_detail", ("models", 0)) in services.calls
            assert not any(call[0] == "verify_product" for call in services.calls)
            await pilot.click("#product-verify")
            await pilot.pause()
            assert ("verify_product", "models") in services.calls
            await pilot.click("#product-next")
            await pilot.pause()
            assert ("product_detail", ("models", 20)) in services.calls
            await pilot.click("#workspace-back")
            await pilot.pause()
            await pilot.click("#products-page-next")
            await pilot.pause()
            assert ("product_rows", 100) in services.calls
            await pilot.click("#products-import")
            await pilot.pause()
            assert isinstance(app.screen, StudyImportWorkspace)
            assert not any(call[0] == "import_study" for call in services.calls)

    asyncio.run(exercise())


def test_import_preview_confirmation_then_apply_without_launch(tmp_path: Path) -> None:
    product, _ = model(tmp_path)
    services = ProductServices(product.to_dict())

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 42)) as pilot:
            screen = StudyImportWorkspace(services)
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#study-import-path", Input).value = str(tmp_path)
            await pilot.pause()
            assert screen.query_one("#study-import-apply", Button).disabled
            await pilot.click("#study-import-verify")
            await pilot.pause()
            assert ("import_study", (tmp_path, False)) in services.calls
            assert not screen.query_one("#study-import-apply", Button).disabled
            await pilot.click("#study-import-apply")
            await pilot.pause()
            assert isinstance(app.screen, ExactConfirmation)
            assert ("import_study", (tmp_path, True)) not in services.calls
            await pilot.click("#exact-apply")
            await pilot.pause()
            assert ("import_study", (tmp_path, True)) in services.calls
            assert screen.query_one("#study-import-apply", Button).disabled
            assert "registered" in str(screen.query_one("#study-import-status", Static).render())
            assert not any(call[0] == "submit_work" for call in services.calls)

    asyncio.run(exercise())


def test_import_pending_and_errors_are_visible_and_duplicates_blocked(tmp_path: Path) -> None:
    product, _ = model(tmp_path)
    release = threading.Event()
    entered = threading.Event()

    class BlockedServices(ProductServices):
        def import_study(self, source: Path, *, apply: bool = False) -> dict[str, Any]:
            self.calls.append(("import_study", (source, apply)))
            entered.set()
            release.wait(timeout=5)
            raise ValueError("[broken] checksum differs")

    services = BlockedServices(product.to_dict())

    async def exercise() -> None:
        app = LambdaForgeApp(services)
        async with app.run_test(size=(130, 42)) as pilot:
            screen = StudyImportWorkspace(services)
            app.push_screen(screen)
            await pilot.pause()
            screen.query_one("#study-import-path", Input).value = str(tmp_path)
            await pilot.pause()
            await pilot.click("#study-import-verify")
            await pilot.pause()
            assert entered.is_set() and screen.busy
            assert "Verifying" in str(screen.query_one("#study-import-status", Static).render())
            assert screen.query_one("#study-import-verify", Button).disabled
            screen._load(apply=False)
            assert len([call for call in services.calls if call[0] == "import_study"]) == 1
            release.set()
            await pilot.pause()
            assert not screen.busy
            assert "checksum differs" in str(
                screen.query_one("#study-import-status", Static).render()
            )
            assert screen.query_one("#study-import-apply", Button).disabled

    try:
        asyncio.run(exercise())
    finally:
        release.set()


def test_studies_offer_import_and_open_read_only_snapshots(tmp_path: Path) -> None:
    from textual.widgets import DataTable

    from lambdaforge.tui.screens.Workspace import StudyWorkspace

    product, _ = model(tmp_path)
    product = product.to_dict()
    services = ProductServices(product)
    services.snapshot = {
        "work": {
            "items": [
                {
                    "name": "imported-science",
                    "execution_id": "execution-imported",
                    "work_id": "import:execution-imported",
                    "study_selector": "import:execution-imported",
                    "state": "failed",
                    "cluster": "imported · local snapshot",
                    "imported": True,
                    "study_expected": True,
                    "study": {"candidates": [], "counts": {"candidates": 2}},
                }
            ]
        },
        "clusters": [],
    }

    async def exercise() -> None:
        app = LambdaForgeApp(services=services)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.click("#nav-studies")
            await pilot.pause()
            await pilot.click("#studies-import")
            await pilot.pause()
            assert isinstance(app.screen, StudyImportWorkspace)
            await pilot.press("escape")
            await pilot.pause()
            table = app.query_one("#studies #screen-table", DataTable)
            table.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, StudyWorkspace)
            for identifier in ("study-cancel", "study-retry", "study-delete", "study-export"):
                assert not app.screen.query_one(f"#{identifier}", Button).display
            assert "read-only" in str(app.screen.query_one("#study-action-status", Static).render())
            assert not any(
                call[0] in {"submit_work", "retry_job", "cancel_work"} for call in services.calls
            )

    asyncio.run(exercise())


def test_console_catalog_services_are_read_only_and_do_not_hash_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lambdaforge.products import ProductRegistry

    product, source = model(tmp_path)
    registry = ProductRegistry(tmp_path / "products")
    registry.publish(product, files={"model": source}, apply=True)
    monkeypatch.setenv("LAMBDAFORGE_PRODUCT_ROOT", str(registry.root))
    services = ConsoleServices.__new__(ConsoleServices)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Metadata browse must not hash artifact bytes.")

    monkeypatch.setattr(ProductRegistry, "_verify_file", forbidden)
    before = snapshot(tmp_path)
    row = services.product_rows()[0]
    assert row["name"] == "models" and row["artifact_count"] == 1
    assert not {"payload", "scientific_meaning", "artifacts", "producer"} & row.keys()
    assert services.product_detail("models")["product"]["content_id"] == product.content_id
    assert snapshot(tmp_path) == before
