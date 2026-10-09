"""Execute the complete native lifecycle locally; no async Jobs or HPC access needed."""

from pathlib import Path

from lambdaforge.work import ResultStore, WorkConfig, WorkRunner


def main() -> None:
    root = Path(__file__).resolve().parent
    study = WorkRunner().run(WorkConfig.from_yaml(root / "train.yaml"))
    assert study.status == "succeeded"
    store = ResultStore()
    assert store.product_status(study.execution_id)["status"] == "published"
    visualization = WorkRunner().run(WorkConfig.from_yaml(root / "visualize.yaml"))
    assert visualization.status == "succeeded"
    assert store.product_status(visualization.execution_id)["status"] == "published"
    report = store.report(visualization.execution_id, root / "work-results.html")
    exported = store.export(visualization.execution_id, root / "exports")
    imported = ResultStore(root / "offline/runs")
    imported.import_export(exported["path"], product_root=root / "offline/products", apply=True)
    offline = imported.report(visualization.execution_id, root / "offline/work-results.html")
    print(f"Original report: {report}\nIndependent imported report: {offline}")


if __name__ == "__main__":
    main()
