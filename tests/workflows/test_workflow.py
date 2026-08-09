"""Focused DAG validation, output-reference and branch execution tests."""

from pathlib import Path

import yaml

from lambdaforge import Workflow, WorkflowPlan, WorkflowResult


def test_workflow_plans_and_resolves_task_outputs(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    common = {
        "kind": "task",
        "schema_version": "1.0",
        "output_root": str(tmp_path / "tasks"),
        "task": {"target": "tests.fixtures.UserTask.UserTask", "params": {"message": "first"}},
    }
    first.write_text(yaml.safe_dump({**common, "name": "first"}), encoding="utf-8")
    second.write_text(yaml.safe_dump({**common, "name": "second"}), encoding="utf-8")
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        yaml.safe_dump(
            {
                "kind": "workflow",
                "schema_version": "1.0",
                "name": "pipeline",
                "output_root": str(tmp_path / "workflows"),
                "max_parallel": 2,
                "nodes": {
                    "prepare": {"config": "first.yaml"},
                    "consume": {
                        "config": "second.yaml",
                        "needs": ["prepare"],
                        "bindings": {"task.params.message": "${nodes.prepare.outputs.message}"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    workflow = Workflow.from_yaml(workflow_path)
    validation = workflow.validate()
    assert validation.is_valid
    assert [node["name"] for node in validation.node_reports] == ["prepare", "consume"]
    plan = workflow.inspect()
    assert isinstance(plan, WorkflowPlan)
    assert plan.levels == (("prepare",), ("consume",))
    result = workflow.run()

    assert isinstance(result, WorkflowResult)
    assert result.status == "ok"
    assert result.nodes["consume"]["outputs"]["message"] == "first"
    assert (result.run_dir / "workflow-result.json").is_file()


def test_workflow_validation_reports_invalid_node_without_execution(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        'kind: task\nschema_version: "1.0"\nname: invalid\n'
        "task: {target: tests.fixtures.DoesNotExist.Missing}\n",
        encoding="utf-8",
    )
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        'kind: workflow\nschema_version: "1.0"\nname: invalid-pipeline\n'
        "nodes:\n  invalid: {config: invalid.yaml}\n",
        encoding="utf-8",
    )

    report = Workflow.from_yaml(workflow_path).validate()

    assert not report.is_valid
    assert "node invalid" in report.summary()
    assert not (tmp_path / "runs").exists()
