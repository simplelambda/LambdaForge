"""Native Study extension preview and explicitly reviewed supported delta submission."""

from __future__ import annotations

import json
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.StudyExtensionService import StudyExtensionService


def run_study_extension_command(arguments: Any) -> int:
    """Do not emulate an unsupported apply through retry, rerun or CLI subprocesses."""
    if arguments.study_command == "extend" and arguments.apply and not arguments.plan_hash:
        raise ValueError(
            "Study extension apply requires --plan-hash from a reviewed --dry-run preview."
        )
    catalog = ClusterCatalog.load(arguments.clusters)
    service = StudyExtensionService(project=catalog.project, catalog=catalog)
    if arguments.study_command == "extend":
        if arguments.apply:
            value = service.apply(
                arguments.study,
                arguments.config,
                target=arguments.on,
                plan_hash=arguments.plan_hash,
                code_policy=arguments.code_policy,
            )
            print(
                json.dumps(value, indent=2)
                if arguments.json
                else (
                    f"Extension {value['state']} · {value['study_series_id']} · "
                    f"Job {value.get('job_id') or 'none (no-op)'}"
                )
            )
            return 0
        value = service.preview(
            arguments.study, arguments.config, target=arguments.on,
            code_policy=arguments.code_policy,
        )
        counts = value["comparison"]["counts"]
        budget = value["budget"]
        human = (
            "Study extension preview · no computation or state change\n"
            f"Origin: {value['origin_execution_id']}\n"
            f"Target: {value['target']}\n"
            f"Preserve {counts['reused']} completed Runs · "
            f"new {counts['new']} · full design {counts['total']}\n"
            f"Historical Attempts: {budget['historical_attempts']} · "
            f"elapsed {budget['historical_elapsed_seconds']:.1f}s\n"
            f"Code proof: {value['code_compatibility']['kind']}\n"
            + (value["code_compatibility"].get("warning", "") + "\n")
            +
            f"Plan: {value['plan_hash']}\n"
            "Exploratory post-hoc evidence; no automatic promotion.\n"
            "No work has started. Review the plan and use "
            "--apply --plan-hash HASH to submit the delta."
        )
    elif arguments.study_command == "refresh":
        value = service.refresh(arguments.series)
        print(
            json.dumps(value, indent=2)
            if arguments.json
            else f"Study series {value['study_series_id']} · "
            + "; ".join(f"revision {op['revision']}: {op['state']}" for op in value["operations"])
        )
        return 0
    elif arguments.study_command == "export":
        value = service.export(arguments.series, arguments.destination)
        human = f"Study series exported · {value['path']} · read-only native Execution packages"
    elif arguments.study_command == "analyze":
        value = service.analysis(arguments.series)
        human = (
            f"Study series analysis · {arguments.series}\n"
            "Exploratory post-hoc evidence; native StudyAnalysis, no origin overwritten.\n"
            "Use --json for details or studies report SERIES --output report.html."
        )
    elif arguments.study_command == "report":
        from lambdaforge.analysis.Report import write_html

        output = write_html(service.analysis(arguments.series), arguments.output)
        print(json.dumps({"report": str(output)}) if arguments.json else str(output))
        return 0
    elif arguments.study_command == "history":
        value = service.history(arguments.series)
        human = (
            "\n".join(
                f"{row['sequence']} · {row['action']} · revision {row['revision']} · "
                f"{row['state']} · {row['at_utc']}"
                for row in value["history"]
            )
            or "No Study series operations."
        )
    else:
        value = service.overview(arguments.series)
        human = (
            f"Study series: {value['study_series_id']}\n"
            f"Revisions: {len(value['revisions'])} · state {value['state']}\n"
            f"Completed {value['counts']['completed']}/{value['counts']['required']} · "
            f"pending {value['counts']['pending']}\n"
            "Every Run retains its physical Execution/Attempt provenance."
        )
    print(json.dumps(value, indent=2, allow_nan=False) if arguments.json else human)
    return 0
