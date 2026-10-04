"""Fleet inspection and preview-first operational controls, without scientific mutations."""

from __future__ import annotations

import json
from argparse import Namespace
from typing import Any

from lambdaforge.controlplane.ClusterCatalog import ClusterCatalog
from lambdaforge.controlplane.FleetResourceService import FleetResourceService


def run_fleet_command(arguments: Namespace) -> int:
    catalog = ClusterCatalog.load(arguments.catalog)
    action = arguments.fleet_command
    payload: dict[str, Any]
    if action == "list":
        payload = {"fleets": [catalog.fleet(name).to_dict() for name in catalog.fleet_names()]}
        human = "\n".join(catalog.fleet_names()) or "No fleets configured."
    else:
        fleet = catalog.fleet(arguments.name)
        if action == "show":
            payload = fleet.to_dict()
            human = f"Fleet {fleet.name} · coordinator {fleet.coordinator}\n" + "\n".join(
                f"  {member.cluster} · {member.state.value} · "
                f"GPUs {member.max_gpus or 'uncapped'} · Runs {member.max_runs or 'uncapped'}"
                for member in fleet.members
            )
        elif action == "offers":
            payload = FleetResourceService(catalog).inspect(fleet)
            human = f"Fleet {fleet.name}: observed, not dispatch-attested\n" + "\n".join(
                f"  {member['cluster']} · {member['health']} · {member['scheduler']} · "
                f"{member['gpu_access']} · local admission not attested"
                for member in payload["members"]
            )
        else:
            fleet.member(arguments.cluster)
            destination = arguments.catalog or ClusterCatalog.project_path()
            payload = {
                "fleet": fleet.name,
                "cluster": arguments.cluster,
                "state": arguments.member_state,
                "apply": arguments.apply,
                "catalog": str(destination),
                "active_jobs_affected": False,
            }
            if arguments.apply:
                FleetResourceService(catalog).set_member_state(
                    fleet.name, arguments.cluster, arguments.member_state, path=destination
                )
            human = (
                f"{'Applied' if arguments.apply else 'Preview'}: {fleet.name} / "
                f"{arguments.cluster} → {arguments.member_state}. Active Jobs are not stopped."
            )
    print(json.dumps(payload, indent=2) if arguments.json else human)
    return 0
