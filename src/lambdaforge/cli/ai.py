"""Agent discovery, local approvals and explicit optional integration management."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

from lambdaforge.agents.contracts import Permission
from lambdaforge.agents.knowledge import KINDS
from lambdaforge.agents.services import AgentServices


def add_ai_parser(commands: argparse._SubParsersAction) -> None:
    ai = commands.add_parser(
        "ai", help="Discover agent capabilities, serve MCP and manage Markdown knowledge."
    )
    ai.add_argument(
        "--project", type=Path, default=Path.cwd(), help="Pin this project root for the session."
    )
    actions = ai.add_subparsers(dest="ai_command", required=True)
    actions.add_parser("status", help="Show project scope and actual integration availability.")
    actions.add_parser(
        "doctor", help="Check optional SDK compatibility without installing anything."
    )
    dependencies = actions.add_parser(
        "dependencies", help="Inspect/install fixed optional Python extras with explicit consent."
    )
    dependency_actions = dependencies.add_subparsers(dest="dependency_command", required=True)
    for name in ("check", "install"):
        dependency = dependency_actions.add_parser(name)
        dependency.add_argument("--extra", choices=("agents",), default="agents")
        if name == "install":
            dependency.add_argument(
                "--wheelhouse", help="Project-relative wheels; disables network/index use."
            )
            dependency.add_argument("--apply", action="store_true")
    approvals = actions.add_parser(
        "approvals", help="Local human review; never exposed as MCP approval tools."
    )
    approval_actions = approvals.add_subparsers(dest="approval_command", required=True)
    listing = approval_actions.add_parser("list")
    listing.add_argument("--offset", type=int, default=0)
    listing.add_argument("--limit", type=int, default=20)
    for name in ("show", "approve", "reject"):
        review = approval_actions.add_parser(name)
        review.add_argument("request_id")
        if name == "approve":
            review.add_argument("--plan-hash", required=True)
    for action in ("init", "sync"):
        initialize = actions.add_parser(action, help="Preview/install managed agent assets safely.")
        initialize.add_argument("--agents", choices=("none", "portable", "codex"))
        initialize.add_argument("--knowledge", choices=("none", "markdown", "obsidian"))
        initialize.add_argument("--apply", action="store_true")
        initialize.add_argument(
            "--install-deps",
            action="store_true",
            help="Install declared agents extras in this environment, after asset setup.",
        )
        initialize.add_argument(
            "--wheelhouse", help="Project-relative offline wheels for --install-deps."
        )
    skills = actions.add_parser("skills", help="Inspect official installed Skills and examples.")
    skill_actions = skills.add_subparsers(dest="skills_command", required=True)
    skill_actions.add_parser("list")
    show = skill_actions.add_parser("show")
    show.add_argument("name")
    skill_actions.add_parser("examples")
    capabilities = actions.add_parser(
        "capabilities", help="Page real capabilities; unsupported is not enabled."
    )
    capabilities.add_argument("--describe")
    capabilities.add_argument("--search", default="")
    capabilities.add_argument("--offset", type=int, default=0)
    capabilities.add_argument("--limit", type=int, default=20)
    actions.add_parser(
        "api", help="Inspect installed public Work signatures without consumer imports."
    )
    mcp = actions.add_parser(
        "mcp",
        help="Optional official SDK transport; stdio default or explicitly paired local HTTP.",
    )
    mcp_actions = mcp.add_subparsers(dest="mcp_command", required=True)
    token = mcp_actions.add_parser(
        "token", help="Prepare a private local HTTP pairing file; never print its secret."
    )
    token.add_argument("--apply", action="store_true")
    for name in ("serve", "test"):
        command = mcp_actions.add_parser(
            name,
            help="Serve stdio."
            if name == "serve"
            else "Real SDK client handshake/tool/resource test.",
        )
        command.add_argument("--toolsets", default="core,work,knowledge")
        command.add_argument(
            "--permissions",
            default="READ,PLAN",
            help="Operator-only permissions; WRITE_PROJECT enables separately approved note "
            "edits, not scientific operations.",
        )
        if name == "serve":
            command.add_argument("--transport", choices=("stdio", "http"), default="stdio")
            command.add_argument("--port", type=int, default=8765)
            command.add_argument(
                "--origin",
                action="append",
                default=[],
                help="Exact browser origin authorized for local HTTP; no wildcard.",
            )
        command.add_argument(
            "--allow-project-code",
            action="store_true",
            help="Allow native validation to import trusted project Python (not Work execution).",
        )
    knowledge = actions.add_parser(
        "knowledge", help="Explicit portable project notes, not scientific evidence."
    )
    knowledge_actions = knowledge.add_subparsers(dest="knowledge_command", required=True)
    knowledge_actions.add_parser("status")
    obsidian = knowledge_actions.add_parser(
        "obsidian", help="Inspect optional desktop/CLI hints; probing may start the application."
    )
    obsidian.add_argument("--probe-cli", action="store_true")
    obsidian.add_argument("--apply", action="store_true")
    opening = knowledge_actions.add_parser("open", help="Preview opening the owned vault README.")
    opening.add_argument("--apply", action="store_true")
    audit = knowledge_actions.add_parser("audit", help="Page native edit metadata, never bodies.")
    audit.add_argument("--cursor", type=int, default=0, help="Exact next_cursor from prior page.")
    audit.add_argument("--limit", type=int, default=20)
    export = knowledge_actions.add_parser(
        "export", help="Preview/export exact canonical Markdown as a notes-only ZIP."
    )
    export.add_argument("destination", help="New project-relative .zip, outside the vault.")
    export.add_argument("--apply", action="store_true")
    initialize = knowledge_actions.add_parser(
        "init", help="Prepare Markdown vault; --apply makes changes."
    )
    initialize.add_argument("--obsidian", action="store_true")
    initialize.add_argument("--apply", action="store_true")
    search = knowledge_actions.add_parser("search")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--kind", choices=KINDS)
    search.add_argument("--tag")
    search.add_argument("--status")
    search.add_argument("--offset", type=int, default=0)
    search.add_argument("--limit", type=int, default=20)
    read = knowledge_actions.add_parser("read")
    read.add_argument("note_id")
    backlinks = knowledge_actions.add_parser("backlinks")
    backlinks.add_argument("note_id")
    templates = knowledge_actions.add_parser("templates")
    templates.add_argument("kind", nargs="?", choices=KINDS)
    reindex = knowledge_actions.add_parser(
        "reindex", help="Rebuild disposable index only; never change notes."
    )
    reindex.add_argument("--force", action="store_true")
    reindex.add_argument(
        "--rebuild", action="store_true", help="Rebuild derived SQLite, preserving its old copy."
    )
    link = knowledge_actions.add_parser("link", help="Revision-checked link to an existing note.")
    link.add_argument("note_id")
    link.add_argument("target_id")
    link.add_argument("--revision", required=True)
    link.add_argument("--apply", action="store_true")
    create = knowledge_actions.add_parser(
        "create", help="Create an explicit draft/proposed note, never verify science."
    )
    create.add_argument("kind", choices=KINDS)
    create.add_argument("title")
    create.add_argument("--body")
    create.add_argument("--author-type", choices=("human", "ai"), default="human")
    create.add_argument("--apply", action="store_true")
    update = knowledge_actions.add_parser(
        "update", help="Revision-checked body edit; reload on a conflict."
    )
    update.add_argument("note_id")
    update.add_argument("--revision", required=True)
    update.add_argument("--body", required=True)
    update.add_argument("--apply", action="store_true")


def run_ai_command(arguments: argparse.Namespace) -> int:
    service = AgentServices(arguments.project)
    command = arguments.ai_command
    if command == "dependencies":
        from lambdaforge.agents.dependencies import DependencyService

        dependencies = DependencyService(service.scope)
        if arguments.dependency_command == "check":
            payload = dependencies.check(arguments.extra)
        else:
            payload = dependencies.preview(arguments.extra, wheelhouse=arguments.wheelhouse)
            if arguments.apply:
                payload = dependencies.install(
                    plan_hash=payload["plan_hash"],
                    extra=arguments.extra,
                    wheelhouse=arguments.wheelhouse,
                    consent=True,
                )
    elif command == "approvals":
        from lambdaforge.agents.approvals import ApprovalService

        reviewer = ApprovalService(service.scope)
        if arguments.approval_command == "list":
            payload = reviewer.list(offset=arguments.offset, limit=arguments.limit)
        elif arguments.approval_command == "show":
            payload = reviewer.inspect(arguments.request_id, include_arguments=True)
        elif arguments.approval_command == "approve":
            payload = reviewer.approve_interactive(
                arguments.request_id,
                plan_hash=arguments.plan_hash,
                stdin=sys.stdin,
                stderr=sys.stderr,
            )
        else:
            payload = reviewer.reject_local(arguments.request_id)
    elif command in {"init", "sync"}:
        from lambdaforge.agents.integration import IntegrationService

        integration = IntegrationService(arguments.project)
        status = integration.status()
        agents = arguments.agents or (status["agents"] if command == "sync" else "portable")
        knowledge = arguments.knowledge or status["knowledge"]
        if arguments.install_deps and not arguments.apply:
            raise ValueError(
                "--install-deps requires --apply; preview will never install dependencies."
            )
        if arguments.install_deps and agents == "none":
            raise ValueError("--install-deps needs a selected agent integration.")
        payload = integration.preview(agents=agents, knowledge=knowledge)
        if arguments.apply:
            payload = integration.apply(
                plan_hash=payload["plan_hash"], agents=agents, knowledge=knowledge
            )
            if arguments.install_deps:
                from lambdaforge.agents.dependencies import DependencyService

                dependencies = DependencyService(service.scope)
                dependency_plan = dependencies.preview(wheelhouse=arguments.wheelhouse)
                payload["dependencies"] = dependencies.install(
                    plan_hash=dependency_plan["plan_hash"],
                    wheelhouse=arguments.wheelhouse,
                    consent=True,
                )
    elif command == "skills":
        from lambdaforge.agents.integration import example_assets, skill_metadata, skill_names

        if arguments.skills_command == "show":
            payload = skill_metadata(arguments.name)
        elif arguments.skills_command == "examples":
            payload = {"files": example_assets()}
        else:
            payload = {
                "items": [
                    {k: v for k, v in skill_metadata(name).items() if k != "document"}
                    for name in skill_names()
                ]
            }
    elif command == "mcp":
        if arguments.mcp_command == "token":
            from lambdaforge.agents.http import prepare_token

            print(json.dumps(prepare_token(service.scope, apply=arguments.apply), indent=2))
            return 0
        from lambdaforge.agents.mcp_server import create_server, serve

        selected = tuple(value.strip() for value in arguments.toolsets.split(",") if value.strip())
        permissions = cast(
            tuple[Permission, ...],
            tuple(value.strip() for value in arguments.permissions.split(",") if value.strip()),
        )
        if arguments.mcp_command == "serve":
            if arguments.transport == "http":
                if set(permissions) != {"READ", "PLAN"}:
                    raise ValueError(
                        "Paired local HTTP allows READ/PLAN only; "
                        "use stdio for approved note edits."
                    )
                from lambdaforge.agents.http import serve_http

                serve_http(
                    str(arguments.project),
                    port=arguments.port,
                    origins=tuple(arguments.origin),
                    toolsets=selected,
                    allow_project_code=arguments.allow_project_code,
                )
                return 0
            serve(
                arguments.project,
                toolsets=selected,
                allow_project_code=arguments.allow_project_code,
                permissions=permissions,
            )
            return 0

        async def test() -> dict:
            from mcp.client import Client

            server = create_server(
                arguments.project,
                toolsets=selected,
                allow_project_code=arguments.allow_project_code,
                permissions=permissions,
            )
            async with Client(server, read_timeout_seconds=15) as client:
                tools = await client.list_tools()
                resources = await client.list_resources()
                prompts = await client.list_prompts()
                reply = await client.call_tool("project_info")
                return {
                    "protocol_connected": True,
                    "tools": len(tools.tools),
                    "resources": len(resources.resources),
                    "prompts": len(prompts.prompts),
                    "project": reply.structured_content,
                    "transport": "SDK in-memory",
                    "stdio_verified": False,
                }

        payload = asyncio.run(test())
    elif command == "doctor":
        from lambdaforge.agents.mcp_server import dependency_status

        payload = {
            "project": asdict(service.project_info()),
            "mcp": dependency_status(),
            "human_approval": "local interactive CLI/Console; approved knowledge edits only",
            "http": "opt-in paired loopback READ/PLAN; no public-network serving",
            "installation_performed": False,
        }
    elif command == "knowledge":
        action = arguments.knowledge_command
        vault = service.knowledge
        if action in {"obsidian", "open"}:
            from lambdaforge.agents.obsidian import ObsidianService

            desktop = ObsidianService(vault)
            if action == "open":
                payload = desktop.open(apply=arguments.apply)
            elif arguments.probe_cli:
                payload = desktop.probe_cli(consent=arguments.apply)
            else:
                payload = desktop.status()
        elif action == "init":
            payload = (
                vault.initialize(obsidian=arguments.obsidian)
                if arguments.apply
                else {
                    "status": "preview",
                    "directory": vault.directory,
                    "obsidian": arguments.obsidian,
                    "notes_overwritten": False,
                }
            )
        elif action == "export":
            payload = vault.preview_export(arguments.destination)
            if arguments.apply:
                payload = vault.export(arguments.destination, plan_hash=payload["plan_hash"])
        elif action == "audit":
            payload = vault.audit(cursor=arguments.cursor, limit=arguments.limit)
        elif action == "create":
            payload = (
                vault.create(
                    arguments.kind,
                    arguments.title,
                    body=arguments.body,
                    author_type=arguments.author_type,
                )
                if arguments.apply
                else {
                    "status": "preview",
                    "kind": arguments.kind,
                    "title": arguments.title,
                    "author_type": arguments.author_type,
                    "body": arguments.body or vault.template(arguments.kind),
                    "scientifically_verified": False,
                }
            )
        elif action == "update":
            current = vault.read(arguments.note_id)
            if current["revision"] != arguments.revision:
                raise ValueError("Knowledge write conflict; reload the current note.")
            payload = (
                vault.update(arguments.note_id, revision=arguments.revision, body=arguments.body)
                if arguments.apply
                else {
                    "status": "preview",
                    "note_id": arguments.note_id,
                    "revision": arguments.revision,
                    "body": arguments.body,
                }
            )
        elif action == "reindex":
            payload = vault.reindex(force=arguments.force, rebuild=arguments.rebuild)
        elif action == "link":
            current = vault.read(arguments.note_id)
            vault.read(arguments.target_id)
            if current["revision"] != arguments.revision:
                raise ValueError("Knowledge write conflict; reload before linking notes.")
            payload = (
                vault.link(arguments.note_id, arguments.target_id, revision=arguments.revision)
                if arguments.apply
                else {
                    "status": "preview",
                    "note_id": arguments.note_id,
                    "target_id": arguments.target_id,
                    "revision": arguments.revision,
                }
            )
        else:
            methods = {
                "status": lambda: service.knowledge_status(),
                "read": lambda: service.knowledge_read(arguments.note_id),
                "backlinks": lambda: service.knowledge_backlinks(arguments.note_id),
                "templates": lambda: service.knowledge_templates(arguments.kind),
                "search": lambda: service.knowledge_search(
                    arguments.query,
                    arguments.kind,
                    arguments.tag,
                    arguments.status,
                    arguments.offset,
                    arguments.limit,
                ),
            }
            response = methods[action]()
            payload = asdict(response)
            if response.status == "error":
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                return 2
    else:
        if command == "capabilities":
            response = (
                service.capability_describe(arguments.describe)
                if arguments.describe
                else service.capability_search(arguments.search, arguments.offset, arguments.limit)
            )
        elif command == "api":
            response = service.api_reference()
        else:
            response = service.project_info()
        payload = asdict(response)
        if response.status == "error":
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 2
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return (
        2
        if payload.get("status") == "error"
        or payload.get("dependencies", {}).get("status") == "error"
        else 0
    )
