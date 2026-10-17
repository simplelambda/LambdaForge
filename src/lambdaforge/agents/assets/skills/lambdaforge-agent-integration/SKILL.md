---
name: lambdaforge-agent-integration
description: "Configure or troubleshoot LambdaForge MCP in Codex or another client, discover session-enabled tools, or diagnose structured protocol/approval errors. Use for client integration, not ordinary scientific Work authoring."
metadata:
  framework: lambdaforge
  contract-version: "1"
---

# Connect an agent to the native project

## Prerequisites and input contract

Pin the absolute directory containing the intended `pyproject.toml`. Obtain the client,
selected toolsets and operator permissions; changing cwd must not change that identity.

## Procedure

1. Read `lf ai doctor` and `lf ai capabilities --limit 20 --json`. Discover actual installed
   operations, their session availability and trust requirements before proposing actions.
2. For metadata access keep READ/PLAN. Project validation imports consumer Python and needs
   separate operator `--allow-project-code`; it is not a sandbox.
3. Preview `lf ai init --agents codex`. Apply only with operator authorization. Review the
   generated example entry rather than replacing global client configuration.
4. Check optional dependencies, then test the real MCP handshake. A generated config is
   not proof of a connected client; a handshake is not proof of an LLM conversation.
5. Treat `isError`/`error.code` as failure. Preserve project and exact native IDs across
   calls; retrieval time never proves provider freshness.

Read [the integration procedure](references/procedure.md) for commands, permission boundaries
and stale approvals. No tool accepts agent-issued approval or shell commands. Never install
dependencies, submit science or start desktop apps as discovery.

## Validation

Delivery: actual client/interpreter/project, enabled tools, successful protocol checks and
unsupported operations. Notes-only permission is not science authority, and paired loopback
HTTP is not public OAuth.

