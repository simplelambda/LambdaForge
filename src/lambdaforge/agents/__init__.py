"""Optional agent integration; importing this package loads no MCP SDK or model."""

from lambdaforge.agents.integration import IntegrationService
from lambdaforge.agents.knowledge import KnowledgeService
from lambdaforge.agents.scope import ProjectScope
from lambdaforge.agents.services import AgentServices

__all__ = ["AgentServices", "IntegrationService", "KnowledgeService", "ProjectScope"]
