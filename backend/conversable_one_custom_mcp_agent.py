# backend/conversable_one_custom_mcp_agent.py

from autogen import ConversableAgent

MCP_SYSTEM_MESSAGE = """
You are an autonomous planner. You help coordinate multi-agent workflows using Model Context Protocol (MCP).
"""

class ConversableOneCustomMCPAgent(ConversableAgent):
    def __init__(self, name: str, llm_config: dict, system_message: str = MCP_SYSTEM_MESSAGE, description: str = "MCP Orchestrator"):
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message=system_message,
            description=description
        )