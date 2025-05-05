from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_ext.tools.mcp import SseMcpToolAdapter, StdioServerParams, StdioMcpToolAdapter
from llm_config import get_llm_client
import os

class MagenticOneCustomMCPAgent(AssistantAgent):
    """An agent used by MagenticOne that provides coding assistance using an LLM model client.

    The prompts and description are sealed to replicate the original MagenticOne configuration.
    See AssistantAgent if you wish to modify these values.
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        system_message: str,
        description: str,
        adapter,
    ):
        super().__init__(
            name,
            model_client,
            description=description,
            system_message=system_message,
            tools=adapter
        )

    @classmethod
    async def create(
        cls,
        name: str,
        model_client: ChatCompletionClient = None,
        system_message: str = "",
        description: str = "",
    ):
        if model_client is None:
            model_client = get_llm_client(os.getenv("LLM_PROVIDER", "ollama"))

        server_params = StdioServerParams(
            command="python",
            args=["mcp_math_server.py"],
        )
        adapter_addition = await StdioMcpToolAdapter.from_server_params(server_params, "add")
        adapter_multiplication = await StdioMcpToolAdapter.from_server_params(server_params, "multiply")
        adapter_data_provider = await SseMcpToolAdapter.from_server_params(server_params, "data_provider")

        return cls(
            name,
            model_client,
            system_message,
            description,
            [adapter_addition, adapter_multiplication, adapter_data_provider]
        )
