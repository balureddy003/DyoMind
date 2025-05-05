from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from llm_config import get_llm_client
import os

class MagenticOneCustomAgent(AssistantAgent):
    """Custom agent without function calling support."""

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient = None,
        system_message: str = "",
        description: str = "",
    ):
        if model_client is None:
            model_client = get_llm_client(os.getenv("LLM_PROVIDER", "lite-ollama"))

        # Ensure function calling is not used
        super().__init__(
            name=name,
            model_client=model_client,
            description=description,
            system_message=system_message,
            tools=[],                    # ⛔ no tools
            reflect_on_tool_use=False    # ⛔ disable function-calling reflection
        )