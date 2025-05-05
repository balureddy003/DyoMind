from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from azure.identity import DefaultAzureCredential
from llm_config import get_llm_client
import os

MAGENTIC_ONE_RAG_DESCRIPTION = "An agent that has access to internal index and can handle RAG tasks."
MAGENTIC_ONE_RAG_SYSTEM_MESSAGE = """
    You are a helpful AI Assistant.
    When given a user query, use available tools to help the user with their request.
    Reply \"TERMINATE\" in the end when everything is done.
"""

class MagenticOneRAGAgent(AssistantAgent):
    """An agent used by MagenticOne that provides coding assistance using an LLM model client.

    The prompts and description are sealed, to replicate the original MagenticOne configuration.
    """

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient = None,
        index_name: str = "",
        AZURE_SEARCH_SERVICE_ENDPOINT: str = "",
        description: str = MAGENTIC_ONE_RAG_DESCRIPTION,
    ):
        if model_client is None:
            model_client = get_llm_client(os.getenv("LLM_PROVIDER", "ollama"))

        super().__init__(
            name,
            model_client,
            description=description,
            system_message=MAGENTIC_ONE_RAG_SYSTEM_MESSAGE,
            tools=[self.do_search],
            reflect_on_tool_use=True,
        )

        self.index_name = index_name
        self.AZURE_SEARCH_SERVICE_ENDPOINT = AZURE_SEARCH_SERVICE_ENDPOINT

    def config_search(self) -> SearchClient:
        credential = DefaultAzureCredential()
        return SearchClient(
            endpoint=self.AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=self.index_name,
            credential=credential
        )

    async def do_search(self, query: str) -> str:
        """Search indexed data using Azure Cognitive Search with vector-based queries."""
        aia_search_client = self.config_search()
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=1,
            fields="text_vector",
            exhaustive=True
        )
        results = aia_search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["parent_id", "chunk_id", "chunk"],
            top=1
        )
        return "\n".join(result["chunk"] for result in results)
