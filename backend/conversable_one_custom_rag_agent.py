from autogen import ConversableAgent
from azure.search.documents import SearchClient
from azure.identity import DefaultAzureCredential
from azure.search.documents.models import VectorizableTextQuery

class ConversableOneCustomRAGAgent(ConversableAgent):
    def __init__(self, name: str, llm_config: dict, index_name: str, endpoint: str, description: str = ""):
        super().__init__(
            name=name,
            llm_config=llm_config,
            system_message="You are a helpful RAG assistant. Use your internal index to answer questions based on documents.",
            description=description,
        )

        self.index_name = index_name
        self.endpoint = endpoint
        self.credential = DefaultAzureCredential()

        self.register_reply(self.trigger, self.reply_func)

    def trigger(self, msg):
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        # Optional: You can filter for specific trigger words or process all queries
        return True if content else False

    def config_search(self) -> SearchClient:
        return SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )

    def do_search(self, query: str) -> str:
        client = self.config_search()
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=2,
            fields="text_vector",
            exhaustive=True
        )
        results = client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["chunk"],
            top=2
        )

        chunks = [doc["chunk"] for doc in results]
        return "\n\n".join(chunks).strip()

    def reply_func(self, agent, messages, sender, config):
        query = messages[-1]["content"]
        result = self.do_search(query)
        return {"content": f"🔍 Here's what I found:\n\n{result}"}