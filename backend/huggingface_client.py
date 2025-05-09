from autogen_ext.models.openai import OpenAIChatCompletionClient

class HuggingFaceChatCompletionClient(OpenAIChatCompletionClient):
    def __init__(self):
        # LM Studio-style OpenAI-compatible server
        super().__init__(
            model="mistral-openorca",
            base_url="http://localhost:1234/v1",
            api_key="sk-no-key-needed",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "openai"
            }
        )
