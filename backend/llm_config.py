import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

def get_llm_client(provider: str = "azure"):
    provider = provider.lower()

    if provider == "azure":
        return AzureOpenAIChatCompletionClient(
            config={
                "azure_deployment": os.getenv("AZURE_DEPLOYMENT"),
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": "2023-05-15",
            }
        )

    elif provider == "ollama":
        print("Using native Ollama client")
        return OllamaChatCompletionClient(
            config={
                "model": os.getenv("OPENAI_MODEL_NAME", "phi3"),
                "base_url": os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
            }
        )

    elif provider == "lite-ollama":
        print("Using LiteLLM proxy (OpenAI-compatible)")
        return OpenAIChatCompletionClient(
            model=os.getenv("OPENAI_MODEL_NAME", "phi3"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:4000"),
            model_info={
                "vision": False,
                "function_calling": False,  # <- make sure it's disabled
                "json_output": True,
                "structured_output": True,
                "family": "openai"
            }
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
