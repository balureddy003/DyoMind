import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from huggingface_client import HuggingFaceChatCompletionClient

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

    elif provider == "huggingface":
        return OpenAIChatCompletionClient(
            model="huggingface/HuggingFaceH4/zephyr-7b-beta",
            api_key=os.getenv("OPENAI_API_KEY", "sk-no-key-needed"),
            base_url="http://localhost:4000",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "openai"
            }
        )

    elif provider == "ollama":
        return OllamaChatCompletionClient(
            config={
                "model": os.getenv("OPENAI_MODEL_NAME", "llama3"),
                "base_url": os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
            }
        )

    elif provider == "lite-ollama":
        return OpenAIChatCompletionClient(
            model=os.getenv("OPENAI_MODEL_NAME", "llama3"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:4000"),
            model_info={
                "function_calling": False,
                "json_output": True,
                "structured_output": True,
                "vision": False,
                "family": "openai"
            }
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")