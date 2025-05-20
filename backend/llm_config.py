
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_ext.models.ollama import OllamaChatCompletionClient
from huggingface_client import HuggingFaceChatCompletionClient

# Monkey-patch Ollama client's tool_message_to_ollama to handle dict-based tool messages
import json
from autogen_ext.models.ollama._ollama_client import Message
import autogen_ext.models.ollama._ollama_client as _ollama_mod

_orig_tool_message_to_ollama = _ollama_mod.tool_message_to_ollama

def _patched_tool_message_to_ollama(message):
    out = []
    # allow message to be either a dict, or an object with .content that may be a list or single item
    if isinstance(message, dict):
        items = [message]
    else:
        raw = message.content
        items = raw if isinstance(raw, list) else [raw]
    for x in items:
        if isinstance(x, dict):
            content = json.dumps(x)
        else:
            content = x.content
        out.append(Message(content=content, role="tool"))
    return out

_ollama_mod.tool_message_to_ollama = _patched_tool_message_to_ollama

def get_llm_client(provider: str = "azure"):
    provider = provider.lower()

    if provider == "azure":
        client = AzureOpenAIChatCompletionClient(
            config={
                "azure_deployment": os.getenv("AZURE_DEPLOYMENT"),
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE"),
                "api_version": "2023-05-15",
            }
        )

    elif provider == "huggingface":
        client = OpenAIChatCompletionClient(
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
        client = OllamaChatCompletionClient(
            model=os.getenv("OPENAI_MODEL_NAME", "phi4-mini"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:4000"),
            model_info={
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "vision": False,
                "family": "ollama"
            }
        )

    elif provider == "lite-ollama":
        client = OpenAIChatCompletionClient(
            model=os.getenv("OPENAI_MODEL_NAME", "ollama/llama3.1"),
            api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
            base_url=os.getenv("OPENAI_API_BASE", "http://localhost:4000"),
            model_info={
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "vision": False,
                "family": "openai"
            }
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # functions-as-tools setup from Ollama blog
    client.functions = [
        {
            "name": "data_provider",
            "description": "Stream rows from a CSV or Excel file by table name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tablename": {
                        "type": "string",
                        "description": "The name of the table (without extension)."
                    }
                },
                "required": ["tablename"]
            }
        }
    ]
    client.function_call = "auto"

    return client