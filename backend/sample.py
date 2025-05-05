import asyncio
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import os
from dotenv import load_dotenv
load_dotenv()


async def example_usage():
    azure_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        azure_credential, "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAIChatCompletionClient(
            model="gpt-4o-2024-11-20",
            azure_deployment="gpt-4o",
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_ad_token_provider=token_provider,
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": "gpt-4o"
            }
        )
    m1 = MagenticOne(client=client )
    task = "write an analysis of the logs in the file monitoring.csv."
    result = await Console(m1.run_stream(task=task))
    print(result)


if __name__ == "__main__":
    asyncio.run(example_usage())