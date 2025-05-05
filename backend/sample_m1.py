import asyncio
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne, MagenticOneGroupChat
from autogen_agentchat.ui import Console
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.code_executors.azure import ACADynamicSessionsCodeExecutor
import os
from anyio import open_file
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
import tempfile
from dotenv import load_dotenv
import csv
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
    #m1 = MagenticOne(client=client )
    pool_endpoint = os.getenv("POOL_MANAGEMENT_ENDPOINT")
    assert pool_endpoint, "POOL_MANAGEMENT_ENDPOINT environment variable is not set"
    cancellation_token = CancellationToken()

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file_1 = "maintenance.csv"
        test_file_1_contents = "test1 contents"


        async with await open_file(os.path.join(temp_dir, test_file_1), "w") as f:  # type: ignore[syntax]
            await f.write(test_file_1)

        assert os.path.isfile(os.path.join(temp_dir, test_file_1))

        code_executor = ACADynamicSessionsCodeExecutor(
            pool_management_endpoint=pool_endpoint, credential=DefaultAzureCredential(), work_dir=temp_dir
        )
        await code_executor.upload_files([test_file_1], cancellation_token)
        print("Files uploaded!")

    '''
    with tempfile.TemporaryDirectory() as temp_dir:# Define the correct path to the data folder for file access
        code_executor=ACADynamicSessionsCodeExecutor(
            pool_management_endpoint=pool_endpoint,
            credential=azure_credential,
            work_dir=temp_dir
        )
        print(code_executor._session_id)
        #code_executor.upload_files((os.path.join(os.getcwd(), "data")), cancellation_token="exit")
        code_executor.upload_files(
            files=["maintenance.csv"],  # using absolute path to the file
            cancellation_token="exit"
        )
        print("Files uploaded!")
        executor = CodeExecutorAgent("Executor",code_executor=code_executor )
    '''
    executor = CodeExecutorAgent("Executor",code_executor=code_executor )

    fs = FileSurfer("FileSurfer", model_client=client)
    ws = MultimodalWebSurfer("WebSurfer", model_client=client)
    coder = MagenticOneCoderAgent("Coder", model_client=client)

    path1 = "maintenance.csv"
    agents = [ coder, executor,]
    m1=MagenticOneGroupChat(model_client=client, participants=agents)
    #task = f"write code to print thefirst row in {path1} , the file is a csv, after you create the code you should execute it"
    task=f"""ask the executor to run this code {{

def read_and_print_csv(file_name):
    try:
        # Check if the file exists
        if not os.path.isfile(file_name):
            print(f"Error: The file '{{file_name}}' does not exist in the current directory.")
            return
        
        print(f"Successfully located the file: {{file_name}}")

        # Open and read the CSV file
        with open(file_name, mode='r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)  # Convert to a list to read all rows

            if not rows:
                print("The file is empty. No data to display.")
                return
            
            print(f"Reading and printing the contents of '{{file_name}}':")
            for row in rows:
                print(row)  # Print each row

    except Exception as e:
        print(f"An error occurred while accessing or reading the file: {{e}}")

# File name
csv_file_name = 'maintenance.csv'

# Execute function
read_and_print_csv(csv_file_name)
}}"""
    
    '''
    task="execute this code import os " \
    "print(cos.cwd())" \
    "print(os.listdir(os.getcwd()))"'
    '''

    result = await Console(m1.run_stream(task=task))
    print(result)


if __name__ == "__main__":
    asyncio.run(example_usage())