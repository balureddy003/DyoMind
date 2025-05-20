import asyncio
import logging
import os
import tempfile

from typing import Optional
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator
from guard_utils import _guard
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.azure import ACADynamicSessionsCodeExecutor
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import SingleThreadedAgentRuntime, CancellationToken
from dotenv import load_dotenv
from magentic_one_custom_mcp_agent import MagenticOneCustomMCPAgent
from magentic_one_custom_agent import MagenticOneCustomAgent
from magentic_one_custom_rag_agent import MagenticOneRAGAgent
load_dotenv()
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from llm_config import get_llm_client


def generate_session_name():
    import random
    adjectives = ["quantum", "neon", "stellar", "galactic", "cyber", "holographic"]
    nouns = ["cyborg", "android", "drone", "mech", "robot", "alien"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{random.randint(1000, 9999)}"


class MagenticOneHelper:
    def __init__(self, logs_dir: str = None, save_screenshots: bool = False, run_locally: bool = False, llm_client=None) -> None:
        self.logs_dir = logs_dir or os.getcwd()
        # allow injecting a pre-configured LLM client (functions-aware)
        self.client = llm_client
        self.runtime: Optional[SingleThreadedAgentRuntime] = None
        self.save_screenshots = save_screenshots
        self.run_locally = run_locally
        self.max_rounds = 50
        self.max_time = 25 * 60
        self.max_stalls_before_replan = 5
        self.return_final_answer = True
        self.start_page = "https://www.bing.com"

        os.makedirs(self.logs_dir, exist_ok=True)

    async def initialize(self, agents, session_id=None) -> None:
        self.runtime = SingleThreadedAgentRuntime()
        # Patch the orchestrator to use our guarded progress step
        MagenticOneOrchestrator._orchestrate_step = _guard
        MagenticOneOrchestrator._progress_guard = True
        logging.getLogger("main").info("✅  Magentic-One safety-net activated")
        self.session_id = session_id or generate_session_name()

        # if no client was injected, build one now
        if self.client is None:
            provider = os.getenv("LLM_PROVIDER", "ollama")
            self.client = get_llm_client(provider)

        self.agents = self.setup_agents(agents, self.client, self.logs_dir)

    def setup_agents(self, agents, client, logs_dir):
        agent_list = []
        for agent in agents:
            # This is default MagenticOne agent - Coder
            if (agent["type"] == "MagenticOne" and agent["name"] == "Coder"):
                coder = MagenticOneCoderAgent("Coder", model_client=client)
                agent_list.append(coder)
                print("Coder added!")

            # This is default MagenticOne agent - Executor
            elif (agent["type"] == "MagenticOne" and agent["name"] == "Executor"):
                # handle local = local docker execution
                if self.run_locally:
                    #docker
                    code_executor = DockerCommandLineCodeExecutor(work_dir=logs_dir, init_command="pip install pandas")
                    code_executor.start()

                    executor = CodeExecutorAgent("Executor", code_executor=code_executor)
                
                # or remote = Azure ACA Dynamic Sessions execution
                else:
                    pool_endpoint = os.getenv("POOL_MANAGEMENT_ENDPOINT")
                    assert pool_endpoint, "POOL_MANAGEMENT_ENDPOINT environment variable is not set"
                    with tempfile.TemporaryDirectory() as temp_dir:# Define the correct path to the data folder for file access
                        code_executor=ACADynamicSessionsCodeExecutor(
                            pool_management_endpoint=pool_endpoint,
                            credential=azure_credential,
                            work_dir=temp_dir
                        )
                        print(code_executor._session_id)
                        #code_executor.upload_files(os.path.join(os.getcwd(), "data"))
                        print("Files uploaded!")
                        executor = CodeExecutorAgent("Executor",code_executor=code_executor )
                
                agent_list.append(executor)
                print("Executor added!")

            # This is default MagenticOne agent - WebSurfer
            elif (agent["type"] == "MagenticOne" and agent["name"] == "WebSurfer"):
                web_surfer = MultimodalWebSurfer("WebSurfer", model_client=client, start_page="https://azure.microsoft.com/en-us/blog/?sort-by=newest-oldest&category=ai-machine-learning&content-type=announcements&date=any&s=")
                agent_list.append(web_surfer)
                print("WebSurfer added!")
            
            # This is default MagenticOne agent - FileSurfer
            elif (agent["type"] == "MagenticOne" and agent["name"] == "FileSurfer"):
                file_surfer = FileSurfer("FileSurfer", model_client=client)
                file_surfer._browser.set_path(os.path.join(os.getcwd(), "data"))  # Set the path to the data folder in the current working directory
                agent_list.append(file_surfer)
                print("FileSurfer added!")
            
            # This is custom agent - simple SYSTEM message and DESCRIPTION is used inherited from AssistantAgent
            elif (agent["type"] == "Custom"):
                custom_agent = MagenticOneCustomAgent(
                    agent["name"], 
                    model_client=client, 
                    system_message=agent["system_message"], 
                    description=agent["description"]
                    )

                agent_list.append(custom_agent)
                print(f'{agent["name"]} (custom) added!')
            
            elif (agent["type"] == "CustomMCP"):
                try:
                    from mcp_math_server import get_all_function_map
                    function_map = get_all_function_map()
                except Exception as e:
                    print(f"Warning: failed to load function map for {agent['name']}. Error: {e}")
                    function_map = {}

                custom_agent = MagenticOneCustomMCPAgent(name=agent["name"])
                if hasattr(custom_agent, "function_map") and isinstance(function_map, dict):
                    try:
                        custom_agent.__dict__["function_map"] = function_map
                        print(f"{agent['name']} (CustomMCP) initialized with function map: {list(function_map.keys())}")
                    except Exception as set_err:
                        print(f"Failed to assign function map: {set_err}")
                else:
                    print(f"{agent['name']} (CustomMCP) initialized without function map")
                agent_list.append(custom_agent)

            
            # This is custom agent - RAG agent - you need to specify index_name and Azure Cognitive Search service endpoint and admin key in .env file
            elif (agent["type"] == "RAG"):
                # RAG agent
                rag_agent = MagenticOneRAGAgent(
                    agent["name"], 
                    model_client=client, 
                    index_name=agent["index_name"],
                    description=agent["description"],
                    AZURE_SEARCH_SERVICE_ENDPOINT=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
                    # AZURE_SEARCH_ADMIN_KEY=os.getenv("AZURE_SEARCH_ADMIN_KEY")
                    )
                agent_list.append(rag_agent)
                print(f'{agent["name"]} (RAG) added!')
            else:
                raise ValueError('Unknown Agent!')
        return agent_list

    def main(self, task):
        team = MagenticOneGroupChat(
            participants=self.agents,
            model_client=self.client,
            max_turns=self.max_rounds,
            max_stalls=self.max_stalls_before_replan,
            emit_team_events=False
        )
        cancellation_token = CancellationToken()
        stream = team.run_stream(task=task, cancellation_token=cancellation_token)
        return stream, cancellation_token


async def main(agents, task, run_locally) -> None:
    magentic_one = MagenticOneHelper(logs_dir=".", run_locally=run_locally)
    await magentic_one.initialize(agents)
    team = MagenticOneGroupChat(
        participants=magentic_one.agents,
        model_client=magentic_one.client,
        max_turns=magentic_one.max_rounds,
        max_stalls=magentic_one.max_stalls_before_replan,
    )
    try:
        await Console(team.run_stream(task=task))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await team.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MagenticOneHelper with specified task.")
    parser.add_argument("--task", "-t", type=str, required=True, help="The task to run, e.g., 'Explain async IO in Python'")
    parser.add_argument("--run_locally", action="store_true", help="Run code execution locally")
    args = parser.parse_args()

    MAGENTIC_ONE_DEFAULT_AGENTS = [
        {"input_key": "0001", "type": "MagenticOne", "name": "Coder", "system_message": "", "description": "", "icon": "👨‍💻"},
        {"input_key": "0002", "type": "MagenticOne", "name": "Executor", "system_message": "", "description": "", "icon": "💻"},
        {"input_key": "0003", "type": "MagenticOne", "name": "FileSurfer", "system_message": "", "description": "", "icon": "📂"},
        {"input_key": "0004", "type": "Custom", "name": "CustomAssistant", "system_message": "You are helpful.", "description": "Basic assistant agent."}
    ]

    asyncio.run(main(MAGENTIC_ONE_DEFAULT_AGENTS, args.task, args.run_locally))