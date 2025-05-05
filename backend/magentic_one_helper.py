import asyncio
import logging
import os
import tempfile

from typing import Optional
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.code_executors.azure import ACADynamicSessionsCodeExecutor
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import SingleThreadedAgentRuntime, CancellationToken
from dotenv import load_dotenv

load_dotenv()

from llm_config import get_llm_client


def generate_session_name():
    import random
    adjectives = ["quantum", "neon", "stellar", "galactic", "cyber", "holographic"]
    nouns = ["cyborg", "android", "drone", "mech", "robot", "alien"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{random.randint(1000, 9999)}"


class MagenticOneHelper:
    def __init__(self, logs_dir: str = None, save_screenshots: bool = False, run_locally: bool = False) -> None:
        self.logs_dir = logs_dir or os.getcwd()
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
        self.session_id = session_id or generate_session_name()

        provider = os.getenv("LLM_PROVIDER", "ollama")
        self.client = get_llm_client(provider)

        self.agents = await self.setup_agents(agents, self.client, self.logs_dir)

    async def setup_agents(self, agents, client, logs_dir):
        agent_list = []

        for agent in agents:
            if agent["type"] == "MagenticOne" and agent["name"] == "Coder":
                agent_list.append(MagenticOneCoderAgent("Coder", model_client=client))

            elif agent["type"] == "MagenticOne" and agent["name"] == "Executor":
                if self.run_locally:
                    executor = CodeExecutorAgent("Executor", code_executor=await DockerCommandLineCodeExecutor(work_dir=logs_dir).start())
                else:
                    pool_endpoint = os.getenv("POOL_MANAGEMENT_ENDPOINT")
                    assert pool_endpoint, "POOL_MANAGEMENT_ENDPOINT env var is not set"
                    with tempfile.TemporaryDirectory() as temp_dir:
                        code_executor = ACADynamicSessionsCodeExecutor(
                            pool_management_endpoint=pool_endpoint,
                            credential=None,
                            work_dir=temp_dir
                        )
                        executor = CodeExecutorAgent("Executor", code_executor=code_executor)
                agent_list.append(executor)

            elif agent["type"] == "MagenticOne" and agent["name"] == "FileSurfer":
                file_surfer = FileSurfer("FileSurfer", model_client=client)
                file_surfer._browser.set_path(os.path.join(os.getcwd(), "data"))
                agent_list.append(file_surfer)

            elif agent["type"] == "Custom":
                from magentic_one_custom_agent import MagenticOneCustomAgent
                agent_list.append(MagenticOneCustomAgent(
                    agent["name"], model_client=client,
                    system_message=agent["system_message"],
                    description=agent["description"]
                ))

            elif agent["type"] == "RAG":
                from magentic_one_custom_rag_agent import MagenticOneRAGAgent
                agent_list.append(MagenticOneRAGAgent(
                    agent["name"],
                    model_client=client,
                    index_name=agent["index_name"],
                    description=agent["description"],
                    AZURE_SEARCH_SERVICE_ENDPOINT=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
                ))

            else:
                print(f"⚠️ Skipping unsupported or function-calling-dependent agent: {agent}")
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