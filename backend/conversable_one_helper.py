import asyncio
import logging
import os
import random
from typing import Optional, List

from autogen import ConversableAgent, GroupChat, GroupChatManager
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO)
log = logging.info

# === SESSION NAME UTILITY ===
def generate_session_name():
    adjectives = ["quantum", "neon", "stellar", "galactic", "cyber", "holographic"]
    nouns = ["cyborg", "android", "drone", "mech", "robot", "alien"]
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{random.randint(1000, 9999)}"

# === CONVERSABLE HELPER ===
class ConversableOneHelper:
    def __init__(self, agents_config: list, llm_config: dict):
        self.agents_config = agents_config
        self.llm_config = llm_config
        self.max_rounds = 10
        self.max_stalls = 2
        self.session_id = generate_session_name()

    def build_agent(self, agent_info):
        from conversable_one_custom_agent import ConversableOneCustomAgent
        from conversable_one_custom_rag_agent import ConversableOneCustomRAGAgent
        from conversable_one_custom_mcp_agent import ConversableOneCustomMCPAgent

        agent_type = agent_info["type"]
        if agent_type == "Custom":
            return ConversableOneCustomAgent(
                name=agent_info["name"],
                llm_config=self.llm_config,
                system_message=agent_info.get("system_message", ""),
                description=agent_info.get("description", "")
            )
        elif agent_type == "RAG":
            return ConversableOneCustomRAGAgent(
                name=agent_info["name"],
                llm_config=self.llm_config,
                index_name=agent_info["index_name"],
                description=agent_info["description"],
                endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
            )
        elif agent_type == "MCP":
            return ConversableOneCustomMCPAgent(
                name=agent_info["name"],
                llm_config=self.llm_config,
                system_message=agent_info.get("system_message", ""),
                description=agent_info.get("description", "")
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    def initialize(self):
        self.agents = [self.build_agent(cfg) for cfg in self.agents_config]
        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            max_round=self.max_rounds
        )
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config
        )

    def main(self, task):
        cancellation_token = CancellationToken()
        stream = self.group_chat.run_stream(task=task, cancellation_token=cancellation_token)
        return stream, cancellation_token

    async def run_console(self, task: str):
        self.initialize()
        try:
            await Console(self.group_chat.run_stream(task=task))
        except Exception as e:
            log(f"[❌ ERROR] {str(e)}")
        finally:
            await self.group_chat.shutdown()