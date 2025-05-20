"""
A *fault-tolerant* MCP-enabled assistant.

Key points
──────────
•  The constructor will happily run with *no tools at all*.
•  `create()` starts each tool individually; failures are logged but DO NOT
   abort the agent – you simply get fewer tools.
•  The caller can still check `agent.tools` (empty list == no MCP tools).
"""

from __future__ import annotations
import asyncio
import logging
import os
import pathlib
from typing import List, Optional

from autogen import ConversableAgent
from mcp_math_server import calculate_sum
from autogen_agentchat.messages import TextMessage

from function_schema import functions

# Configure debug logging for the MCP agent
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)


class MagenticOneCustomMCPAgent(ConversableAgent):
    def __init__(self, name: str):
        self.produced_message_types = [TextMessage]
        super().__init__(
            name=name,
            llm_config={
                "config_list": [
                    {
                        "model": "ollama/llama3.1",
                        "base_url": "http://localhost:4000",
                        "api_key": "sk-no-key-needed",
                        "api_type": "openai"
                    }
                ],
                "functions": functions
                
            },
            function_map={
                "calculate_sum": calculate_sum
            }
        )
    async def on_reset(self, cancellation_token=None):
        # Optional: clean internal state or logs
       
        return {"status": "reset successful"}
    
    async def on_message(self, message, sender, config=None):
        try:
            # Use built-in AutoGen method for handling function calls
            success, reply = await self.a_generate_function_call_reply(
                messages=[message],
                sender=sender,
                config={**(config or {}), "function_call": "auto"}
            )
            if success:
                return reply
            else:
                # Fallback to normal reply
                return await self.generate_reply(messages=[message], sender=sender, config=config)
        except Exception as e:
            return TextMessage(
                content=f"⚠️ Error: {str(e)}",
                source=getattr(sender, "name", getattr(sender, "id", "agent")),
                metadata={},
                models_usage=None
            )
    
    async def on_messages_stream(self, messages, sender, config=None):
        for msg in messages:
            try:
                result = await self.on_message(msg, sender, config)

                if isinstance(result, TextMessage):
                    # Ensure str content and explicit metadata/models_usage
                    yield TextMessage(
                        content=str(result.content),
                        source=getattr(sender, "name", getattr(sender, "id", "agent")),
                        metadata={},
                        models_usage=None
                    )
                elif isinstance(result, dict) and "content" in result:
                    yield TextMessage(
                        content=str(result["content"]),
                        source=getattr(sender, "name", getattr(sender, "id", "agent")),
                        metadata={},
                        models_usage=None
                    )
                elif isinstance(result, str):
                    yield TextMessage(
                        content=str(result),
                        source=getattr(sender, "name", getattr(sender, "id", "agent")),
                        metadata={},
                        models_usage=None
                    )
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, TextMessage):
                            yield TextMessage(
                                content=str(item.content),
                                source=getattr(sender, "name", getattr(sender, "id", "agent")),
                                metadata={},
                                models_usage=None
                            )
                        elif isinstance(item, dict) and "content" in item:
                            yield TextMessage(
                                content=str(item["content"]),
                                source=getattr(sender, "name", getattr(sender, "id", "agent")),
                                metadata={},
                                models_usage=None
                            )
                        elif isinstance(item, str):
                            yield TextMessage(
                                content=str(item),
                                source=getattr(sender, "name", getattr(sender, "id", "agent")),
                                metadata={},
                                models_usage=None
                            )
                        else:
                            yield TextMessage(
                                content="⚠️ Error: Unexpected item in list",
                                source=getattr(sender, "name", getattr(sender, "id", "agent")),
                                metadata={},
                                models_usage=None
                            )
                else:
                    yield TextMessage(
                        content="⚠️ Error: Unexpected return type from on_message",
                        source=getattr(sender, "name", getattr(sender, "id", "agent")),
                        metadata={},
                        models_usage=None
                    )
            except Exception as e:
                yield TextMessage(
                    content=f"⚠️ Error: {str(e)}",
                    source=getattr(sender, "name", getattr(sender, "id", "agent")),
                    metadata={},
                    models_usage=None
                )