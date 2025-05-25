"""
A fault-tolerant MCP-enabled assistant.
"""

from __future__ import annotations

import sys
import json as _json
import types as _types
import asyncio
import logging
from typing import List
from autogen import AssistantAgent
from llm_config import llm_config
from mcp_math_server import calculate_sum, data_provider
# Use Message for better compatibility with AutoGen versions
# Fallback for fix_busted_json
if 'fix_busted_json' not in sys.modules:
    _fix = _types.SimpleNamespace(loads=_json.loads, dumps=_json.dumps)
    sys.modules['fix_busted_json'] = _fix

# Async wrapper
def ensure_async(func):
    if asyncio.iscoroutinefunction(func):
        return func
    async def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def sanitize_args(args: dict) -> dict:
    cleaned = {}
    for key, val in args.items():
        if isinstance(val, str):
            lower = val.lower()
            if lower == "null" or val == "":
                cleaned[key] = None
                continue
            try:
                if "." in val:
                    cleaned[key] = float(val)
                else:
                    cleaned[key] = int(val)
                continue
            except ValueError:
                pass
        cleaned[key] = val
    return cleaned

# Example function schema
function_schemas = [
    {
        "name": "calculate_sum",
        "description": "Calculate the sum of two numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First number" },
                "b": { "type": "number", "description": "Second number" }
            },
            "required": ["a", "b"]
        }
    },
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

async def get_function_details(function_name: str) -> dict:
    for fn in function_schemas:
        if fn["name"] == function_name:
            return fn
    return {}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
)

class MagenticOneCustomMCPAgent(AssistantAgent):

    produced_message_types = []
    consumed_message_types = []

    def __init__(self, name: str):
        handlers = [
            ensure_async(calculate_sum),
            ensure_async(data_provider),
        ]
        super().__init__(
            name=name,
            llm_config=llm_config,
          
            functions=handlers
        )

    async def on_message(self, message, sender=None):
        try:
            logging.debug(f"Received message: {message}")

            chat_payload = {
                "model": self.llm_config.config_list[0]["model"],
                "messages": [{"role": "user", "content": message}],
                "functions": self.function_schemas,
                "function_call": "auto"
            }

            import httpx
            url = self.llm_config.config_list[0]["base_url"] + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.llm_config.config_list[0]['api_key']}"
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=chat_payload)
                response.raise_for_status()
                data = response.json()

            choice = data["choices"][0]
            message_content = choice["message"]

            if "function_call" in message_content:
                function_name = message_content["function_call"]["name"]
                arguments = message_content["function_call"].get("arguments", "{}")
                logging.debug(f"Function to call: {function_name} with args: {arguments}")

                function_schema = await get_function_details(function_name)
                if not function_schema:
                    raise ValueError(f"No schema found for: {function_name}")

                args = sanitize_args(json.loads(arguments))
                func = self.functions.get(function_name)
                if not func:
                    raise ValueError(f"No handler found for: {function_name}")

                result = await func(**args)
                return f"✅ `{function_name}` executed. Result: {result}"
            else:
                return message_content.get("content", "")

        except Exception as e:
            logging.error(f"on_message error: {e}", exc_info=True)
            return f"❌ An error occurred: {e}"

    async def on_messages_stream(self, messages, cancellation_token=None):
        """
        Stream responses using the same message class as the incoming messages
        to ensure registration consistency.
        """
        for message in messages:
            # Generate a response for each incoming message
            response = await self.on_message(message.content)
            # Instantiate the same message type as received
            MessageClass = type(message)
            # Yield a new message instance with the response content
            yield MessageClass(content=response)
 
    async def on_reset(self, cancellation_token):
        logging.debug("Resetting agent state...")
        # Perform any cleanup or reinitialization needed
        return {"status": "reset"}
