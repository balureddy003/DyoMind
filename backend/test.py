import asyncio
import json
import logging
from autogen_agentchat.messages import TextMessage
from autogen import ConversableAgent

# wrap sync tools if needed
def ensure_async(fn):
    if asyncio.iscoroutinefunction(fn):
        return fn
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

# --- your business logic tool ---
async def data_provider(path: str):
    # pretend to read CSV & return a list of dicts
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

class MagneticAgent(ConversableAgent):
    produced_message_types = [TextMessage]
    consumed_message_types = [TextMessage]

    def __init__(self, name="MagneticAgent"):
        super().__init__(
            name=name,
            llm_config={
                "config_list": [{
                   "model": "gpt-4o-mini",
                   "base_url": "http://localhost:4000",
                   "api_key": os.getenv("OPENAI_API_KEY",""),
                   "api_type": "openai"
                }]
            },
            function_map={
                "data_provider": ensure_async(data_provider)
            }
        )

    async def on_message(self, message, sender, config=None):
        logging.debug("on_message: %r", message)
        # build chat history entry
        user_msg = {"role":"user","content":message.content}
        # await generate_reply directly
        is_final, reply = await self.generate_reply(
            messages=[user_msg], sender=sender, config=config
        )
        # handle a function call
        if reply.function_call:
            name = reply.function_call.name
            args = json.loads(reply.function_call.arguments)
            result = await self.function_map[name](**args)
            func_msg = {"role":"function", "name":name, "content":json.dumps(result)}
            is_final, reply = await self.generate_reply(
                messages=[user_msg, reply, func_msg], sender=sender, config=config
            )
        # unwrap into TextMessage
        text = reply.content if hasattr(reply, "content") else str(reply)
        return TextMessage(source=self.name, content=text, metadata={})

    async def on_messages_stream(self, messages, cancellation_token=None):
        # only last message matters
        is_final, reply = await self.generate_reply(
            messages=[{"role":"user","content":messages[-1].content}],
            sender=None
        )
        yield TextMessage(source=self.name, content=reply.content, metadata={})

# --- run a quick test ---
import asyncio

if __name__ == "__main__":
    async def main():
        ag = MagneticAgent("TestMagnetic")
        # simulate a user asking to read sensor.csv
        user = TextMessage(source="user", content="Read sensor.csv please")
        bot_reply = await ag.on_message(user, sender="user")
        print(bot_reply)  # Add this line to print the bot's reply

    asyncio.run(main())