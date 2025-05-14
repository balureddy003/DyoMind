import asyncio
import httpx
import json

OLLAMA_URL = "http://localhost:4000/v1/chat/completions"
MODEL      = "ollama-llama3"

FUNCTIONS = [
    {
        "name": "search_documents",
        "description": "Search indexed documents by keyword.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
]

async def inspect_ollama():
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Call search_documents with query 'climate'."}
        ],
        "functions": FUNCTIONS,
        "function_call": "auto"
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(OLLAMA_URL, json=payload, timeout=30)
    data = r.json()
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    asyncio.run(inspect_ollama())