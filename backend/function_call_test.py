import requests
import json

# Define the tool/function schema (OpenAI-style)
functions = [
    {
        "type": "function",  # Required for LM Studio
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# Prepare the request payload
payload = {
    "model": "deepseek-r1-distill-qwen-7b",  # or whatever name you gave in LM Studio
    "messages": [
        {"role": "user", "content": "What's the weather like in Paris?"}
    ],
    "tools": functions,
    "tool_choice": "auto"
}

# Send the request to LM Studio's local server
response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

# Print raw response
print(json.dumps(response.json(), indent=2))