import os

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b

def data_provider(tablename: str) -> str:
    """Returns CSV contents of the table (as filename in ./data)."""
    filename = os.path.join("data", f"{tablename}.csv")
    if not os.path.isfile(filename):
        return f"File {filename} not found."
    with open(filename, "r") as f:
        return f.read()

TOOL_FUNC_MAP = {
    "add": add,
    "multiply": multiply,
    "data_provider": data_provider,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "data_provider",
            "description": "Fetches CSV file contents for a table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tablename": {"type": "string"}
                },
                "required": ["tablename"]
            }
        }
    }
]