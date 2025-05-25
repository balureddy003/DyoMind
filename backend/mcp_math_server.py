import logging
import os
import json
import aiofiles
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# --- Universal async wrapper ---
def ensure_async(fn):
    if asyncio.iscoroutinefunction(fn):
        return fn
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

async def calculate_sum(a: float, b: float) -> float:
    if a is None or b is None:
        raise ValueError("❌ Invalid arguments received: a or b is None")
    logger = logging.getLogger("calculate_sum")
    logger.info(f"calculate_sum() called with a={a}, b={b}")
    return float(a) + float(b)

async def data_provider(tablename: str) -> str:
    """
    Retrieve data from CSV file for a given table name from the ./data directory.
    
    Parameters:
        tablename (str): Name of the table (without .csv extension) to fetch.
    
    Returns:
        str: Contents of the CSV file or an error message.
    """
    logger = logging.getLogger("data_provider")
    data = "This is some data."
    logger.warning(f"Table '{tablename}' requested.")

    try:
        tablename = tablename.strip() + ".csv"
        _file_json = find_file(tablename)
        _file_info = json.loads(_file_json)
        _file_path = _file_info["path"]
        if not _file_path:
            logger.error(f"File '{tablename}' not found.")
            return f"File '{tablename}' not found."
        logger.warning(f"File '{tablename}' found at '{_file_path}'.")
        # Async file read
        async with aiofiles.open(_file_path, "r") as file:
             data = await file.read()
        return data or "No data found."
    except Exception as e:
        logger.error(f"Error reading file '{tablename}': {e}")
        return f"Error: {e}"

def find_file(filename: str) -> str:
    """
    Searches recursively within the ./data folder for an exact filename match.
    Returns a JSON string with the full relative path and the original filename.
    """
    logger = logging.getLogger("find_file")
    for root, _, files in os.walk("./data"):
        if filename in files:
            full_path = os.path.join(root, filename)
            logger.warning(f"Found file: {full_path}")
            return json.dumps({
                "path": full_path,
                "filename": filename
            })
    logger.warning(f"File '{filename}' not found in './data' directory.")
    return json.dumps({
        "path": None,
        "filename": filename
    })

# --- Make handle_request_as_agent async and always return a string ---
async def handle_request_as_agent(query: str) -> str:
    logger = logging.getLogger("handle_request")
    try:
        if "data_provider" in query.lower():
            # extract table name (default to "sensor" if not found)
            import re
            m = re.search(r"(\w+)\s+table", query, re.IGNORECASE)
            table = m.group(1) if m else "sensor"
            return await data_provider(table)
        if "add" in query.lower():
            parts = query.lower().replace("add", "").replace("and", ",").split(",")
            a, b = map(lambda x: float(x.strip()), parts)
            result = await calculate_sum(a, b)
            return f"The result is {result}."
        else:
            return "I'm not sure how to handle that request."
    except Exception as e:
        logger.error(f"Failed to handle request '{query}': {e}")
        return f"Error: {e}"

# --- Function map, all wrapped as async ---
def get_all_function_map():
    function_map = {
        "calculate_sum": calculate_sum,
        "data_provider": data_provider
        
        # add more functions here
    }
    return {k: ensure_async(v) for k, v in function_map.items()}