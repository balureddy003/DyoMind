# pip install fastmcp
from mcp.server.fastmcp import FastMCP
import os
import json
import logging

# Initialize the server
mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    return a * b

# @mcp.tool()
# def data_provider() -> str:
#     """A tool that provides some data."""
#     # This is a placeholder for the actual data provider logic
#     data = "This is some data."
#     # read from a file
#     with open("data/pred_maint/sensor.csv", "r") as file:
#         data = file.read()
    
#     return data


@mcp.tool()
def data_provider(tablename: str) -> str:
    """A tool that provides data from database based on given table name as parameter.
    
    Args:
        tablename (str): The table to read data from.
        
    Returns:
        str: The content of the file.

    """
    logger = logging.getLogger("file_provider")
    # This is a placeholder for the actual data provider logic
    data = "This is some data."
    logger.warning(f"Table '{tablename}' requested.")

    try:
        tablename = tablename.strip() + ".csv"
        # locate the file in .data folder recursively
        _file_json = find_file(tablename)
        _file_info = json.loads(_file_json)
        _file_path = _file_info["path"]
        if not _file_path:
            logger.error(f"File '{tablename}' not found.")
            return f"File '{tablename}' not found."
        logger.warning(f"File '{tablename}' found at '{_file_path}'.")
        # read from a file
        with open(_file_path, "r") as file:
            data = file.read()
    
        return data
    except Exception as e:
        logger.error(f"Error reading file '{tablename}': {e}")
        return None

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
    logging.warning(f"File '{filename}' not found in './data' directory.")
    return json.dumps({
        "path": None,
        "filename": filename
    })

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # data = file_provider("maintenance.csv")
    # print(data)