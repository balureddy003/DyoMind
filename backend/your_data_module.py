import os
import pandas as pd
from typing import List, Dict, Any

DATA_DIR = os.getenv("DATA_DIR", "/data")

def data_provider(tablename: str) -> List[Dict[str, Any]]:
    """
    Reads `<tablename>.csv` (or .xlsx) from DATA_DIR and returns a list of row dicts.
    """
    csv_path = os.path.join(DATA_DIR, f"{tablename}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        xlsx_path = os.path.join(DATA_DIR, f"{tablename}.xlsx")
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
        else:
            raise FileNotFoundError(f"No data file found for table '{tablename}'")

    # Convert timestamps to ISO strings if necessary
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Turn each row into a dict
    return df.to_dict(orient="records")