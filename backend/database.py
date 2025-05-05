import os, glob, json
from pymongo import MongoClient
from typing import Optional, Dict, Union
from bson import ObjectId
from dotenv import load_dotenv


def convert_objectid(obj: Union[dict, list]) -> Union[dict, list]:
    """Recursively convert ObjectId fields to strings."""
    if isinstance(obj, list):
        return [convert_objectid(i) for i in obj]
    elif isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, ObjectId):
                new_obj[k] = str(v)
            elif isinstance(v, (dict, list)):
                new_obj[k] = convert_objectid(v)
            else:
                new_obj[k] = v
        return new_obj
    else:
        return obj


class CosmosDB:
    def __init__(self):
        load_dotenv(".env", override=True)
        uri = os.getenv("COSMOS_DB_URI", "mongodb://localhost:27017")
        db_name = os.getenv("COSMOS_DB_DATABASE", "ag_demo")
        self.client = MongoClient(uri)
        self.database = self.client[db_name]
        self.use_local = True
        self.containers = {}

    def get_container(self, name: str):
        if name in self.containers:
            return self.containers[name]
        container = self.database[name]
        self.containers[name] = container
        return container

    def create_team(self, team: dict):
        container = self.get_container("agent_teams")
        result = container.insert_one(team)
        return {"inserted_id": str(result.inserted_id)}

    def get_teams(self):
        container = self.get_container("agent_teams")
        teams = list(container.find({}))
        return convert_objectid(teams)

    def get_team(self, team_id: str):
        container = self.get_container("agent_teams")
        team = container.find_one({"team_id": team_id})
        return convert_objectid(team) if team else None

    def update_team(self, team_id: str, team: dict):
        container = self.get_container("agent_teams")
        result = container.update_one({"team_id": team_id}, {"$set": team})
        if result.matched_count == 0:
            return {"error": "Team not found"}
        updated = container.find_one({"team_id": team_id})
        return convert_objectid(updated)

    def delete_team(self, team_id: str):
        container = self.get_container("agent_teams")
        result = container.delete_one({"team_id": team_id})
        if result.deleted_count == 0:
            return {"error": "Team not found"}
        return {"deleted": True}

    def initialize_teams(self):
        teams_folder = os.path.join(os.path.dirname(__file__), "./data/teams-definitions")
        json_files = glob.glob(os.path.join(teams_folder, "*.json"))
        created = 0
        for file_path in json_files:
            with open(file_path, "r") as f:
                team = json.load(f)
                self.create_team(team)
                created += 1
        return f"Successfully created {created} teams."