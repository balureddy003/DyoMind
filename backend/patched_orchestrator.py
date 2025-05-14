# patched_orchestrator.py

from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator

class PatchedMagenticOneOrchestrator(MagenticOneOrchestrator):
    def __init__(self, *args, model_client=None, **kwargs):
        kwargs["model_client"] = model_client
        super().__init__(*args, **kwargs)