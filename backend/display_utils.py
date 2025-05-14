

import json
from typing import Optional, Dict, Any
from schemas import AutoGenMessage
from autogen_agentchat.messages import (
    MultiModalMessage, TextMessage,
    ToolCallExecutionEvent, ToolCallRequestEvent,
    SelectSpeakerEvent, ToolCallSummaryMessage
)
from autogen_agentchat.base import TaskResult
import crud

def get_current_time() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

async def display_log_message(
    log_entry: Any,
    logs_dir: str,
    session_id: str,
    user_id: str,
    conversation: Optional[Dict] = None
) -> AutoGenMessage:
    # (Paste here the full body of your previous display_log_message,
    # but update calls to `app.state.db` to use `crud` directly.)
    # For example:
    msg = AutoGenMessage(
        time=get_current_time(),
        session_id=session_id,
        session_user=user_id
    )

    if isinstance(log_entry, TaskResult):
        msg.type    = "TaskResult"
        msg.source  = "TaskResult"
        msg.content = log_entry.messages[-1].content
        msg.stop_reason = log_entry.stop_reason
        crud.store_conversation(log_entry, msg, conversation)

    # ... (other branches unchanged) ...

    crud.save_message(
        id=None,
        user_id=user_id,
        session_id=session_id,
        message=msg.to_json(),
        agents=None,
        run_mode_locally=None,
        timestamp=msg.time
    )
    return msg