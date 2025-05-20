# backend/guard_utils.py
import re
import json
import logging
from autogen_agentchat.messages import TextMessage

# preserve reference to the unwrapped orchestrate_step
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator
_orig = MagenticOneOrchestrator._orchestrate_step

_LOG = logging.getLogger(__name__)

async def _guard(self, cancellation_token=None):
    """
    Wraps the orchestrator’s step function to:
      1) Ensure progress_ledger and tool_call slots are always present.
      2) Regex‐parse any “Call foo(bar: 1, baz: 2)” text answers into a tool_call.
    """
    # 1. Ensure the ledger structure
    if getattr(self, "progress_ledger", None) is None:
        self.progress_ledger = {}
    # Normalize known sections
    self.progress_ledger.setdefault("is_request_satisfied", {"answer": "", "plan": ""})
    self.progress_ledger.setdefault("instruction_or_question", {"answer": "", "plan": ""})
    self.progress_ledger.setdefault("tool_call", {"tool_name": "", "arguments": {}})

    _LOG.debug("Orchestrator ledger BEFORE guard: %r", self.progress_ledger)

    try:
        result = await _orig(self, cancellation_token=cancellation_token)
    except Exception as exc:
        # Only swallow/paraphrase specific ledger-parsing failures
        if "parse ledger" in str(exc) or (isinstance(exc, KeyError) and exc.args == ("answer",)):
            _LOG.warning(
                "⚠️  LLM returned unparsable/incomplete progress-ledger; resetting. Details: %s",
                exc,
            )
            # Reinitialize minimal ledger
            self.progress_ledger = {
                "is_request_satisfied": {"answer": False, "plan": False},
                "instruction_or_question": {"answer": "", "plan": ""},
                "tool_call": {"tool_name": "", "arguments": {}},
            }
            return None
        raise

    # Extract a function call instruction, e.g., "call data_provider('sensor')"
    instr = self.progress_ledger.get("instruction_or_question", {}).get("answer", "")
    m = re.search(
        r"call\s+([a-zA-Z_][\w]*)\s*\(\s*([^)]*?)\s*\)",
        instr,
        flags=re.IGNORECASE,
    )
    if m and not self.progress_ledger["tool_call"].get("tool_name"):
        fname = m.group(1)
        params = m.group(2)
        args = {}
        # Split on commas between arguments
        for part in [p.strip() for p in params.split(",")] if params else []:
            if ":" in part:
                k, v = part.split(":", 1)
                k = k.strip().strip("'\"")
                try:
                    args[k] = json.loads(v)
                except:
                    args[k] = v.strip("'\"")
            else:
                # Single positional argument → treat as 'tablename'
                args["tablename"] = part.strip("'\"")
        self.progress_ledger["tool_call"] = {"tool_name": fname, "arguments": args}

    _LOG.debug("Orchestrator ledger AFTER guard: %r", self.progress_ledger)
    return result