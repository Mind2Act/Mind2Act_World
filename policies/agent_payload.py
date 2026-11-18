"""Reject incomplete or non-English multimodal agent inputs before provider I/O.

This gate checks required surfaces, not word-for-word task conformance. It does
not implement missing tools; the selected runtime still needs a rendered audit.
"""
from collections.abc import Mapping
import json
import re

# Include halfwidth forms and supplementary kana/Hangul at the send boundary.
_CJK = re.compile("[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7ff"
                  "\uf900-\ufaff\ufe30-\ufe4f\uff65-\uffdc"
                  "\U00016fe0-\U00018dff\U0001aff0-\U0001b2ff\U0001f200-\U0001f2ff\U00020000-\U0003ffff]")
_TOOLS = ("move_eef", "give_up", "wait_steps", "memory_resolve")
_OBSERVATION = (
    "Instruction:", "Measured grasp-point state:",
    "Read-only arm joint angles (radians):", "Previous action outcome:",
    "Remaining simulated time (seconds):", "Remaining control rows (",
    "Remaining LLM decision calls:", "Current RGB:",
)


def require_english(value):
    """Check history/feedback and wire strings/keys without echoing their text."""
    if isinstance(value, str):
        if _CJK.search(value):
            raise ValueError("multimodal agent payload contains CJK text")
        if value.lstrip().startswith(('{', '[')):
            try:
                decoded = json.loads(value)
            except ValueError:
                pass
            else:
                require_english(decoded)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            require_english(key)
            require_english(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            require_english(child)


def validate_input(messages, tools):
    """Fail before reserving an upstream attempt for a legacy/incomplete input."""
    require_english((messages, tools))
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("multimodal agent input requires System, Goal and Observation")
    if not all(isinstance(message, Mapping) for message in messages):
        raise ValueError("multimodal agent messages must be objects")
    system, goal, observation = messages[0], messages[1], messages[-1]
    system_text = system.get("content")
    if (system.get("role") != "system" or not isinstance(system_text, str)
            or not system_text.startswith("You are controlling a simulated robot embodiment named 'robodojo-arx-x5'.")
            or "Embodiment notes:" not in system_text):
        raise ValueError("multimodal agent common simulated-robot System is required")
    goal_text = goal.get("content")
    if (goal.get("role") != "user" or not isinstance(goal_text, str)
            or not goal_text.startswith("Goal: ") or not re.search(r'\n\s*TASK RECIPE:\n', goal_text)
            or not goal_text.partition("TASK RECIPE:\n")[0][len("Goal: "):].strip()
            or not goal_text.partition("TASK RECIPE:\n")[2].strip()):
        raise ValueError("multimodal agent Goal must include a nonempty TASK RECIPE")
    if (not isinstance(tools, list) or not all(isinstance(tool, Mapping) and isinstance(tool.get('function'), Mapping) for tool in tools)
            or tuple(tool['function'].get('name') for tool in tools) != _TOOLS):
        raise ValueError("multimodal agent input requires move_eef, give_up, wait_steps, memory_resolve in order")
    content = observation.get("content")
    if (observation.get("role") != "user" or not isinstance(content, list)
            or not all(isinstance(block, Mapping) for block in content)):
        raise ValueError("multimodal agent current Observation must be a user content-block message")
    text = "\n".join(block.get("text", "") for block in content if block.get("type") == "text")
    cursor = 0
    for label in _OBSERVATION:
        index = text.find(label, cursor)
        if index < 0:
            raise ValueError("multimodal agent current Observation template is incomplete or out of order")
        cursor = index + len(label)
