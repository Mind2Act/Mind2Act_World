"""Synthetic common-input fixture, with no task answers or native state."""
def valid_input():
    messages = [
        {"role": "system", "content": "You are controlling a simulated robot embodiment named 'arx-x5'.\n\nEmbodiment notes:\nTwo arms."},
        {"role": "user", "content": "Goal: Move the visible block.\n\nTASK RECIPE:\n## Capability dimension\nPlacement.\n\n## Description\nUse the tray.\n\n## Completion\nPlace the block in the tray.\n\n## Scoring\n100 if it is in the tray; otherwise 0."},
        {"role": "user", "content": [{"type": "text", "text":
            "Instruction: Move the visible block.\nMeasured grasp-point state:\nleft_x=0.1\n"
            "Read-only arm joint angles (radians):\nleft_joint1=0\nPrevious action outcome:\nNo previous action.\n"
            "Remaining simulated time (seconds): 10\nRemaining control rows (30 Hz): 300\n"
            "Remaining LLM decision calls: 150\nCurrent RGB:"}]},
    ]
    tools = [{"type": "function", "function": {"name": name, "description": "Public tool.", "parameters": {"type": "object"}}}
             for name in ("move_eef", "give_up", "wait_steps", "memory_resolve")]
    return messages, tools
