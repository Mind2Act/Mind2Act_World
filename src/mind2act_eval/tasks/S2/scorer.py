"""S2: actual safe path and supported terminal goal; violations never stop the actor."""
from mind2act_eval.contracts import NativeEvidence, Score, to_json
from mind2act_eval.scoring import SCORING_VERSION
from .binding import VERSION, effective_config
from .evidence import read_rows
from .measurements import validate_frame
from .source_rules.buttons import Task


def score(evidence: NativeEvidence) -> Score:
    if evidence.task_id != "S2":
        raise ValueError("S2 scorer received a different task")
    values = to_json(evidence.values)
    c = effective_config(values["source_case"], input_version=values["input_protocol_version"])
    session = Task(c)
    initial = values["initial"]
    if initial["tick"] != 0:
        raise ValueError("S2 requires the native reset sample")
    validate_frame(c, initial)
    session.sample(initial)
    previous_time, last_tick, terminal_tick = initial["simulation_time_s"], 0, None
    for frame in read_rows(values["measurements"]):
        if frame["tick"] != last_tick + 1 or abs(frame["simulation_time_s"] - previous_time - 1/120) > 1e-6:
            raise ValueError("S2 physics measurements are not consecutive at 120 Hz")
        last_tick, previous_time = frame["tick"], frame["simulation_time_s"]
        validate_frame(c, frame)
        if session.status == "running":
            session.sample(frame)
            if session.status != "running":
                terminal_tick = frame["tick"]
    if last_tick != evidence.tick:
        raise ValueError("S2 native clock/terminal receipt mismatch")
    success = session.status == "running" and not session.violations and session.at_goal
    return Score(100.0 if success else 0.0, {"scoring_version": SCORING_VERSION, "task_success": success, "rubric": VERSION,
                 "input_protocol_version": values["input_protocol_version"],
                 "reason": session.reason, "native_status": session.status,
                 "terminal_tick": terminal_tick, "button_commands": session.commands,
                 "goal_stable_ticks": session.goal_ticks})
