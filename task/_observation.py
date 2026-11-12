"""Shared Observation construction for task scenes (same clock/protocol handling everywhere)."""
from mind2act_eval.contracts import InstanceSpec, Observation

BASELINE_PHYSICS_HZ = 120


def require_physics_hz(spec: InstanceSpec, hz: int = BASELINE_PHYSICS_HZ) -> None:
    """Ports whose mechanism timelines are authored at a fixed rate reject any other clock."""
    if spec.clock.physics_hz != hz:
        raise ValueError("%s is authored at %d Hz physics; spec clock is %d Hz" % (spec.task_id, hz, spec.clock.physics_hz))


def observation(spec: InstanceSpec, robot, tick: int, instruction: str = "") -> Observation:
    state = robot.observe()
    return Observation(tick=tick, sim_time_s=tick / spec.clock.physics_hz, left_arm=state["left_arm"],
                       right_arm=state["right_arm"], left_gripper=state["left_gripper"],
                       right_gripper=state["right_gripper"], rgb=state.get("rgb", {}),
                       instruction=instruction, protocol_version=spec.protocol_version)
