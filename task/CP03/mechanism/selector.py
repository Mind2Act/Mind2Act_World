"""CP03 gear selector: a measured-angle one-way pawl plus finite detent resistance.

Source: cp03-em-b-20261028 / cp03-hard-b-20261028 cases/CP/CP03/runtime/selector_mechanics.py
(identical in both). No pose writes, task-target lookup, contact suppression or commanded
motion. Negative X rotation advances 1 -> 2 -> 3; a physical joint limit blocks reversal.

Changed from the source only structurally: a class with `step()` replaces the closure that
wrapped `world.after_step`; the call order (physics grading, then this, then plate quality)
is preserved by CP03Scene.after_step.
"""
from mind2act_eval.tasks.CP03.evidence import euler_xyz_deg


class SelectorMechanics:
    def __init__(self, stage, gear_body, joint_path):
        from pxr import UsdPhysics

        self._gear = gear_body
        self._joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joint_path))
        self._drive = UsdPhysics.DriveAPI.Get(self._joint.GetPrim(), "angular")
        self.state = {"minimum_deg": self.angle(), "held_target_deg": self.angle(),
                      "reverse_slack_deg": .25, "detent_yield_deg": 1.5}
        self._drive.GetStiffnessAttr().Set(3.)
        self._drive.GetDampingAttr().Set(.08)
        self._drive.GetMaxForceAttr().Set(.15)
        self._drive.GetTargetPositionAttr().Set(self.state["held_target_deg"])
        self._joint.GetUpperLimitAttr().Set(min(19., self.state["minimum_deg"] + .25))

    def angle(self):
        return float(euler_xyz_deg(self._gear.get_world_pose()[1])[0])

    def step(self):
        state, drive, joint = self.state, self._drive, self._joint
        measured = self.angle()
        state["minimum_deg"] = min(state["minimum_deg"], measured)
        joint.GetUpperLimitAttr().Set(max(-18.75, min(19., state["minimum_deg"] + .25)))
        target = state["held_target_deg"]
        if measured < target - state["detent_yield_deg"]:
            target = measured
        # The three physical detents are independent of the requested task gear.
        for detent in (18., 0., -18.):
            if abs(measured - detent) < .5 and detent < target:
                target = detent
        state["held_target_deg"] = target
        drive.GetTargetPositionAttr().Set(target)
        # Native finger acceptance may alter stiffness; preserve this mechanism.
        drive.GetStiffnessAttr().Set(3.)
        drive.GetDampingAttr().Set(.08)
        drive.GetMaxForceAttr().Set(.15)
