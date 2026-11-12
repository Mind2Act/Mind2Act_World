"""Per-tick bridge facts and autonomous physical start interlock, CPU testable.

Inputs are measured poses/contacts from scene.py. Slots and acceptable geometry
come from the private, scene-bound source manifest, never a policy/expert trace.
No phase, active-piece selector, action index or expert-written milestone is read.
"""
from copy import deepcopy
from math import acos, dist, isfinite

from .scorer import SCHEMA
from mind2act_eval.scoring import SCORING_VERSION, PressEdge
from .rules import L_ROAD_RULE, placed, rule_for_kind
from .binding import goal_pose

# Selected runtime.py lane test: road_width / 2 - .019 (vehicle half-width).
VEHICLE_HALF_WIDTH_M = .019


def rotation_error(a, b):
    """Sign-independent angular distance of normalized wxyz quaternions."""
    def unit(q):
        if len(q) != 4 or not all(isfinite(x) for x in q):
            raise ValueError("CP07 invalid physical quaternion")
        norm = sum(x*x for x in q) ** .5
        if norm < 1e-12:
            raise ValueError("CP07 zero physical quaternion")
        return [x/norm for x in q]
    return 2 * acos(min(1., abs(sum(x*y for x, y in zip(unit(a), unit(b))))))


class BridgeFacts:
    def __init__(self, binding, initial):
        self.binding = binding
        self.difficulty = binding["identity"]["difficulty"]
        self.pieces = {p["id"]: p for p in binding["pieces"]}
        self._validate_state(initial)
        self.initial = deepcopy(initial)
        self.last = deepcopy(initial)
        self.history = {name: dict(started=False, ticks=0, bilateral=0, lift_m=0.) for name in self.pieces}
        self.tick = 0
        self.button_rest = initial["button_z_m"]
        self.press_ticks = 0
        self.press_open = False
        self.press_eligible = False
        self.pending_press = False
        self.valid_button = False
        self.starts = 0
        self.launched = False
        self.route_valid = True
        self.parking_ok = self._parked(initial["car"])
        self.max_joint_error = 0.
        self.sampled = False
        self.button_edge=PressEdge(.0007)

    def _parked(self, car):
        if self.difficulty == "easy":
            return True
        b = self.binding
        # Keep the selected runtime's x/z/up parking conditions. The same
        # source vehicle/lane geometry bounds lateral displacement; no new
        # free tolerance or unsourced heading threshold is introduced.
        return (abs(car["p_m"][0] - b["car_start_x_m"]) <= .008
                and abs(car["p_m"][1] - b["lane_y_m"]) <= b["road_width_m"] / 2 - VEHICLE_HALF_WIDTH_M
                and abs(car["p_m"][2] - b["roadtop_start_m"]) <= .003
                and car["up"] >= .98)

    def _validate_state(self, state):
        if set(state["pieces"]) != set(self.pieces):
            raise ValueError("CP07 physical piece evidence missing")
        for p in state["pieces"].values():
            if len(p["p_m"]) != 3 or not all(isfinite(v) for v in p["p_m"]):
                raise ValueError("CP07 non-finite physical pose")
            rotation_error(p["q_wxyz"], p["q_wxyz"])
            if any(type(p[k]) is not bool for k in ("bilateral", "robot_contact", "inside_bridge")):
                raise ValueError("CP07 contact facts must be booleans")
            if any(not isinstance(path, str) or not path.startswith("/World/") for path in p["support_contacts"]):
                raise ValueError("CP07 support contact path missing")
        values = [state["button_z_m"], state["car"]["up"],
                  state["car"]["heading_rad"], *state["car"]["p_m"]]
        if (len(state["car"]["p_m"]) != 3 or not all(type(v) in (int, float) and isfinite(v) for v in values)
                or state.get("joint_error_rad") is not None and (not isfinite(state["joint_error_rad"]) or state["joint_error_rad"]<0)
                or any(type(state[k]) is not bool for k in ("button_robot_contact", "hands_clear"))):
            raise ValueError("CP07 invalid native state")
        for key in ('far_bank_supported','far_bank_center_inside'):
            if type(state['car'][key]) is not bool:raise ValueError('CP07 actual bank contact/geometry required')

    def _slot(self, slot, state):
        candidates = []
        for name in slot["compatible_piece_ids"]:
            p = state["pieces"][name]
            h = self.history[name]
            goal = goal_pose(slot, name)
            # A supported lower layer must be measured at its final position;
            # a contact with an unrelated stock block cannot stand in for it.
            support = any(path in p["support_contacts"] for path in slot["support_paths"])
            rule = rule_for_kind(slot["kind"])
            angular = rotation_error(p["q_wxyz"], goal["q_wxyz"]) if rule == L_ROAD_RULE else None
            candidates.append(dict(piece_id=name, lift_m=h["lift_m"],
                kind=slot["kind"], orientation_rule=rule, orientation_error_rad=angular,
                bilateral_fraction=h["bilateral"] / h["ticks"] if h["ticks"] else 0.,
                position_error_m=dist(p["p_m"], goal["p_m"]),
                released=not p["robot_contact"], supported=support))
        # Prefer an actually valid transported piece over a closer ungrasped
        # interchangeable piece. One-to-one assignment is done per solution.
        return sorted(candidates, key=lambda x: (not self._placed(x), x["position_error_m"]))

    @staticmethod
    def _placed(row):
        return placed(row)

    def solutions(self, state):
        rows = []
        for solution in self.binding["solutions"]:
            candidates = [self._slot(slot, state) for slot in solution["slots"]]
            best, best_k = None, -1

            def measured(index, raw, selected):
                row = dict(raw)
                contacts = state["pieces"][row["piece_id"]]["support_contacts"]
                lower = solution["slots"][index].get("support_slots", [])
                # Dynamic support must be the actual one-to-one assigned lower
                # body, already proven at its own final pose in this prefix.
                row["supported"] |= any(i in selected and self.pieces[selected[i]]["path"] in contacts
                                        for i in lower)
                return row

            def finish(selected):
                nonlocal best, best_k
                # Complete the material assignment geometrically after the first
                # physical failure. This does not award any later placement.
                owner = {name: index for index, name in selected.items()}
                fixed = set(owner)
                def fill(index, seen):
                    choices = sorted(candidates[index], key=lambda r: (
                        not state["pieces"][r["piece_id"]]["inside_bridge"], r["position_error_m"]))
                    for row in choices:
                        name = row["piece_id"]
                        if name in fixed or name in seen:
                            continue
                        seen.add(name)
                        if name not in owner or fill(owner[name], seen):
                            owner[name] = index
                            return True
                    return False
                for index in range(len(selected), len(candidates)):
                    if not fill(index, set()):
                        raise ValueError("CP07 solution cannot assign distinct compatible stock")
                complete = {index: name for name, index in owner.items()}
                slots = [measured(i, next(r for r in candidates[i] if r["piece_id"] == complete[i]), selected)
                         for i in range(len(candidates))]
                used = set(complete.values())
                extra = [i for name, p in state["pieces"].items() if name not in used and p["inside_bridge"]
                         for i, slot in enumerate(solution["slots"])
                         if set(p["support_contacts"]).intersection(
                             (*slot["support_paths"], self.pieces[slots[i]["piece_id"]]["path"]))]
                error = min(extra) if extra else None
                k = min(len(selected), error) if error is not None else len(selected)
                if k > best_k:
                    best_k, best = k, dict(slots=slots, first_extra_slot=error)

            def search(index, selected):
                if index == len(candidates):
                    finish(selected)
                    return
                found = False
                for raw in candidates[index]:
                    name = raw["piece_id"]
                    if name in selected.values() or not self._placed(measured(index, raw, selected)):
                        continue
                    found = True
                    search(index + 1, {**selected, index: name})
                if not found:
                    finish(selected)
            search(0, {})
            rows.append(best)
        return rows

    def after_step(self, tick, state):
        if tick != self.tick + 1:
            raise ValueError("CP07 requires every consecutive physics tick")
        self._validate_state(state)
        self.tick, self.sampled = tick, True
        self.last = deepcopy(state)
        self.max_joint_error = max(self.max_joint_error, state.get('joint_error_rad') or 0.)
        self.parking_ok = self._parked(state['car'])
        pressed=self.button_edge.update(self.button_rest-state['button_z_m'])
        if pressed:self.starts+=1
        launch=bool(pressed and not self.launched)
        if launch:self.launched=True;self.valid_button=True
        return launch

    def snapshot(self):
        car=deepcopy(self.last['car'])
        values=dict(schema=SCHEMA,scoring_version=SCORING_VERSION,difficulty=self.difficulty,
                    native_samples_available=self.sampled,terminal_car=car,drive_launched=bool(self.launched),
                    car_reached=bool(self.launched and car['far_bank_supported'] and car['far_bank_center_inside']),
                    stock_count=len(self.pieces),valid_start_count=self.starts,
                    binding_mode=self.binding['mode'],solutions_scope=self.binding['solutions_scope'])
        if self.binding['mode']=='source_bound_test':
            values.update(self.binding['original_source']);values['source_protocol']=self.binding['source_protocol']
        return values
