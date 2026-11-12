"""PR3 per-tick measurement of the pen tip.

Each physics tick records, from the simulated nib body and native contact reports, whether the tip touches
the board validly (inside the paper, at paper height) and the contact point. Only valid-ink samples are
kept, so evidence grows with the trace and not with the whole episode.
"""
from mind2act_eval.contracts import NativeEvidence


class TipRecorder:
    def __init__(self, task_id, instance_id, parameters, paper, ink, direction_gate, settle_ticks):
        self._task, self._instance = task_id, instance_id
        self._parameters, self._paper, self._ink = parameters, paper, ink
        self._ticks, self._strokes, self._points = [], [], []
        self._stroke, self._previous_contact, self._board_at_end = 0, False, False
        self._measured = 0
        self._direction_gate, self._settle = bool(direction_gate), settle_ticks
        self._lifted_run = 0

    def sample(self, tick, board_contact, point):
        """point = nib center minus its radius (the sphere low point)."""
        if tick != self._measured + 1:
            raise ValueError(f"tick {tick} skips the measurement record")
        self._measured = tick
        if self._previous_contact and not board_contact:
            self._stroke += 1
        self._previous_contact = board_contact
        self._board_at_end = board_contact
        self._lifted_run = 0 if board_contact else self._lifted_run + 1
        ink = self._ink
        valid = bool(board_contact and abs(point[0]) <= ink['valid_x_m']
                     and abs(point[1] - self._paper['center_y_m']) <= ink['valid_y_half_m'])
        if valid:
            self._ticks.append(tick)
            self._strokes.append(self._stroke)
            self._points.append((float(point[0]), float(point[1]), self._paper['top_z_m']))
        return valid

    def evidence(self, tick) -> NativeEvidence:
        return NativeEvidence(self._task, self._instance, tick, dict(
            parameters=self._parameters, paper=self._paper, measured_ticks=self._measured,
            direction_gate=self._direction_gate,
            normal_completion=bool(self._ticks) and self._lifted_run >= self._settle,
            board_contact_at_end=self._board_at_end,
            samples=dict(tick=self._ticks, stroke=self._strokes, point=self._points)))
