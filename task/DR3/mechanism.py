"""The selected FR3 slider schedules, independent of any policy or expert.

Source: original scene.py:schedule and dual_config.py:Gate.state at b6432b4.
All values come from the frozen source case; targets do not establish an opening.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Gate:
    name: str
    x: float
    period: float
    first_open: float
    open_seconds: float
    travel: float = .34
    transition: float = .8

    def target(self, time):
        if time < self.first_open - self.transition:
            return 0.
        phase = (time - self.first_open) % self.period
        if phase < self.open_seconds:
            return self.travel
        if phase < self.open_seconds + self.transition:
            return self.travel * (1 - (phase - self.open_seconds) / self.transition)
        if phase >= self.period - self.transition:
            return self.travel * (phase - self.period + self.transition) / self.transition
        return 0.


def single_target(time, source):
    phase = time % source['PERIOD']
    closed, transition, duration = source['CLOSED'], source['TRANS'], source['DUR']
    if phase < closed:
        return 0.
    if phase < closed + transition:
        return source['TRAVEL'] * (phase - closed) / transition
    if phase < closed + transition + duration:
        return source['TRAVEL']
    return source['TRAVEL'] * (source['PERIOD'] - phase) / transition


def gates(source, difficulty):
    if difficulty != 'hard':
        return {'single': None}
    result = {}
    for row in source['gates']:
        values = {key: row[key] for key in ('name', 'x', 'period', 'first_open', 'open_seconds')}
        values.update(travel=source['TRAVEL'], transition=source['TRANS'])
        gate = Gate(**values)
        if not all(math.isfinite(float(value)) for key, value in values.items() if key != 'name'):
            raise ValueError('nonfinite DR3 gate timing')
        result[gate.name] = gate
    if tuple(result) != ('Slow', 'Fast'):
        raise ValueError('DR3 Hard requires the original ordered Slow/Fast gates')
    return result


def gate_definitions(source, difficulty):
    return [dict(id=name, origin=[source['GATE_X'] if gate is None else gate.x,
                                source['GATE_Y'], source['Z'] + .005],
                 aperture_low_yz=[-source['W'] / 2, 0.],
                 aperture_high_yz=[source['W'] / 2, source['H']],
                 half_thickness_m=.007, fully_open_position_m=source['TRAVEL'],
                 open_tolerance_m=.003, closed_tolerance_m=.002)
            for name, gate in gates(source, difficulty).items()]
