"""Versioned CP04 evaluation, historical source and Easy diagnostic clocks.

The opt-in is part of the immutable instance, never an environment/global knob.
Logical mechanism/scoring time stays 120 Hz in every profile.
"""
from dataclasses import dataclass

SOURCE = 'cp04-source-physics960-v1'
EVALUATION = 'cp04-evaluation-physics120-v1'
DIAGNOSTIC = 'cp04-easy-physics120-diagnostic-v1'
LOGICAL_HZ = 120


@dataclass(frozen=True)
class CookingClock:
    version: str = EVALUATION
    physics_hz: int = 120

    @property
    def substeps(self):
        return self.physics_hz // LOGICAL_HZ

    @property
    def initialization_ticks(self):
        return self.physics_hz // 10  # same 0.1 s before logical tick zero

    def receipt(self):
        return dict(clock_profile_version=self.version, physics_hz=self.physics_hz,
                    logical_hz=LOGICAL_HZ, initialization_physics_ticks=self.initialization_ticks,
                    source_physics_hz=960,
                    comparison_group=f'cp04-physics{self.physics_hz}-logical120-v1',
                    diagnostic_only=self.version == DIAGNOSTIC)


def resolve(spec):
    version = spec.config.get('cp04_clock_profile', EVALUATION)
    if version == EVALUATION:
        if (spec.clock.physics_hz, spec.clock.control_hz, spec.clock.rgb_hz) != (120, 30, 30):
            raise ValueError('CP04 evaluation requires 120/30/30 physics/control/RGB clock')
        return CookingClock()
    if version == SOURCE:
        if spec.clock.physics_hz != 960 or spec.clock.control_hz not in (30, 120) or spec.clock.rgb_hz != 30:
            raise ValueError('CP04 source requires 960 Hz physics, explicit 30/120 Hz control and 30 Hz RGB')
        return CookingClock(SOURCE, 960)
    if (version != DIAGNOSTIC or spec.difficulty != 'easy'
            or (spec.clock.physics_hz, spec.clock.control_hz, spec.clock.rgb_hz) != (120, 30, 30)):
        raise ValueError('CP04 diagnostic requires explicit Easy120/30/30 clock profile')
    return CookingClock(DIAGNOSTIC, 120)
