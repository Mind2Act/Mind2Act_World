"""Automatic E1 batch timeline: arrival -> observe -> (first batch departs) -> execute -> departure.

Source: E-142 baseline `cases/E/sdk_task.py:Timeline`, decoupled from the SDK clock. Per the
Feishu snapshot a batch seals at its first 1 s stable placement (benchmark maintainer 10-30: no per-batch deadline; a batch
with no submission waits until the episode budget ends and counts 0), and an incorrect batch does not
end the episode. The phase is the
task's own native state (batch conveyor schedule), not an expert label.
"""
PHYSICS_HZ = 120


class Timeline:
    def __init__(self, transport, rounds, timing, intervals, carrier_x, evidence):
        self.transport, self.rounds, self.timing = transport, rounds, timing
        self.intervals, self.carrier_x, self.evidence = intervals, carrier_x, evidence
        self.round, self.phase, self.deadline = 0, "initial", 0
        self.terminal_reason = None

    def _duration(self, start, end, tick):
        return self.transport.start_motion(self.round, start, end, tick / PHYSICS_HZ, self.timing["belt_speed_m_s"])

    def enter(self, tick):
        rn = self.round
        self.transport.show(rn)
        self.transport.advance_queue(rn, tick / PHYSICS_HZ)
        duration = self._duration(self.carrier_x[rn], self.rounds[rn]["observation_center_offset_x"], tick)
        self.phase, self.deadline = "batch_arrival", tick + round(duration * PHYSICS_HZ) + 1

    def depart(self, tick):
        duration = self._duration(self.carrier_x[self.round], self.intervals[self.round][1] + .003, tick)
        self.phase, self.deadline = "batch_departure", tick + round(duration * PHYSICS_HZ) + 1

    def before_tick(self, tick):
        if self.phase == "initial":
            self.enter(tick)

    def after_tick(self, tick, measured):
        """`measured` is mutated with phase/round and fed to the evidence recorder."""
        measured.update(phase=self.phase, round=self.round)
        if self.evidence.sample(measured):
            # A sealed batch (correct or not) never stops the episode: later batches continue.
            if self.round == len(self.rounds) - 1:
                self.terminal_reason = "completed"
            else:
                self.depart(tick)
        elif tick >= self.deadline:
            if self.phase == "batch_arrival":
                self.phase = "batch_observe"
                self.deadline = tick + round(self.timing["observation_seconds"] * PHYSICS_HZ)
            elif self.phase == "batch_observe":
                if self.round == 0:
                    self.depart(tick)
                else:
                    self.phase = "batch_execute"  # stays until the batch seals; no per-batch deadline
            elif self.phase == "batch_departure":
                self.transport.hide(self.round)
                self.transport.retire(self.round, tick / PHYSICS_HZ)
                self.phase = "batch_blank"
                self.deadline = tick + max(10, round(self.timing.get("inter_batch_blank_seconds", 1 / 12) * PHYSICS_HZ))
            elif self.phase == "batch_blank":
                self.round += 1
                self.enter(tick)

    @property
    def complete(self):
        return self.terminal_reason is not None
