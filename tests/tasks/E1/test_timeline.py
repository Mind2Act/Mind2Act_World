from mind2act_eval.tasks.E1.mechanism.timeline import Timeline


class Transport:
    def __init__(self):
        self.calls = []

    def show(self, rn): self.calls.append(("show", rn))
    def hide(self, rn): self.calls.append(("hide", rn))
    def retire(self, rn, t): self.calls.append(("retire", rn))
    def advance_queue(self, rn, t): self.calls.append(("queue", rn))

    def start_motion(self, index, start, end, time, speed):
        self.calls.append(("move", index))
        return 0.05


class Evidence:
    def __init__(self): self.committed_at = None; self.t = 0; self.phases = []

    def sample(self, row):
        self.phases.append(row["phase"])
        return False


def make(rounds=2):
    ev = Evidence()
    rs = [dict(observation_center_offset_x=0.0) for _ in range(rounds)]
    tl = Timeline(Transport(), rs, dict(belt_speed_m_s=0.3, observation_seconds=0.1),
                  [(-1.0, 1.0)] * rounds, [0.0] * rounds, ev)
    return tl, ev


def test_phases_progress_to_execute_for_second_round():
    tl, ev = make()
    for t in range(1, 3000):
        tl.before_tick(t)
        tl.after_tick(t, dict(physics_tick=t))
        if tl.round == 1 and tl.phase == "batch_execute":
            break
    assert tl.round == 1 and tl.phase == "batch_execute"
    assert ev.phases[0] == "batch_arrival"
    assert "batch_departure" in ev.phases and "batch_blank" in ev.phases


def test_first_batch_is_observe_only():
    tl, ev = make()
    for t in range(1, 3000):
        tl.before_tick(t); tl.after_tick(t, dict(physics_tick=t))
        if tl.round == 1:
            break
    assert "batch_execute" not in ev.phases[: ev.phases.index("batch_blank")]


def test_seal_ends_episode_only_on_last_round():
    tl, ev = make(rounds=1)
    ev.sample = lambda row: True
    tl.before_tick(1); tl.after_tick(1, dict(physics_tick=1))
    assert tl.terminal_reason == "completed" and tl.complete
    tl, ev = make(rounds=3)
    ev.sample = lambda row: True
    tl.before_tick(1); tl.after_tick(1, dict(physics_tick=1))
    assert tl.terminal_reason is None and tl.phase == "batch_departure"  # a seal never ends the episode
