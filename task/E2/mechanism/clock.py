"""Training display clock; no action labels or replay metadata are read."""


def display_state(tick, case):
    learn_end = round(case["learn_seconds"] * 120)
    shuffle_end = learn_end + round(case["shuffle_seconds"] * 120)
    ready = input_start(case)
    if tick < learn_end:
        return "learn", -105.0, False
    if tick < shuffle_end:
        progress = (tick - learn_end) / (shuffle_end - learn_end)
        closed = min(1.0, max(0.0, progress / .25 if progress < .5 else (1 - progress) / .25))
        return "shuffle", -105.0 * (1 - closed), progress >= .5
    if tick < ready:
        return "reasoning", -105.0, True
    return "input", -105.0, True


def input_start(case):
    return round((case["learn_seconds"] + case["shuffle_seconds"] + case["reasoning_seconds"]) * 120)
