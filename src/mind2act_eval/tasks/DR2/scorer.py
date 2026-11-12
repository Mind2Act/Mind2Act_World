"""DR2 score from native button measurements only (CommonInput snapshot 20261101 DR2 sections 5-6).

k = independently correct cues. All ten correct without unlit presses scores 100;
otherwise each correct cue earns 3 / 2.2 / 1.5 points in Easy / Medium / Hard.
Failed cues do not end the round. The cue clock is replayed on the per-tick physical
rows; no expert label and no live-clock verdict is read. The scene's live clock must agree with the replay
(mismatch = invalid evidence).
"""
from mind2act_eval.contracts import NativeEvidence, Score, score_from_completion

from .event_clock import EventClock
from mind2act_eval.scoring import SCORING_VERSION


def replay(values):
    clock = EventClock(values['sequence'], values['press_ticks'], values['release_ticks'], values['gap_ticks'],
                       values['release_depression_m'], values['press_depression_m'])
    ticks = values['ticks']
    if not ticks or any(row['tick'] != i for i, row in enumerate(ticks, 1)):
        raise ValueError('DR2 requires every physics tick, first tick = 1')
    for row in ticks:
        clock.advance(row['tick'], row['button_depression_m'], row['button_contacts'])
    return clock


def score(evidence: NativeEvidence) -> Score:
    v = evidence.values
    clock = replay(v)
    keys = ('index', 'button', 'cue_tick', 'press_tick', 'release_tick', 'status', 'reason')
    if [{k: r[k] for k in keys} for r in clock.records] != [{k: r[k] for k in keys} for r in v['live_records']]:
        raise ValueError('live cue records differ from the independent replay')
    k, total = clock.valid_count, len(clock.sequence)
    level = v.get('difficulty') or {480:'easy',384:'medium',312:'hard'}[v['press_ticks']]
    cap = {'easy':30.,'medium':22.,'hard':15.}[level]
    base = score_from_completion(k == total and not clock.unlit_presses, k, total, components=dict(
        scoring_version=SCORING_VERSION, unlit_presses=clock.unlit_presses,
        round_complete=clock.done,
        failures=[dict(index=r['index'], reason=r['reason']) for r in clock.records if r['status'] == 'failed']))

    return Score(100. if base.value==100. else cap*k/total, dict(base.components, difficulty=level, partial_ceiling=cap))
