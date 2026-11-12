"""CP05 physical key edges, sequence LCS and independent final rack return."""
from mind2act_eval.scoring import cp_score_from_completion
from collections import Counter

from mind2act_eval.contracts import NativeEvidence, Score
from mind2act_eval.scoring import SCORING_VERSION, lcs_length

NOTES = {'easy':8,'medium':8,'hard':12}
TOOLS = {'easy':('single',),'medium':('left','right'),'hard':('left','right')}


def score(evidence: NativeEvidence) -> Score:
    v=evidence.values; difficulty=v['difficulty']; target=tuple(v['target_keys'])
    if difficulty not in NOTES or len(target)!=NOTES[difficulty]:
        raise ValueError('CP05 difficulty/target note count differs')
    e=v['cp05_evidence']; hard=difficulty=='hard'
    play,demo=e['play_tick'],e['demonstration_completed_tick']
    play_ok=bool(e['triggered_play'] and play is not None)
    events=tuple(e['replay_events'])
    if any(b['press_tick']<a['press_tick'] for a,b in zip(events,events[1:])):
        raise ValueError('CP05 physical key events must be chronological')
    usable=tuple(r['key'] for r in events if play_ok and e['demonstration_completed']
                 and demo is not None and r['press_tick']>=demo)
    note_k=lcs_length(usable,target)
    tools=tuple(e['tools'])
    returned=bool(Counter(t['tool_id'] for t in tools)==Counter(TOOLS[difficulty])
                  and all(t['returned'] for t in tools))
    max_tick=e.get('volume_max_tick',e.get('volume_max_released_tick'))
    max_ok=bool(e['volume']=='MAX' and max_tick is not None and play is not None and max_tick<play)
    k=int(play_ok)+note_k+int(returned)+(int(max_ok) if hard else 0)
    n=NOTES[difficulty]+2+int(hard)
    success=bool(k==n and usable==target and len(events)==len(target))
    return cp_score_from_completion(success,note_k,len(target),difficulty=difficulty,components={
        'scoring_version':SCORING_VERSION,'note_lcs':note_k,
        'play':play_ok,'rack_return':returned,'max_before_play_and_at_t0':max_ok if hard else None})
