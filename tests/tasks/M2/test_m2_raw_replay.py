"""Synthetic measured IO, never a claim of native dynamics or training success."""
import json
from copy import deepcopy

import pytest

from mind2act_eval.contracts import to_json
from mind2act_eval.tasks.M2.evidence import RawTrace, replay_raw
from mind2act_eval.tasks.M2.scorer import score
from test_m2_physical import Trace


@pytest.mark.parametrize('tier,pairs,expected', [('easy', 2, 100), ('medium', 2, 11.0), ('hard', 8, 100), ('hard', 0, 0.0)])
def test_terminal_raw_replay_at_eof_budget_or_success(tmp_path,tier,pairs,expected):
    t=Trace(tier)
    t.scene.trace=RawTrace(tmp_path/'physical.jsonl',t.spec,t.binding,t.scene.last)
    for i in range(pairs):t.pair(i)
    if not pairs:t.press(0)  # budget/EOF during a flip, no invented completed pair
    terminal=t.scene.evidence(t.tick)
    receipt=terminal.values['raw_physical']
    t.scene.close()
    replay=replay_raw(t.spec,receipt)
    assert score(replay).value == expected == score(terminal).value
    assert to_json(replay.values) == {k:to_json(v) for k,v in terminal.values.items() if k!='raw_physical'}
    assert replay.tick==t.tick==receipt['physics_rows']
    assert t.scene.game.events==[]
    assert len(t.scene.game.pairs)<=8


def test_zero_row_eof_and_no_history_resident(tmp_path):
    t=Trace()
    t.scene.trace=RawTrace(tmp_path/'raw.jsonl',t.spec,t.binding,t.scene.last)
    receipt=t.scene.trace.receipt()
    assert score(t.scene.evidence(0)).value==score(replay_raw(t.spec,receipt)).value==0
    assert not hasattr(t.scene.trace,'rows')
    t.scene.close()


@pytest.mark.parametrize('damage',['hash','identity','gap','nan','contact_type'])
def test_raw_corruption_rejected(tmp_path,damage):
    t=Trace();p=tmp_path/'raw.jsonl'
    t.scene.trace=RawTrace(p,t.spec,t.binding,t.scene.last);t.step(2)
    receipt=t.scene.trace.receipt();t.scene.close()
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    if damage=='identity':rows[0]['binding']['identity']['seed']+=1
    if damage=='gap':rows[1]['physics_tick']=2
    if damage=='nan':rows[1]['physical']['button_depths_m'][0]=float('nan')
    if damage=='contact_type':rows[1]['physical']['finger_contacts'][0]=1
    p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError):replay_raw(t.spec,receipt)
