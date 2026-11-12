from types import SimpleNamespace
import pytest
import mind2act_eval.contracts as contracts
from mind2act_eval.tasks.F1.scene import SliderScene


def test_f1_tick_evidence_does_not_refreeze_previous_native_rows(monkeypatch):
    scene = SliderScene(SimpleNamespace(task_id='F1', difficulty='easy', instance_id='case', config={'instruction':'original'}),
                        None, dict(case={}, protocol={}, contacts=SimpleNamespace(pairs={})))
    visited = [0]
    original = contracts.freeze
    def counted(value):
        if isinstance(value, dict) and value.get('old_row_marker') is True:
            visited[0] += 1
        return original(value)
    monkeypatch.setattr(contracts, 'freeze', counted)
    snapshots = []
    for tick in range(64):
        result = scene.rows.append({'tick':tick,'old_row_marker':True,'q':[0.,1.]})
        if result is not None:
            scene.rows = result
        snapshots.append(scene.evidence(tick))
    assert visited[0] == 64, 'each physics row must be frozen once, not once per future tick'
    assert len(snapshots[0].values['rows']) == 1
    assert len(snapshots[-1].values['rows']) == 64
    assert contracts.to_json(snapshots[-1])['values']['rows'][-1]['tick'] == 63
    with pytest.raises(TypeError):
        snapshots[0].values['rows'][0]['q'][0] = 42


@pytest.mark.parametrize('route,expected', [(['B'],100.), (['A','B','B'],15.), (['A'],0.)])
def test_persistent_trace_preserves_full_f1_score_and_export(route, expected):
    from test_f1_judge import Rig, plan
    from mind2act_eval.tasks.F1.scorer import score
    rig = Rig()
    rig.run(2400, plan(['B','A'],route))
    before = rig.evidence()
    history = contracts.FrozenHistory()
    for row in rig.rows:
        history = history.append(row)
    after = contracts.NativeEvidence('F1','x',rig.tick,dict(case=rig.case,protocol=rig.protocol,rows=history))
    assert contracts.to_json(before) == contracts.to_json(after)
    assert score(before) == score(after)
    assert score(after).value == expected


def test_persistent_trace_rejects_nonfinite_rows_and_mutation():
    history = contracts.FrozenHistory().append({'value':[1.]})
    with pytest.raises(ValueError):
        history.append({'value':float('nan')})
    with pytest.raises(TypeError):
        history._length = 100
    with pytest.raises(TypeError):
        del history._length
    assert history[-1]['value'] == (1.,)
