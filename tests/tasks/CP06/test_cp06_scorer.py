import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP06.instances import selected
from mind2act_eval.tasks.CP06.audit import Audit
from mind2act_eval.tasks.CP06.scorer import score
from test_cp06_physical import fixture


@pytest.mark.parametrize('level,load,zero,green,rider,expected',[
    ('easy',False,False,True,False,0.),('easy',True,False,False,False,15.),
    ('medium',False,True,False,False,0.),('hard',True,True,False,False,7.),
    ('hard',True,True,False,True,7.),('hard',True,True,True,True,100.)])
def test_final_numeric_score_from_checked_append_audit(tmp_path,level,load,zero,green,rider,expected):
    core,records,initial=fixture(level,load=load,zero=zero,green=green,rider=rider)
    row=selected(level);a=Audit(tmp_path/'raw.jsonl');a.append(dict(tick=0,raw=initial))
    for r in records:a.append(dict(tick=r['physics_tick'],raw=r['raw']))
    receipt=a.receipt();a.close()
    e=NativeEvidence('CP06',row['instance_id'],len(records),dict(difficulty=level,
        source_spec_sha256=row['source_spec_sha256'],source_task_sha256=row['declared_task_sha256'],
        source_task=core,audit=receipt))
    assert score(e).value==pytest.approx(expected)


def test_derived_export_cannot_masquerade_as_original_h5_spec():
    row=selected('easy')
    with pytest.raises(ValueError,match='original selected'):
        score(NativeEvidence('CP06',row['instance_id'],0,dict(difficulty='easy',
            source_spec_sha256=row['derived_spec_sha256'],source_task_sha256=row['declared_task_sha256'])))


def test_explicit_give_up_at_tick_zero_has_complete_initial_audit_and_zero(tmp_path):
    level='easy';core,records,initial=fixture(level);row=selected(level)
    a=Audit(tmp_path/'raw.jsonl');a.append(dict(tick=0,raw=initial));receipt=a.receipt();a.close()
    e=NativeEvidence('CP06',row['instance_id'],0,dict(difficulty=level,
        source_spec_sha256=row['source_spec_sha256'],source_task_sha256=row['declared_task_sha256'],
        source_task=core,audit=receipt))
    assert score(e).value==0.


@pytest.mark.parametrize('bottom_z', [0., .014974, .014975])
def test_real_pan_support_has_no_added_minimum_collider_bottom_height(tmp_path,bottom_z):
    from mind2act_eval.tasks.CP06.terminal import TerminalFacts
    core,records,initial=fixture('easy',load=True,green=True)
    for r in records:
        for obj in r['raw']['objects'].values():obj['min_pan_local_z_m']=bottom_z
    row=selected('easy');a=Audit(tmp_path/'bottom-height.jsonl');a.append(dict(tick=0,raw=initial))
    facts=TerminalFacts(core,initial)
    for r in records:
        a.append(dict(tick=r['physics_tick'],raw=r['raw']));facts.append(r['raw'])
    receipt=a.receipt();a.close()
    e=NativeEvidence('CP06',row['instance_id'],len(records),dict(difficulty='easy',
        source_spec_sha256=row['source_spec_sha256'],source_task_sha256=row['declared_task_sha256'],
        source_task=core,audit=receipt))
    assert score(e).value==100
    assert facts.candidate()


@pytest.mark.parametrize('level', ['easy','medium','hard'])
def test_extra_weight_fall_caps_the_public_score_at_partial_credit(tmp_path, level):
    from test_cp06_physical import extra_weight_fall_fixture
    core, records, initial, path = extra_weight_fall_fixture(level)
    row = selected(level)
    audit = Audit(tmp_path/'extra-weight-fall.jsonl')
    audit.append(dict(tick=0, raw=initial))
    for record in records:
        audit.append(dict(tick=record['physics_tick'], raw=record['raw']))
    receipt = audit.receipt()
    audit.close()
    evidence = NativeEvidence('CP06',row['instance_id'],len(records),dict(difficulty=level,
        source_spec_sha256=row['source_spec_sha256'],source_task_sha256=row['declared_task_sha256'],
        source_task=core,audit=receipt))
    result = score(evidence)
    assert result.components['physical_report']['dropped_objects'] == (path,)
    assert result.value == {'easy': 15., 'medium': 10., 'hard': 7.}[level]
    assert not result.components['task_success']
