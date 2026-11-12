from copy import deepcopy
from types import SimpleNamespace
import pytest
from mind2act_eval.tasks.CP06.scene import BalanceScene
from mind2act_eval.tasks.CP06.scorer import score
from mind2act_eval.tasks.CP06.terminal import TerminalFacts
from mind2act_eval.tasks.CP06.pose_views import flanges
from test_cp06_bindings import runtime_spec
from test_cp06_physical import fixture


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_bounded_terminal_matches_full_positive_physical_fixture(level):
    config,records,initial=fixture(level,load=True,zero=level!='easy',green=True,rider=level=='hard')
    facts=TerminalFacts(config,initial)
    for row in records:facts.append(row['raw'])
    assert len(facts.tail)==360 and facts.candidate()


def test_no_action_never_replays_growing_history_per_tick(tmp_path,monkeypatch):
    import mind2act_eval.tasks.CP06.scene as scene_module
    config,records,initial=fixture('easy',green=True)
    class Source:
        index=0
        def read(self):
            if not self.index:
                raw=deepcopy(records[0]['raw']);raw.update(initial);return raw
            return records[self.index-1]['raw']
        def before_step(self):pass
        def after_step(self):self.index+=1
    def forbidden(*args):raise AssertionError('no terminal candidate, no full-history evaluation')
    monkeypatch.setattr(scene_module,'evaluate',forbidden)
    scene=BalanceScene(runtime_spec(),SimpleNamespace(output_dir=tmp_path),Source())
    for tick in range(1,601):
        scene.before_step(tick);scene.after_step(tick);assert not scene.is_complete()
    evidence=scene.evidence(600)
    assert 'raw_rows' not in evidence.values and evidence.values['audit']['rows']==601
    assert score(evidence).value==0.  # Green is a full-success prerequisite, not a core item.
    scene.close()


def test_readonly_robot_frame_constructor_never_prepares_contacts_or_resets():
    calls=[]
    class View:
        def __init__(self,**kw):
            assert kw['reset_xform_properties'] is kw['prepare_contact_sensors'] is kw['track_contact_forces'] is kw['disable_stablization'] is False
            assert kw['contact_filter_prim_paths_expr']==[];calls.append(kw)
    stage=SimpleNamespace(GetPrimAtPath=lambda p:SimpleNamespace(HasAPI=lambda api:True))
    world=SimpleNamespace(scene=SimpleNamespace(add=lambda v:v))
    assert set(flanges(stage,world,view_type=View,non_root=lambda **kw:True))=={'Left','Right'}
    assert len(calls)==2


@pytest.mark.parametrize('level', ['easy','medium','hard'])
def test_extra_weight_fall_never_becomes_a_full_success_candidate(level):
    from test_cp06_physical import extra_weight_fall_fixture
    config, rows, initial, path = extra_weight_fall_fixture(level)
    facts = TerminalFacts(config, initial)
    for row in rows:
        facts.append(row['raw'])
    assert facts.drops.dropped == {path}
    assert not facts.candidate()
    scene = object.__new__(BalanceScene)
    scene.config, scene.rows, scene.initial, scene.facts = config, rows, initial, facts
    scene._candidate = scene._complete = False
    assert not scene.is_success()
