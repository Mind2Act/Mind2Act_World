"""M1 per-tick evidence freezes each native row once; scoring stays identical."""
from types import SimpleNamespace
import pytest
import mind2act_eval.contracts as contracts
from mind2act_eval.tasks.M1.evidence import ContactTable,Recorder
from mind2act_eval.tasks.M1.scene import RouteScene
from mind2act_eval.tasks.M1.instances import plain


def test_m1_real_recorder_and_evidence_never_refreeze_previous_ticks(monkeypatch):
 recorder=Recorder(dict(target='/World/T',cover='/World/C'),lambda _:[0,0,.7],
  lambda:([0,0,0],[0,0,0]),ContactTable(),dict(D='/World/Pad'),['cover'],lambda:[False,False])
 scene=object.__new__(RouteScene)
 scene.spec=SimpleNamespace(task_id='M1',instance_id='cpu-case')
 scene.recorder=recorder;scene.M=dict(case=dict(expected_path=['D']),stations=dict(D=dict(center_xy=[0,0])),
  submission_stations={},bodies=dict(target=dict(dimensions=[1,1,1])))
 scene.B=scene.M['bodies'];scene.target='target';scene.cover='cover';scene.physical_start=None
 scene.arrivals=[];scene.closed_baseline={};scene.target_initial_z=None
 visits=[0];original=contracts.freeze
 def counted(value):
  if isinstance(value,dict) and 'target_robot_contact' in value:visits[0]+=1
  return original(value)
 monkeypatch.setattr(contracts,'freeze',counted)
 snapshots=[]
 for tick in range(1,65):
  recorder.record(tick,'target');snapshots.append(scene.evidence(tick))
 assert visits[0]==64,'per-tick evidence must not recursively freeze the full prior history'
 assert len(snapshots[0].values['ticks'])==1 and len(snapshots[-1].values['ticks'])==64
 assert plain(snapshots[-1].values['ticks'])[-1]['tick']==64
 with pytest.raises(TypeError):snapshots[0].values['ticks'][0]['target_center'][0]=42


@pytest.mark.parametrize('route,expected',[
 (['D','B','C','A'],100.), (['D','B','A'],15.), (['A'],0.)])
def test_m1_persistent_evidence_preserves_json_and_full_scorer(route,expected):
 from test_m1_scorer import Film
 from mind2act_eval.tasks.M1.scorer import score
 film=Film(route).build();before=film.evidence();history=contracts.FrozenHistory()
 for row in film.rows:history=history.append(row)
 values=dict(before.values,ticks=history)
 after=contracts.NativeEvidence('M1','x',len(film.rows),values)
 assert contracts.to_json(before)==contracts.to_json(after)
 assert score(before)==score(after)
 assert score(after).value==expected
