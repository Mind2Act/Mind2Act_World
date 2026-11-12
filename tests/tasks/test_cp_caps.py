"""Reviewed caps reward core objects, never setup-only progress."""
import importlib
import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.scoring import SCORING_VERSION

CAPS={'easy':15,'medium':10,'hard':7}

@pytest.mark.parametrize('task,core,steps,key',[
 ('CP01',[3,6,9],[3,8,12],'credited_foods'),
 ('CP02',[4,6,6],[4,8,10],'credited_macarons'),
 ('CP03',[3,5,7],[4,7,9],'credited_plates')])
@pytest.mark.parametrize('tier',['easy','medium','hard'])
@pytest.mark.parametrize('count',[0,1])
def test_only_core_units_earn_partial_credit(task,core,steps,key,tier,count):
 i=['easy','medium','hard'].index(tier);n=core[i]
 v=dict(difficulty=tier,scoring_version=SCORING_VERSION,required_steps=steps[i],
        steps_satisfied=steps[i]-n+count,wrong_or_extra_items=0,
        task_success=False,physical_task_success=False,task_process_pass=False)
 v[key]=count
 if task=='CP02':v.update(core_wrong_or_extra_items=0,placements=[dict(stable=j<count) for j in range(n)])
 s=importlib.import_module('mind2act_eval.tasks.'+task+'.scorer').score(NativeEvidence(task,'caps',1,v))
 assert s.value==pytest.approx(CAPS[tier]*count/n)
 assert s.components['completion']==dict(k=count,n=n,ratio=count/n)

@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_cp04_setup_without_ingredients_earns_zero(tier):
 from tests.tasks.test_cp04_scorer import good,run
 v=good(tier);v['cp04_evidence']['ingredient_events']=[]
 assert run(v).value==0

@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_cp05_play_max_return_without_notes_earns_zero(tier):
 from tests.tasks.test_cp05_scorer import good,run
 v=good(tier);v['cp05_evidence']['replay_events']=[]
 assert run(v).value==0


def test_cp02_cover_error_blocks_full_success_without_deducting_macarons():
 from mind2act_eval.tasks.CP02.scorer import score
 v=dict(difficulty='hard',scoring_version=SCORING_VERSION,required_steps=10,
        steps_satisfied=9,wrong_or_extra_items=1,credited_macarons=6,
        core_wrong_or_extra_items=0,physical_task_success=False,task_process_pass=False)
 s=score(NativeEvidence('CP02','cover-error',1,v))
 assert s.value==7 and s.components['core_k']==6


@pytest.mark.parametrize('task',['CP01','CP02'])
def test_old_aggregate_is_not_upgraded_by_core_arithmetic(task):
 scorer=importlib.import_module('mind2act_eval.tasks.'+task+'.scorer').score
 with pytest.raises(ValueError,match='evidence version'):
  scorer(NativeEvidence(task,'old',1,dict(difficulty='easy')))
