import copy
import pytest
from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.CP08.audit import Audit
from mind2act_eval.tasks.CP08.instances import selected
from mind2act_eval.tasks.CP08.physical import CONTACT_SCHEMA
from mind2act_eval.tasks.CP08.scorer import score
from test_cp08_projection_audit import projection


def raw(config,tick):
    c=next(b['target'] for b in config['blocks'] if b['id']=='C')
    return dict(build_area_bounds_m=dict(lower=[c[0]-.0915,c[1]-.0515],upper=[c[0]+.0915,c[1]+.0515]),tick=tick,blocks={b['id']:dict(position=b['initial'],orientation_wxyz=[1.,0.,0.,0.],
        linear_velocity_m_s=[0.,0.,0.]) for b in config['blocks']+config.get('distractors',[])},
        contacts=[],contact_measurement_schema=CONTACT_SCHEMA,cue_visible=True,
        tcps={s:[0.,0.,1.] for s in ('Left','Right')},
        flange_poses={s:dict(position=[-.145,0.,1.],orientation_wxyz=[1.,0.,0.,0.]) for s in ('Left','Right')},
        joints={s:[0.]*6 for s in ('Left','Right')},
        measured_finger_positions_m={s:[.044,.044] for s in ('Left','Right')})


@pytest.mark.parametrize('level', ['easy','medium','hard'])
def test_no_action_never_credits_preinstalled_blocks(tmp_path,level):
    row=selected(level);config=row['source_case'];a=Audit(tmp_path/'raw.jsonl')
    for tick in range(3):a.append(raw(config,tick))
    receipt=a.receipt();a.close()
    e=NativeEvidence('CP08',row['instance_id'],2,dict(source_case=config,
        source_projection=projection(level),audit=receipt))
    result=score(e)
    assert result.value==0
    assert result.components['completion']['n']=={'easy':3,'medium':4,'hard':5}[level]
    assert not result.components['task_success']


def test_source_geometry_tampering_invalid(tmp_path):
    row=selected('easy');config=copy.deepcopy(row['source_case']);config['tolerance']*=2
    with pytest.raises(ValueError,match='source instance'):
        score(NativeEvidence('CP08',row['instance_id'],0,dict(source_case=config)))


def test_scene_retains_any_early_eof_without_stepping_sim(tmp_path):
    from types import SimpleNamespace
    from mind2act_eval.contracts import InstanceSpec
    from mind2act_eval.tasks.CP08.instances import instructions
    from mind2act_eval.tasks.CP08.scene import StructureScene
    row=selected('easy')
    spec=InstanceSpec(task_id='CP08',instance_id=row['instance_id'],difficulty='easy',
        source_id='fr3-replay-new15-20261031',config_version='cp08-source-projection-v1',
        asset_version='test-assets',protocol_version='v1',config=dict(
        source_case=row['source_case'],scene_sha256=row['scene_sha256'],physics_budget=row['physics_budget'],
        source_spec_sha256=row['source_spec_sha256'],source_projection=projection(),
        instruction=instructions('easy')['en']))
    class Sensors:
        errors=[]
        def read(self,tick):return raw(row['source_case'],tick)
        def set_cue(self,visible):assert visible
        def close(self):pass
    scene=StructureScene(spec,SimpleNamespace(output_dir=tmp_path),Sensors())
    for tick in range(1,31):scene.before_step(tick);scene.after_step(tick)
    assert score(scene.evidence(30)).value==0
    assert not scene.is_complete()
    scene.close()


@pytest.mark.parametrize('names,expected',[(('B',),5.),(('B','A'),10.),(('B','A','C'),100.),(('A','B','C'),100.),(('B','C','A'),100.),(('dualB',),5.)])
def test_raw_owned_contacts_to_final_numeric_score(tmp_path,names,expected):
    from test_cp08_physical import trajectory, dual_trajectory
    config,initial,normalized=dual_trajectory() if names==('dualB',) else trajectory(names)
    # These assertions exercise formal prefix rules after the visible cue.
    # Cue contact immunity is covered independently by the physical tests.
    idle=copy.deepcopy(normalized[0])
    for name,body in idle['blocks'].items():
        body.update(position=list(initial['blocks'][name]['position']),
                    table_support=True,supports=[],robot_contact=False,
                    bilateral_by_arm={'Left':False,'Right':False})
    idle['measured_jaws']={side:[.044,.044] for side in ('Left','Right')}
    normalized=[copy.deepcopy(idle) for _ in range(288)]+normalized
    for tick,state in enumerate(normalized,1):state['tick']=tick
    source=selected('easy');a=Audit(tmp_path/'raw.jsonl');a.append(raw(config,0))
    for state in normalized:
        r=raw(config,state['tick']);contacts={}
        r['measured_finger_positions_m']=state['measured_jaws']
        r['tcps']=state['tcp_by_arm']
        r['flange_poses']={s:dict(position=[p[0]-.145,*p[1:]],orientation_wxyz=[1.,0.,0.,0.]) for s,p in r['tcps'].items()}
        for name,body in state['blocks'].items():
            root='/World/CP11/Blocks/'+name+'/collision'
            r['blocks'][name]['position']=body['position']
            others=[]
            if body['table_support']:others.append(('/World/Worktable/collision',True))
            others.extend(('/World/CP11/Blocks/'+s+'/collision',True) for s in body['supports'])
            for side,on in body['bilateral_by_arm'].items():
                if on:others.extend(('/World/Robot'+side+'/Asset/'+link+'/collision',False) for link in ('link7','link8'))
            for other,support in others:
                pair=sorted((root,other));sign=1 if pair[0]==root else -1
                contacts[tuple(pair)]=dict(pair=pair,impulse_ns=[0.,0.,.001],contact_count=1,
                    owned_support_points=[dict(normal_on_pair0=[0.,0.,sign if support else 0.],separation_m=-.0001)])
        r['contacts']=list(contacts.values());r['cue_visible']=r['tick']<=288;a.append(r)
    receipt=a.receipt();a.close()
    result=score(NativeEvidence('CP08',source['instance_id'],len(normalized),
        dict(source_case=config,source_projection=projection(),audit=receipt)))
    assert result.value==expected


def test_missing_initial_contact_schema_is_invalid(tmp_path):
    source=selected('easy');initial=raw(source['source_case'],0)
    initial.pop('contact_measurement_schema');a=Audit(tmp_path/'raw.jsonl');a.append(initial)
    receipt=a.receipt();a.close()
    with pytest.raises(ValueError,match='initial owned contact'):
        score(NativeEvidence('CP08',source['instance_id'],0,dict(source_case=source['source_case'],
            source_projection=projection(),audit=receipt)))
