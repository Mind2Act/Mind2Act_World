"""CPU raw-fact checks; no source expert replay or native simulation claim."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.tasks.P2.binding import SOURCE_ID,VERSION,DISMANTLING_RULE,digest,geometry,instruction,validate
from mind2act_eval.tasks.P2.evidence import Rows,read_rows
from mind2act_eval.tasks.P2.measurements import bilateral,robot_paths,supports,tilt,validate_frame


def source(level='easy'):
    n=3 if level=='easy' else 4
    return dict(instance_id='P_'+level+'_train_000',difficulty=level,split='train',seed=0,
        target_foods_bottom_to_top=['food'+str(i) for i in range(n)],
        target_drop_xyz={f'Tier{i+1}':[i*.1,-.22,.784] for i in range(n)},
        decoy_foods=['d1','d2'] if level=='hard' else [],
        decoy_xyz={'Decoy1':[-.1,-.22,.784],'Decoy2':[.1,-.22,.784]} if level=='hard' else {})


def frame(row,tick=0):
    g=geometry(row);points={**g['target_initial_xyz'],'Lid':g['lid_initial_xyz'],**g['decoy_xyz']}
    return dict(physics_tick=tick,simulation_time_s=8.025+tick/120,
        poses={n:dict(position=p,quaternion_wxyz=[1.,0.,0.,0.],linear_velocity_m_s=[0.,0.,0.],
                      angular_velocity_rad_s=[0.,0.,0.]) for n,p in points.items()},
        preplaced_xyz=[-.35,-.06,.833],q={'Left':[0.]*8,'Right':[0.]*8},
        qd={'Left':[0.]*8,'Right':[0.]*8},contacts=[])


def contact(a,b,normal=(0.,0.,1.)):
    return dict(pair=[a,b],active=True,points=[dict(position=[0.,0.,0.],normal=list(normal),
        impulse=[0.,0.,.01],separation=0.)])


@pytest.mark.parametrize('level',['easy','medium','hard'])
def test_resolver_sha_and_canonical_instruction_bind_without_mutating_source(tmp_path,level):
    row=source(level);before=deepcopy(row)
    path=tmp_path/'scene.usda';path.write_text('#usda 1.0\n')
    cfg=dict(source_instance=row,source_record_sha256=digest(row),instruction=instruction(level),
             dismantling_rule_version=DISMANTLING_RULE,
             input_protocol_version=VERSION,scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             task_scene_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             food_assets_sha256={f:hashlib.sha256(path.read_bytes()).hexdigest()
                                 for f in row['target_foods_bottom_to_top']+row['decoy_foods']})
    spec=InstanceSpec('P2','synthetic',level,SOURCE_ID,VERSION,'test','v1',config=cfg)
    ctx=NS(assets=NS(resolve=lambda name:path))
    assert validate(spec,ctx)[1]==geometry(row) and row==before
    path.write_text('#usda 1.0\n# changed\n')
    with pytest.raises(ValueError,match='scene SHA'):
        validate(spec,ctx)


def test_coding_goal_explains_current_rule_and_e2e_instruction_is_frozen():
    from pathlib import Path
    from mind2act_eval.tasks.P2 import binding
    goals=json.loads((Path(binding.__file__).with_name('config')/'goals.json').read_text())
    assert 'without using any distractor baskets' in instruction('hard')
    assert 'onto the mats' in instruction('hard')
    assert goals['goals']['hard'].split('\n\n',1)[0]=='Goal: '+instruction('hard').replace('onto the mats', 'onto the table or wooden board').replace(' without using any distractor baskets','')
    assert 'do not use' not in goals['goals']['hard']
    assert 'movement/use' not in goals['goals']['hard']
    assert 'within 1 mm displacement' not in goals['goals']['hard']
    assert 'No mat, assigned drop location or landing region is scored' in goals['goals']['hard']
    assert 'distractor' not in instruction('easy')


def test_current_native_support_preserves_manifold_owner_normal_sign():
    row=frame(source())
    row['contacts']=[contact('/World/Task/Tier1/Collision','/World/Task/Board/Collision')]
    validate_frame(row,row['poses'])
    assert supports(row,'Tier1')==['Board']
    assert supports(row,'Board')==[]
    row['contacts'][0]['pair'].reverse()
    row['contacts'][0]['points'][0]['normal']=[0.,0.,-1.]
    assert supports(row,'Tier1')==['Board']
    row['contacts'][0]['points'][0]['normal']=[1.,0.,0.]
    assert supports(row,'Tier1')==[]


def test_bilateral_grasp_requires_both_real_fingers_on_the_same_arm():
    row=frame(source());name='/World/Task/Tier1/Collision'
    row['contacts']=[contact(name,'/World/RobotLeft/Asset/link7/Collision')]
    assert robot_paths(row,'Tier1') and not bilateral(row,'Tier1')
    row['contacts'].append(contact(name,'/World/RobotRight/Asset/link8/Collision'))
    assert not bilateral(row,'Tier1')
    row['contacts'][-1]['pair'][1]='/World/RobotLeft/Asset/link8/Collision'
    assert bilateral(row,'Tier1')
    row['contacts'][-1]['active']=False
    assert not bilateral(row,'Tier1')


def test_actual_orientation_determines_tilt_without_a_claimed_tilt_field():
    from math import sin,cos,radians
    assert tilt([1.,0.,0.,0.])==0
    assert tilt([cos(radians(5)/2),sin(radians(5)/2),0.,0.])==pytest.approx(5)
    with pytest.raises(ValueError,match='zero'):
        tilt([0.,0.,0.,0.])


def test_measurement_seal_rejects_tamper_and_missing_ticks(tmp_path):
    rows=Rows(tmp_path/'measurements.jsonl')
    row=frame(source(),1);rows.append(row)
    receipt=rows.receipt()
    assert list(read_rows(receipt))==[row]
    with pytest.raises(ValueError,match='gap'):
        rows.append(frame(source(),3))
    rows.close()
    raw=(json.dumps(frame(source(),2))+'\n').encode();rows.path.write_bytes(raw)
    with pytest.raises(ValueError,match='SHA'):
        list(read_rows(receipt))


def test_nonfinite_physical_evidence_is_rejected():
    row=frame(source());row['poses']['Tier1']['position'][0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        validate_frame(row,row['poses'])
