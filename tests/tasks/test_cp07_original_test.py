"""Original source-qualified eight-piece inputs and synthetic physical receipts."""
import base64
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mind2act_eval.contracts import InstanceSpec,NativeEvidence,to_json
from mind2act_eval import original_test_inputs as source
from mind2act_eval.tasks.CP07 import original_inputs as original
from mind2act_eval.tasks.CP07.binding import validate
from mind2act_eval.tasks.CP07.rules import rule_for_kind
from mind2act_eval.tasks.CP07.scorer import score
from mind2act_eval.tasks.CP07.scene import BridgeScene


@pytest.fixture(scope='module')
def rows():
    path=os.environ.get('MIND2ACT_ORIGINAL_SNAPSHOT');request=os.environ.get('MIND2ACT_ORIGINAL_REQUEST')
    if not path or not request:pytest.skip('private original fixed50 sources not configured')
    raw=Path(path).read_bytes();assert source.digest(raw)==source.SNAPSHOT_SHA256
    return [r for r in json.loads(raw)['records'] if r['task_id']=='CP07'],Path(request)


def fixture(tmp_path,row,request):
    raw=base64.b64decode(row['source_bytes_base64']);data=json.loads(raw);slot=row['logical_slot']
    (tmp_path/'source.json').write_bytes(raw)
    metadata=dict(schema=source.SCHEMA,instance_set='test50-r5',split='test',logical_slot=slot,
        native_case_id=row['native_case_id'],source_document_id=None,source_seed=row['seed'],
        source_configuration_sha256=row['source_configuration_sha256'],source_index_sha256=row['source_index']['sha256'],
        source_request_sha256=source.REQUEST_SHA256,snapshot_sha256=source.SNAPSHOT_SHA256)
    spec=InstanceSpec('CP07',slot,row['difficulty'],'cp07-original8-train-v2-cpu','1','cpu','v1',
        seed=row['seed'],config=dict(cp07_mode='source_bound_test',test_source=metadata))
    assets=NS(resolve=lambda key:request if key=='test_source_request' else tmp_path/'source.json')
    by_id={p['name']:p for p in data['stock']}
    binding=dict(schema='cp07.source-binding.v2',mode='source_bound_test',source_protocol=original.PROTOCOL,
        solutions_scope='original-source-solutions',identity={k:getattr(spec,k) for k in
            ('task_id','instance_id','difficulty','source_id','seed','asset_version')},
        source_manifest_sha256=row['source_configuration_sha256'],source_code_sha256=original.WORLD_SHA256,
        scene_sha256='a'*64,original_source=source.audit_source(spec,assets),
        pieces=[dict(id=p['name'],path=p['path'],kind=p['kind'],geometry_signature=original.geometry(p)) for p in data['stock']],
        bridge_bounds_m=dict(lo=[-.25,-.22,.768],hi=[.22,.02,.95]),button_xy_m=[.255,-.070],
        car_start_x_m=-.190,roadtop_start_m=.8243,roadtop_end_m=.85,lane_y_m=-.1,road_width_m=.065,finish_x_m=.170,
        solutions=[])
    for solution in data['solutions']:
        slots=[]
        for i,name in enumerate(original.sequence(solution,by_id)):
            piece=by_id[name]
            lower=([i-1] if 0<i<=3 else [2] if i in (4,5) else [3] if i>5 else [])
            paths=(['/World/Terrain/Foundation'] if i==0 else ['/World/Terrain/BankLeft'] if i in (4,5)
                   else ['/World/Terrain/BankRight'] if i>5 else [])
            slots.append(dict(**original.pose(data,solution,i),kind=piece['kind'],orientation_rule=rule_for_kind(piece['kind']),
                compatible_piece_ids=[n for n,p in by_id.items() if original.geometry(p)==original.geometry(piece)],
                support_slots=lower,support_paths=paths))
        binding['solutions'].append(dict(source_solution=solution,slots=slots))
    validate(spec,binding)
    return spec,binding,data


@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_all50_original_cases_keep_stock_all_solutions_and_source_identity(rows,tmp_path,tier):
    records,request=rows;selected=[r for r in records if r['difficulty']==tier];assert len(selected)==50
    for row in selected:
        spec,binding,data=fixture(tmp_path,row,request)
        assert len(binding['pieces'])==original.STOCK[tier]
        assert len(binding['solutions'])==data['solution_count']
        assert all(len(s['slots'])==8 for s in binding['solutions'])
        assert spec.seed==data['seed'] and binding['source_manifest_sha256']==row['source_configuration_sha256']


class Trace:
    def __init__(self,tmp_path,row,request):
        self.spec,self.binding,self.raw=fixture(tmp_path,row,request);self.tick=0;self.drives=[]
        self.state=dict(pieces={p['name']:dict(p_m=p['start'][:],q_wxyz=[1.,0.,0.,0.],
            robot_contact=False,bilateral=False,support_contacts=['/World/Worktable'],inside_bridge=False) for p in self.raw['stock']},
            button_z_m=.780,button_robot_contact=False,hands_clear=True,joint_error_rad=0.,
            car=dict(p_m=[-.190,-.1,.8243],up=1.,heading_rad=0.,far_bank_supported=True,far_bank_center_inside=False))
        def read():
            measured=deepcopy(self.state);car=measured['car']
            car['far_bank_center_inside']=bool(.170<=car['p_m'][0]<=.5 and -.3<=car['p_m'][1]<=.1)
            car['far_bank_supported']=bool(car['p_m'][2]>.75)
            return measured
        sensors=NS(read=read,drive=lambda *v:self.drives.append(v))
        self.scene=BridgeScene(self.spec,NS(),self.binding,sensors)
    def step(self,n=1):
        for _ in range(n):
            self.tick+=1;self.scene.before_step(self.tick);self.scene.after_step(self.tick)
    def assemble(self,solution=0,count=8):
        by_id={p['name']:p for p in self.raw['stock']};plan=self.binding['solutions'][solution]
        names=original.sequence(self.raw['solutions'][solution],by_id)
        for i,name in enumerate(names[:count]):
            p=self.state['pieces'][name];p.update(robot_contact=True,bilateral=True);self.step()
            p['p_m'][2]+=.05;p['support_contacts']=[];self.step()
            slot=plan['slots'][i]
            p.update(p_m=slot['p_m'][:],q_wxyz=slot['q_wxyz'][:],robot_contact=False,bilateral=False,inside_bridge=True,
                support_contacts=slot['support_paths']+[by_id[names[j]]['path'] for j in slot['support_slots']])
            self.step()
    def press(self):
        self.state.update(button_z_m=.779,button_robot_contact=True);self.step(8)
        self.state.update(button_z_m=.780,button_robot_contact=False);self.step()
    def result(self):return score(self.scene.evidence(self.tick))


@pytest.mark.parametrize('tier',['easy','medium','hard'])
def test_source_eight_scoring_prefix_launch_finish_and_success_lock(rows,tmp_path,tier):
    records,request=rows;row=next(r for r in records if r['difficulty']==tier)
    t=Trace(tmp_path,row,request);t.assemble(count=3)
    assert t.result().value==0 and t.result().components['completion']['n']==1
    # Fresh physical sequence to avoid using initial placement as a lift receipt.
    target=tmp_path/'complete';target.mkdir();t=Trace(target,row,request);t.assemble();t.press()
    assert t.drives==[(True,)] and t.result().value==0
    t.state['car']['p_m'][0]=.170;t.step();assert t.result().value==100
    t.state['car']['p_m'][0]=.171;t.step();assert t.result().value==100
    assert t.scene.is_complete() and t.drives[-1]==(False,)
    t.state['car']['p_m']=[.5,.5,0.];t.step();assert t.result().value==0


@pytest.mark.parametrize('mutation',['split','seed','stock','solutions','target','lane','diagnostic','source_sha'])
def test_source_eight_binding_rejects_cross_source_and_legacy_aliases(rows,tmp_path,mutation):
    records,request=rows;row=next(r for r in records if r['difficulty']=='medium')
    spec,binding,_=fixture(tmp_path,row,request);c=to_json(spec.config)
    if mutation=='split':c['test_source']['split']='train'
    elif mutation=='seed':spec=replace(spec,seed=spec.seed+1)
    elif mutation=='stock':binding['pieces'][0]['geometry_signature'][1]+=.001
    elif mutation=='solutions':binding['solutions'].pop()
    elif mutation=='target':binding['solutions'][0]['slots'][0]['p_m'][2]+=.013
    elif mutation=='lane':binding['lane_y_m']=-.160
    elif mutation=='diagnostic':c['cp07_mode']='diagnostic_training';binding['mode']='diagnostic_training'
    else:c['test_source']['source_configuration_sha256']='b'*64
    with pytest.raises(ValueError):validate(replace(spec,config=c),binding)


def test_source_eight_scorer_cannot_self_declare_an_original_identity(rows,tmp_path):
    records,request=rows;row=next(r for r in records if r['difficulty']=='hard')
    t=Trace(tmp_path,row,request);t.step();v=to_json(t.scene.evidence(t.tick).values)
    v['test_source']['source_index_sha256']='b'*64
    with pytest.raises(ValueError):score(NativeEvidence('CP07',t.spec.instance_id,t.tick,v))


def test_unselected_source_material_uses_geometry_targets_not_teacher_dest(rows,tmp_path):
    records,request=rows;row=next(r for r in records if r['difficulty']=='medium')
    _,binding,raw=fixture(tmp_path,row,request);by_id={p['name']:p for p in raw['stock']}
    assert original.sequence(raw['solutions'][0],by_id)!=raw['selected_sequence']
    assert by_id['SupportThick1']['dest'][2]==pytest.approx(.787)
    assert binding['solutions'][0]['slots'][0]['p_m'][2]==pytest.approx(.774)


@pytest.mark.parametrize('failure',['lane','up'])
def test_original_world_route_limits_apply_without_canonical_steering(rows,tmp_path,failure):
    records,request=rows;row=next(r for r in records if r['difficulty']=='hard')
    t=Trace(tmp_path,row,request);t.assemble();t.press();t.state['car']['p_m'][0]=.171
    if failure=='lane':t.state['car']['p_m'][1]+=.009
    else:t.state['car']['up']=.93
    t.step();assert t.result().value==100
    assert t.drives==[(True,)]
