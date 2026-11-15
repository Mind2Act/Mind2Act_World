"""CPU full-spec admission plus existing task checks; never starts Kit/GPU."""
import argparse,sys,json,os,hashlib,importlib,copy
from pathlib import Path
from types import SimpleNamespace
sys.dont_write_bytecode=True

def task_checks(spec,assets=None):
    from mind2act_eval.contracts import to_json
    t=spec.task_id;c=to_json(spec.config);calls=[]
    def module(name):return importlib.import_module('mind2act_eval.tasks.'+t+'.'+name)
    if t in ('PR1','PR2','PR3','DR1'):
        module('scene').validate(spec);calls.append('scene.validate')
    elif t in ('DR3','DR4','E2','PR4','PR5','PR6','CP06','CP08','CP09'):
        module('instances').validate(spec);calls.append('instances.validate')
    elif t=='DR2':
        m=module('instances');m.validate_case(c,spec.difficulty);clock=module('scene').make_clock(spec,m.layout());assert clock.sequence==c['sequence'];calls+=['instances.validate_case','scene.make_clock']
    elif t=='CP04':
        module('binding').bind_recipe(spec);calls.append('binding.bind_recipe')
    elif t=='F1':
        i=module('instances');i.require_baseline_clock(spec.clock);module('judge').Judge(c['case'],c['protocol'],i.layout());from mind2act_eval.tasks.F1.space import shortest
        case=c['case'];assert shortest(case['space'],case['mapping'],case['execution']['start'],case['execution']['target']) is not None;calls+=['instances.require_baseline_clock','judge.Judge','space.shortest']
    elif t=='M1':
        i=module('instances');i.require_baseline_clock(spec.clock);m=c['manifest'];from mind2act_eval.tasks.M1.ledger import Ledger
        Ledger(m['submission_stations'],m['bodies'][m['target_body_id']]['dimensions'] if isinstance(m['bodies'],dict) else next(b for b in m['bodies'] if b['id']==m['target_body_id'])['dimensions'],i.layout()['ledger']);calls+=['instances.require_baseline_clock','ledger.Ledger']
    elif t=='CP01':
        setup=c['native_inputs']['setup'];assert setup['difficulty']==spec.difficulty and spec.clock.physics_hz==120;module('evidence').Evidence(setup,c['max_food_tilt_deg']);calls.append('evidence.Evidence and original build difficulty/clock checks')
    elif t=='CP02':
        native=c['native_inputs'];m=native['manifest'];cfg=native['config'];assert m['spec']['task']=='CP02' and m['spec']['difficulty']==cfg['difficulty']==m['difficulty']==spec.difficulty;assert len(m['spec']['items'])=={'easy':4,'medium':6,'hard':6}[spec.difficulty];module('evidence').Timeline(m['spec']);calls.append('evidence.Timeline and original build tier/item checks')
    elif t=='CP05':
        from mind2act_eval.tasks.CP05.mechanism.keyboard_clock import KeyboardClock
        from mind2act_eval.tasks.CP05.mechanism.key_press_policy import KeyPressPolicy
        KeyboardClock(c['setup']['sequence']);calls.append('mechanism.keyboard_clock.KeyboardClock')
        module('scene').validate(spec);calls.append('scene.validate')
    elif t=='S1':
        from mind2act_eval.tasks.S1.mechanism.schedule import timeline_config
        timeline_config(spec);calls.append('mechanism.schedule.timeline_config')
    elif t=='CP10':
        if spec.config.get('layout_split')=='validation':
            from mind2act_eval.cp10_validation_inputs import validate
        else:
            from mind2act_eval.cp10_training_inputs import validate
        validate(spec);module('protocol').validate(spec.config['collection_protocol'],spec.config['rule_version']);calls+=['cp10_training_inputs.validate','protocol.validate']
    if assets is not None:
        if c.get('gap_composition_authority')=='gap-composition-v1':
            from mind2act_eval.gap_composition_inputs import validate as validate_gap
            validate_gap(spec,assets);calls.append('gap_composition_inputs.validate')
        if t in ('CP01','CP02') and spec.config.get('layout_split')=='validation':
            helper=importlib.import_module('mind2act_eval.'+t.lower()+'_validation_inputs');helper.validate_assets(spec,assets);calls.append(t.lower()+'_validation_inputs.validate_assets')
        if t in ('PR1','PR2') and spec.config.get('layout_split')=='validation':
            from mind2act_eval.sim.source_assets import verify_source_file
            verify_source_file(assets,'station',c['scene_sha256']);calls.append('build station SHA gate')
        if (t in ('S1','M1','P1','E1') or (t=='F1' and spec.instance_id.startswith('VAL-F1-gap-'))) and spec.config.get('layout_split')=='validation':
            module('validation_inputs').validate(spec,assets);calls.append('validation_inputs.validate')
        if t=='PR3' and spec.config.get('layout_split')=='validation':
            from mind2act_eval.sim.source_assets import verify_source_file
            verify_source_file(assets,'scene',c['scene_sha256']);calls.append('build scene SHA gate')
        if t=='PR4':module('source_admission').validate_case_source(spec);calls.append('source_admission.validate_case_source')
        if t in ('DR3','DR4','PR4','PR6'):
            from mind2act_eval.sim.source_assets import verify_source_file
            verify_source_file(assets,'station',c['scene_sha256']);calls.append('build station SHA gate')
        if t in ('CP06','CP08','CP09'):module('instances').validate_assets(spec,assets);calls.append('instances.validate_assets')
        elif t=='CP05':module('scene').validate_assets(spec,assets);calls.append('scene.validate_assets')
        elif t=='CP07':module('binding').load(spec,assets);calls.append('binding.load')
        elif t=='CP03':module('instances').verify_assets(assets,to_json(spec));calls.append('instances.verify_assets')
        elif t=='CP10':module('scene').validate_binding(spec,assets);calls.append('scene.validate_binding')
        elif t=='F2':module('scene').validate(spec,SimpleNamespace(assets=assets));calls.append('scene.validate')
        elif t=='S2':module('scene').validate(spec,assets);calls.append('scene.validate')
        elif t=='P2':module('binding').validate(spec,SimpleNamespace(assets=assets));calls.append('binding.validate')
        elif t=='M2':module('binding').load(spec,assets);calls.append('binding.load')
        elif t=='PR5':module('scene').checked_inputs(spec,assets);calls.append('scene.checked_inputs')
        elif t=='DR1':module('scene').validate_station_source(spec,assets);calls.append('scene.validate_station_source')
        elif t=='P1':
            m=json.loads(assets.resolve('manifest').read_text());from mind2act_eval.tasks.P1.mechanism.orders import OrderClock
            OrderClock(m['case']['orders']);calls.append('mechanism.orders.OrderClock')
        elif t=='E1':
            m=json.loads(assets.resolve('manifest').read_text());assert m['case']['difficulty'].lower()==spec.difficulty;calls.append('original scene.build prepared-manifest difficulty check')
        elif t=='S1':
            m=json.loads(assets.resolve('manifest').read_text());assert len(m['tables'])==4;calls.append('original scene.build four-table check')
    return calls

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scenes',type=Path,required=True);p.add_argument('--proof-root',type=Path);p.add_argument('--parameters-only',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
    from mind2act_eval.package_paths import code_root
    code=code_root()
    from mind2act_eval.contracts import InstanceSpec,to_json
    from mind2act_eval.public_layouts import validate
    from mind2act_eval.sim import LocalAssets,source_assets
    from mind2act_eval.runtime_release import validate_release_inputs
    scenes=a.scenes.resolve();results=[];errors=[];seen=set();first=None
    if not a.parameters_only:
        from pxr import Usd
        from mind2act_eval.robot import verify_asset,PROFILE
    for item in json.load(open(scenes)):
        path=(scenes.parent/item['spec']).resolve();spec=InstanceSpec.from_dict(json.load(open(path)));key=(spec.task_id,spec.difficulty,spec.instance_id);assert key not in seen;seen.add(key)
        try:
            validate(spec);assets=None;source_reads=0
            if not a.parameters_only:
                root=(scenes.parent/item['asset_root']).resolve();assets=LocalAssets(root,spec.assets,spec=spec);proof=a.proof_root or next((q for q in [root/'proofs',root/'source-projection-proofs'] if q.exists()),None)
                if proof:os.environ['MIND2ACT_SOURCE_PROJECTION_PROOF_ROOT']=str(proof)
                args=SimpleNamespace(asset_root=str(root),spec=str(path),h5=None,replay_metadata=None,adapter_config=None,variation_manifest=None,policy_kind=None);validate_release_inputs(spec,args);verify_asset(assets.resolve('x5'),PROFILE['asset_sha256'])
                for row in source_assets._catalogue()['approvals']:
                    if row['identity']!=source_assets._identity(spec):continue
                    roles=[]
                    for role in spec.assets:
                        f=assets.resolve(role)
                        if f.is_file() and any(parent.name==row['asset_name'] and str(f.relative_to(parent))==row['runtime_path'] for parent in f.parents):roles.append(role)
                    assert roles,('approval runtime role missing',key,row['runtime_path'])
                    for role in roles:
                        source_assets.verify_source_file(assets,role,row['source_sha256'])
                        if 'usd' in source_assets._restoration(row):source_assets.source_layer(assets,role,row['source_sha256'])
                        elif 'json' in source_assets._restoration(row):source_assets.read_source_bytes(assets,role,row['source_sha256'])
                        source_reads+=1
            calls=task_checks(spec,assets);results.append(dict(task=spec.task_id,difficulty=spec.difficulty,instance_id=spec.instance_id,full_spec_authority=True,original_task_CPU_calls=calls,source_role_reads=source_reads,asset_checks_done=not a.parameters_only,native_tested=False));first=first or spec
        except Exception as exc:errors.append(dict(task=spec.task_id,difficulty=spec.difficulty,instance_id=spec.instance_id,error=type(exc).__name__+': '+str(exc)))
    negatives=[]
    if first:
        for name in ['unknown_identity','changed_seed','changed_condition','random','unknown_task']:
            d=to_json(first)
            if name=='unknown_identity':d['instance_id']='not-public-layout'
            elif name=='changed_seed':d['seed']+=1
            elif name=='changed_condition':d['config']['not_a_runtime_parameter']='changed'
            elif name=='random':d.setdefault('variation',{})['mode']='random'
            else:d['task_id']='UNKNOWN'
            try:validate(InstanceSpec.from_dict(d))
            except ValueError:negatives.append(dict(kind=name,rejected=True))
            else:raise AssertionError('public finite authority accepted '+name)
    report=dict(instances_requested=len(seen),passed=len(results),errors=errors,instances=results,negative_admission=negatives,parameters_only=a.parameters_only,unpublished_candidate=not (code/'SOURCE_COMMIT').exists(),upstream_commit=(code/'UPSTREAM_COMMIT').read_text().strip(),gpu_started=False,native_tested=0)
    if a.out:a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['instances','errors']})+('\n'+json.dumps(errors[:10]) if errors else ''));return bool(errors)

if __name__=='__main__':raise SystemExit(main())
