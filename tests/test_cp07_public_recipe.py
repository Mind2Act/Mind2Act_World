"""Source-qualified public counts and actual CPU policy/wire integration."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace as NS

import pytest

from mind2act_eval import original_test_inputs as source
from mind2act_eval.adapters.agent_input import cp07
from mind2act_eval.adapters.agent_input.templates import VERSION, GOALS, CP07_TEST, render_goal
from mind2act_eval.adapters.agent_input.recipes import CATALOG
from mind2act_eval.adapters.coding_payload import require_english
from mind2act_eval.adapters.coding_agent_common_input import CommonInputIntegration, create
from mind2act_eval.contracts import InstanceSpec, to_json
from common_input_fixture import sdk, Client, response, make_policy, observation, delivered_instruction


@pytest.fixture
def selected(tmp_path, monkeypatch):
    slot='CP07_medium_clean_001'
    row=dict(task_id='CP07',difficulty='medium',seed=25002,logical_slot=slot,
        native_case_id='source-native',source_configuration_sha256='a'*64,source_index={'sha256':'b'*64})
    raw=json.dumps({'records':[row]+[{'logical_slot':f'other-{i}'} for i in range(2249)]}).encode()
    monkeypatch.setattr(source,'REQUEST_SHA256',source.digest(raw))
    path=tmp_path/'public-request-index.json';path.write_bytes(raw)
    identity=dict(schema=source.SCHEMA,instance_set='test50-r5',split='test',logical_slot=slot,
        native_case_id=row['native_case_id'],source_seed=25002,source_configuration_sha256='a'*64,
        source_index_sha256='b'*64,source_request_sha256=source.REQUEST_SHA256,snapshot_sha256=source.SNAPSHOT_SHA256)
    spec=InstanceSpec('CP07',slot,'medium',cp07.TEST_SOURCE,'v2','fixture','v1',seed=25002,
        config={'cp07_mode':'source_bound_test','test_source':identity})
    opened=[]
    def resolve(role):
        opened.append(role)
        if role!='test_source_request':raise AssertionError('Private source or reference requested')
        return path
    return spec,NS(resolve=resolve),opened,path


def test_projection_authenticates_source_index_and_exposes_only_public_counts(selected):
    spec,assets,opened,_=selected
    result=cp07.project_parameters(spec,assets)
    assert result=={'cp07_recipe_profile':{'schema':'cp07.public-recipe.v1',
        'cohort':'original-test-eight','required_pieces':8,'stock_pieces':11,'source_scoring_steps':10,
        'score_kind':'binary','scoring_version':'mind2act-final-scoring-20261105'}}
    assert opened==['test_source_request']
    assert 'source-native' not in json.dumps(result) and 'a'*64 not in json.dumps(result)


@pytest.mark.parametrize('mutation',['unknown_case','tier','seed','native','config_hash','index_hash',
    'request_hash','snapshot_hash','split','source_id','mode','request_bytes'])
def test_unknown_or_mixed_source_never_gets_a_recipe(selected,mutation):
    spec,assets,_,path=selected;config=to_json(spec.config)
    if mutation=='unknown_case':
        spec=replace(spec,instance_id='CP07_medium_clean_999');config['test_source']['logical_slot']=spec.instance_id
    elif mutation=='tier':spec=replace(spec,difficulty='hard')
    elif mutation=='seed':spec=replace(spec,seed=25003)
    elif mutation=='source_id':spec=replace(spec,source_id=cp07.TRAINING_SOURCE)
    elif mutation=='mode':config['cp07_mode']='diagnostic_training'
    elif mutation=='request_bytes':path.write_bytes(b'{}')
    else:
        key={'native':'native_case_id','config_hash':'source_configuration_sha256',
             'index_hash':'source_index_sha256','request_hash':'source_request_sha256',
             'snapshot_hash':'snapshot_sha256','split':'split'}[mutation]
        config['test_source'][key]='changed'
    with pytest.raises(ValueError):cp07.project_parameters(replace(spec,config=config),assets)


@pytest.mark.parametrize('tier,stock', [('easy',8),('medium',11),('hard',14)])
def test_test_recipe_has_final_arrival_score_and_authoritative_goal(tier,stock):
    parameters={'cp07_recipe_profile':cp07.public_profile('original-test-eight',tier)}
    text=render_goal(dict(task='CP07',difficulty=tier,environment_parameters=parameters,input_protocol=cp07.TEST_PROTOCOL, instruction=delivered_instruction('CP07',tier)))
    assert '| 100 | The car reaches the opposite bank. |' in text and '| 0 | The car does not reach the opposite bank. |' in text
    assert text.partition('TASK RECIPE:')[0]=='Goal: '+delivered_instruction('CP07',tier)+'\n\n'
    assert CP07_TEST['profiles'][tier]==dict(required_pieces=8,stock_pieces=stock,scoring_steps=10)
    assert '20 × k /' not in text
    require_english(text)


@pytest.mark.parametrize('tier,required,stock', [('easy',8,8),('medium',13,18),('hard',13,28)])
def test_training_stays_separate_and_opens_no_source_or_reference(tier,required,stock):
    spec=InstanceSpec('CP07','training',tier,cp07.TRAINING_SOURCE,'v2','fixture','v1',
        config={'cp07_mode':'diagnostic_training'})
    parameters=cp07.project_parameters(spec,NS(resolve=lambda role:pytest.fail('Training projection opened assets')))
    profile=parameters['cp07_recipe_profile']
    assert (profile['required_pieces'],profile['stock_pieces'],profile['source_scoring_steps'])==(required,stock,required+2)
    rendered=render_goal(dict(task='CP07',difficulty=tier,environment_parameters=parameters,instruction=delivered_instruction('CP07',tier)))
    assert rendered.partition('TASK RECIPE:')[0]=='Goal: '+delivered_instruction('CP07',tier)+'\n\n'
    assert '| 100 | The car reaches the opposite bank. |' in rendered and '| 0 | The car does not reach the opposite bank. |' in rendered


@pytest.mark.parametrize('mutation',['missing','unknown_cohort','extra_private_key','wrong_stock','float_count','old_protocol'])
def test_render_rejects_unqualified_or_self_asserted_profiles(mutation):
    parameters={'cp07_recipe_profile':cp07.public_profile('original-test-eight','hard')};protocol=cp07.TEST_PROTOCOL
    if mutation=='missing':parameters={}
    elif mutation=='old_protocol':protocol=VERSION
    elif mutation=='unknown_cohort':parameters['cp07_recipe_profile']['cohort']='anything'
    elif mutation=='extra_private_key':parameters['cp07_recipe_profile']['solution']='private-canary'
    elif mutation=='wrong_stock':parameters['cp07_recipe_profile']['stock_pieces']=28
    else:parameters['cp07_recipe_profile']['required_pieces']=8.0
    with pytest.raises(ValueError):render_goal(dict(task='CP07',difficulty='hard',environment_parameters=parameters,input_protocol=protocol,instruction=delivered_instruction('CP07','hard')))


def test_binding_refuses_configured_training_profile_for_selected_test(selected):
    spec,assets,_,_=selected
    integration=CommonInputIntegration.__new__(CommonInputIntegration)
    integration.config=dict(task_id='CP07',difficulty='medium',input_protocol=cp07.TEST_PROTOCOL,
        public_parameters={'cp07_recipe_profile':cp07.public_profile('diagnostic-training','medium')})
    integration.policy=NS(input_config={})
    with pytest.raises(ValueError,match='differ'):integration.bind_public_task('CP07','medium',cp07.project_parameters(spec,assets))


def test_cp07_cannot_observe_before_selected_source_binding():
    integration=CommonInputIntegration.__new__(CommonInputIntegration)
    integration.config={'task_id':'CP07'};integration._public_task_bound=False
    with pytest.raises(ValueError,match='selected source'):integration.observe(None)


def test_metadata_records_only_detached_validated_public_profile(monkeypatch,selected):
    from mind2act_eval.adapters.coding_agent_eef import EefIntegration
    spec,assets,_,_=selected
    integration=CommonInputIntegration.__new__(CommonInputIntegration)
    integration.config=dict(task_id='CP07',difficulty='medium',input_protocol=cp07.TEST_PROTOCOL,public_parameters={})
    integration.policy=NS(input_config={},profile={});integration.physics_ticks=100
    parameters=cp07.project_parameters(spec,assets)
    integration.bind_public_task('CP07','medium',parameters)
    parameters['cp07_recipe_profile']['private-canary']='must not be copied after binding'
    monkeypatch.setattr(EefIntegration,'result_metadata',lambda self:{})
    metadata=integration.result_metadata()
    assert metadata['input_protocol']==cp07.TEST_PROTOCOL
    assert metadata['task_recipe_profile']==cp07.public_profile('original-test-eight','medium')
    assert 'private-canary' not in json.dumps(metadata)


def test_audited_client_accepts_and_archives_cp07_explicit_version(tmp_path):
    from mind2act_eval.adapters.coding_agent_client import LoggedNativeClient
    from test_coding_agent_client import ledger
    client=LoggedNativeClient(NS(api_key=''),ledger=ledger(tmp_path,1),total_limit=1,
        episode='CPU-only',episode_limit=1,attempt_dir=tmp_path/'attempts')
    client.enable_input_protocol(cp07.TEST_PROTOCOL)
    client._capture_input({'input':[]},{'attempt':1})
    archive=json.loads((tmp_path/'attempts/attempt-00001.input.json').read_text())
    assert archive['input_protocol']==cp07.TEST_PROTOCOL and archive['credentials_included'] is False


def test_cp07_protocol_is_not_enabled_for_another_task():
    fields={'source_root','urdf','mounts','physics_ticks','camera_map','action_spec','policy_env',
        'client_factory','client_config','planner_factory','planner_config','input_protocol',
        'task_id','difficulty','public_parameters','table_surface_z_m'}
    config={k:None for k in fields};config.update(task_id='CP06',input_protocol=cp07.TEST_PROTOCOL)
    with pytest.raises(ValueError,match='Explicit'):create(config)


@pytest.mark.parametrize('tier', ['easy','medium','hard'])
@pytest.mark.parametrize('cohort', ['original-test-eight','diagnostic-training'])
def test_actual_policy_and_native_wire_preserve_selected_recipe(sdk,tier,cohort):
    from XPolicyLab.utils.openai_responses import ReasoningReplayStore
    parameters={'cp07_recipe_profile':cp07.public_profile(cohort,tier)}
    protocol=cp07.TEST_PROTOCOL if cohort=='original-test-eight' else VERSION
    client=Client([response('wait_steps',{'steps':1})])
    policy,budget=make_policy(sdk,client,task='CP07',tier=tier,public_parameters=parameters,input_protocol=protocol)
    policy.act(observation(sdk,task='CP07',tier=tier))
    native=sdk[0].AzureAgentClient.__new__(sdk[0].AzureAgentClient)
    native.model='offline-fixture';native.reasoning_effort='medium';native.session_id=None;native.reasoning_store=ReasoningReplayStore()
    wire=native._responses_request(client.sent[0][0],client.sent[0][1])
    require_english(wire)
    goal=client.sent[0][0][1]['content']
    assert goal.partition('TASK RECIPE:')[0]=='Goal: '+delivered_instruction('CP07',tier)+'\n\n'
    assert '| 100 | The car reaches the opposite bank. |' in goal and '| 0 | The car does not reach the opposite bank. |' in goal
    assert policy.last_input['protocol']==protocol and budget.used==1


def test_runner_projects_selected_cp07_before_adapter_action(selected,tmp_path):
    from test_runner import fixture
    from mind2act_eval.adapters.demo import DemoAdapter
    from mind2act_eval.runner import EpisodeLimits,run_episode
    spec,assets,_,_=selected
    _,task,ctx,_=fixture(tmp_path);ctx=replace(ctx,assets=assets)
    class Adapter(DemoAdapter):
        def bind_public_task(self,task_id,difficulty,parameters):self.bound=(task_id,difficulty,parameters)
    adapter=Adapter();run_episode(spec,task,ctx,adapter,limits=EpisodeLimits(4,10))
    assert adapter.bound==('CP07','medium',cp07.project_parameters(spec,assets))
