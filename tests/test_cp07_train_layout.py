"""Training-layout public protocol and CPU frozen delivery integration."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from mind2act_eval.adapters.agent_input import cp07
from mind2act_eval.adapters.agent_input.templates import render_goal
from mind2act_eval.adapters.coding_payload import require_english
from mind2act_eval.cli import read_spec
from mind2act_eval.sim import LocalAssets
from mind2act_eval.tasks.CP07.binding import load
from common_input_fixture import sdk, Client, response, make_policy, observation, delivered_instruction


@pytest.mark.parametrize('tier,required,stock',[('easy',8,8),('medium',13,18),('hard',13,28)])
def test_source_counts_and_final_recipe(tier,required,stock):
    profile=cp07.public_profile('training-layout-test',tier)
    assert (profile['required_pieces'],profile['stock_pieces'],profile['source_scoring_steps'])==(required,stock,required+2)
    text=render_goal(dict(task='CP07',difficulty=tier,environment_parameters={'cp07_recipe_profile':profile},input_protocol=cp07.TRAIN_LAYOUT_PROTOCOL,instruction=delivered_instruction('CP07',tier)))
    assert '| 100 | The car reaches the opposite bank. |' in text and '| 0 | The car does not reach the opposite bank. |' in text
    require_english(text)
    with pytest.raises(ValueError):
        cp07.validate_parameters({'cp07_recipe_profile':profile},tier,cp07.TEST_PROTOCOL)


def test_all150_cpu_binding_and_actual_wire(sdk,monkeypatch,tmp_path):
    value=os.environ.get('CP07_TRAIN_LAYOUT_ROOT')
    if not value:pytest.skip('explicit cold CP07 delivery required')
    root=Path(value);paths=sorted(root.glob('test-cases/CP07/*/*/spec.json'))
    assert len(paths)==150
    random_paths=sorted(root.glob('test50-random/CP07/specs/*/*.json'))
    assert len(random_paths)==150
    paths+=random_paths
    import socket
    def blocked(*a,**kw):raise AssertionError('network forbidden during CPU validation')
    monkeypatch.setattr(socket,'socket',blocked)
    records=[]
    for path in paths:
        spec=read_spec(path);assets=LocalAssets(root,spec.assets,spec=spec)
        binding=load(spec,assets)
        assert len(binding['pieces'])=={'easy':8,'medium':18,'hard':28}[spec.difficulty]
        assert {len(s['slots']) for s in binding['solutions']}=={8 if spec.difficulty=='easy' else 13}
        # The public projector can use only the trusted spec/index; no private IO.
        public=cp07.project_parameters(spec,NS(resolve=lambda *a:pytest.fail('private IO')))
        client=Client([response('wait_steps',{'steps':1})])
        policy,budget=make_policy(sdk,client,task='CP07',tier=spec.difficulty,ticks=20000,
                                  public_parameters=public,input_protocol=cp07.TRAIN_LAYOUT_PROTOCOL)
        policy.act(observation(sdk,task='CP07',tier=spec.difficulty));messages,tools=client.sent[0]
        from XPolicyLab.utils.openai_responses import ReasoningReplayStore
        native=sdk[0].AzureAgentClient.__new__(sdk[0].AzureAgentClient)
        native.model='offline-fixture';native.reasoning_effort='medium';native.session_id=None;native.reasoning_store=ReasoningReplayStore()
        wire=native._responses_request(messages,tools)
        require_english(dict(messages=messages,tools=tools,responses_payload=wire))
        assert '| 100 | The car reaches the opposite bank. |' in messages[1]['content']
        assert '| 0 | The car does not reach the opposite bank. |' in messages[1]['content']
        assert budget.used==1 and policy.last_input['protocol']==cp07.TRAIN_LAYOUT_PROTOCOL
        with pytest.raises(ValueError):cp07.project_parameters(replace(spec,seed=spec.seed+1),assets)
        config=dict(spec.config);config['cp07_binding_sha256']='0'*64
        with pytest.raises(ValueError):load(replace(spec,config=config),assets)
        records.append(dict(case_id=spec.instance_id,mode=spec.variation.mode,profile=public,cjk=0))
    report=Path(os.environ.get('CP07_TRAIN_LAYOUT_REPORT',str(tmp_path/'cpu-wire-records.json')))
    report.write_text(json.dumps(records,indent=2))
