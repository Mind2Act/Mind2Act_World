"""Opt-in CPU inventory of all150 original CP07 sources, both rendering modes.

Reads only published spec/request metadata. No private binding or raw source is
opened; numerical state, RGB and station geometry are explicit CPU fixtures.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
from collections import Counter

import pytest

from mind2act_eval.adapters.agent_input.cp07 import TEST_PROTOCOL, project_parameters
from mind2act_eval.adapters.coding_payload import require_english
from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.release_contract import load_release
from mind2act_eval.sim import LocalAssets
from common_input_fixture import sdk,Client,response,make_policy,observation,delivered_instruction


def test_all150_original_sources_both_modes_use_actual_public_recipe_and_wire(sdk,monkeypatch):
    root=os.environ.get('MIND2ACT_CP07_RECIPE_RELEASE')
    old_code=os.environ.get('MIND2ACT_CP07_RECIPE_RELEASE_CODE')
    output=os.environ.get('MIND2ACT_CP07_RECIPE_OUTPUT')
    if not root or not old_code or not output:
        pytest.skip('explicit read-only CP07 release/code and fresh diagnostic output required')
    release=load_release(root,code_root=old_code)
    rows=[r for r in release.select_ready_cases() if r['task_id']=='CP07']
    assert len(rows)==150 and Counter(r['difficulty'] for r in rows)=={'easy':50,'medium':50,'hard':50}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    candidate=Path(__file__).resolve().parents[1]
    candidate_commit=(candidate/'SOURCE_COMMIT').read_text().strip()
    assert len(candidate_commit)==40 and all(c in '0123456789abcdef' for c in candidate_commit)
    from XPolicyLab.utils.openai_responses import ReasoningReplayStore
    def blocked(*args,**kwargs):raise AssertionError('Model/API/network access is forbidden')
    monkeypatch.setattr(socket,'socket',blocked);monkeypatch.setattr(socket,'create_connection',blocked)
    records=[]
    for row in rows:
        for mode,field in [('clean','spec'),('random','random_spec')]:
            path=release.asset_root/row[field]['path'];raw=path.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==row[field]['sha256']
            spec=InstanceSpec.from_dict(json.loads(raw));assets=LocalAssets(release.asset_root,spec.assets,spec=spec)
            opened=[]
            class PublicAssets:
                def resolve(self,role):
                    assert role=='test_source_request','Private source/binding/reference was requested'
                    opened.append(role);return assets.resolve(role)
            parameters=project_parameters(spec,PublicAssets())
            assert opened==['test_source_request']
            profile=parameters['cp07_recipe_profile']
            assert profile['required_pieces']==8 and profile['source_scoring_steps']==10
            assert profile['score_kind']=='binary'
            assert profile['stock_pieces']=={'easy':8,'medium':11,'hard':14}[spec.difficulty]
            client=Client([response('wait_steps',{'steps':1})])
            policy,budget=make_policy(sdk,client,task='CP07',tier=spec.difficulty,
                ticks=row['limits']['physics_ticks'],public_parameters=parameters,input_protocol=TEST_PROTOCOL)
            policy.act(observation(sdk,task='CP07',tier=spec.difficulty))
            messages,tools=client.sent[0]
            goal=messages[1]['content']
            assert '100: the car reaches the opposite bank.' in goal
            assert '0: otherwise.' in goal and '20 × k /' not in goal
            native=sdk[0].AzureAgentClient.__new__(sdk[0].AzureAgentClient)
            native.model='offline-fixture';native.reasoning_effort='medium';native.session_id=None
            native.reasoning_store=ReasoningReplayStore()
            wire=native._responses_request(messages,tools)
            payload=dict(messages=messages,tools=tools,responses_payload=wire)
            require_english(payload)
            data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode()
            target=out/f'{mode}-{row["case_id"]}.payload.json';target.write_bytes(data)
            records.append(dict(case_id=row['case_id'],mode=mode,tier=spec.difficulty,
                spec_path=str(path),spec_sha256=row[field]['sha256'],public_profile=profile,
                payload=target.name,payload_sha256=hashlib.sha256(data).hexdigest(),
                cjk=0,decisions=budget.used,input_protocol=policy.last_input['protocol']))
    assert len(records)==300 and all(r['decisions']==1 and r['input_protocol']==TEST_PROTOCOL for r in records)
    report=dict(schema='cp07.source-test-public-recipe-render.v1',input_protocol=TEST_PROTOCOL,
        unique_sources=150,rendered=300,cjk=0,source_code_commit=release.code_commit,
        candidate_code_commit=candidate_commit,
        source_sha256={name:hashlib.sha256((candidate/name).read_bytes()).hexdigest() for name in
            ('src/mind2act_eval/adapters/agent_input/cp07.py',
             'src/mind2act_eval/adapters/agent_input/data/cp07-source-test.json',
             'src/mind2act_eval/adapters/agent_input/templates.py',
             'src/mind2act_eval/adapters/agent_input/policy.py',
             'src/mind2act_eval/adapters/coding_agent_common_input.py',
             'src/mind2act_eval/adapters/coding_agent_client.py',
             'src/mind2act_eval/runner/__init__.py')},
        scope='Actual source-qualified metadata and policy/native-wire builder; CPU state/RGB/station fixtures; no native/provider run.',
        opened_asset_roles=['test_source_request'],private_roles_opened=[],network_requests=0,cases=records)
    data=json.dumps(report,ensure_ascii=False,indent=2).encode();(out/'REPORT.json').write_bytes(data)
    hashes={r['payload']:r['payload_sha256'] for r in records};hashes['REPORT.json']=hashlib.sha256(data).hexdigest()
    (out/'ARTIFACTS.json').write_text(json.dumps(hashes,indent=2)+'\n')
