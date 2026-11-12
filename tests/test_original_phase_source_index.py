"""Original DR3 phase cases keep seed zero without disabling seed guards."""
import hashlib
import json

import pytest

from mind2act_eval.release_contract import ReleaseContractError
from mind2act_eval.task_shards import load_source_index


SNAPSHOT = '917fa443a3ac000e4050837ebf08a84b09052e5707f72d17146d4fc799545103'


def index(task='DR3'):
    rows = []
    for tier in ('easy', 'medium', 'hard'):
        for i in range(50):
            case = f'{task}_{tier}_clean_{i:03}'
            rows.append(dict(case_id=case, logical_slot=case, task_id=task, difficulty=tier,
                native_case_id=f'FR3-alignment-v1-{tier}-p{i:04}', seed=0,
                source_document_id=f'FR3-alignment-v1-{tier}-p{i:04}',
                source_configuration_sha256=hashlib.sha256(case.encode()).hexdigest(),
                coding_agent_selected=i < 5, limits=dict(physics_ticks=1200, wall_s=300, retries=0),
                source_condition=dict(kind='initial_mechanism_phase', phase_ticks=i,
                    source_revision='b6432b408607981059acf36ee668b731e36d70a0', original_formal_eligible=False)))
    return dict(schema='mind2act_eval.test_source_index.v1', instance_set='test50-r5',
        cases_per_cell=50, coding_agent_per_cell=5, tasks=[task],
        source_snapshot_sha256=SNAPSHOT, instances=rows)


def load(tmp_path, data, monkeypatch):
    from mind2act_eval import original_test_inputs
    # Trust one synthetic immutable request at the external authority boundary;
    # production keeps the code-owned original request SHA.
    records=index()['instances'] + [dict(logical_slot=f'other-{i}') for i in range(2100)]
    request=json.dumps(dict(records=records)).encode()
    (tmp_path/'request.json').write_bytes(request)
    pin=hashlib.sha256(request).hexdigest()
    monkeypatch.setattr(original_test_inputs,'REQUEST_SHA256',pin)
    data['source_request']=dict(path='request.json',sha256=pin)
    raw=json.dumps(data).encode();(tmp_path/'index.json').write_bytes(raw)
    return load_source_index(tmp_path, dict(path='index.json', sha256=hashlib.sha256(raw).hexdigest()),
                             tasks=data['tasks'])


def test_original_phase_conditions_preserve_repeated_source_seed(tmp_path,monkeypatch):
    rows=load(tmp_path,index(),monkeypatch)
    assert len(rows)==150 and {r['seed'] for r in rows}=={0}
    assert all(r['source_condition']['original_formal_eligible'] is False for r in rows)


@pytest.mark.parametrize('mutation', ['duplicate_phase','missing_condition','snapshot','negative_phase',
                                    'bool_phase','revision','eligibility','native_alias'])
def test_phase_exception_requires_complete_distinct_original_conditions(tmp_path,monkeypatch,mutation):
    data=index();row=data['instances'][1]
    if mutation=='duplicate_phase':row['source_condition']['phase_ticks']=0
    elif mutation=='missing_condition':row.pop('source_condition')
    elif mutation=='snapshot':data['source_snapshot_sha256']='b'*64
    elif mutation=='negative_phase':row['source_condition']['phase_ticks']=-1
    elif mutation=='bool_phase':row['source_condition']['phase_ticks']=True
    elif mutation=='revision':row['source_condition']['source_revision']='unknown'
    elif mutation=='eligibility':row['source_condition']['original_formal_eligible']='true'
    elif mutation=='native_alias':row['source_document_id']='another-source'
    with pytest.raises(ReleaseContractError):load(tmp_path,data,monkeypatch)


def test_old_task_duplicate_seeds_still_rejected_even_with_phase_fields(tmp_path,monkeypatch):
    with pytest.raises(ReleaseContractError):load(tmp_path,index('PR1'),monkeypatch)
