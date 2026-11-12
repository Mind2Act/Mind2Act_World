"""Explicit ready-case subsets against the unchanged full test campaign."""
from collections import Counter
import json
from types import MappingProxyType

from .instance_catalog import asset_descriptor, digest_field, file_descriptor, INSTANCE_SET, TIERS
from .release_contract import ReleaseContractError


def _seed_identity(root, data, row):
    task, tier, seed = row.get('task_id'), row.get('difficulty'), row.get('seed')
    condition = row.get('source_condition')
    if task != 'DR3':
        return task, tier, seed
    from .original_test_inputs import REQUEST_SHA256, SNAPSHOT_SHA256, request_records
    if condition is None:
        if data.get('source_request') is not None or data.get('source_snapshot_sha256') == SNAPSHOT_SHA256:
            raise ReleaseContractError('source-qualified DR3 phase identity required')
        return task, tier, seed
    descriptor = data.get('source_request')
    try:
        if (data.get('source_snapshot_sha256') != SNAPSHOT_SHA256
                or not isinstance(descriptor, dict) or descriptor.get('sha256') != REQUEST_SHA256):
            raise ValueError('original request/snapshot pin required')
        records = request_records(file_descriptor(root, descriptor, verify=True))
        original = records.get(row.get('logical_slot'))
        if (original is None or any(row.get(k) != original[k] for k in
                ('case_id','task_id','difficulty','seed','logical_slot','native_case_id',
                 'source_configuration_sha256','coding_agent_selected'))
                or not isinstance(condition, dict)
                or set(condition) != {'kind','phase_ticks','source_revision','original_formal_eligible'}
                or condition['kind'] != 'initial_mechanism_phase'
                or type(condition['phase_ticks']) is not int or condition['phase_ticks'] < 0
                or condition['original_formal_eligible'] is not False
                or condition['source_revision'] != 'b6432b408607981059acf36ee668b731e36d70a0'
                or row.get('source_document_id') != row['native_case_id']
                or row['native_case_id'] not in {
                    f"FR3-{family}-v1-{tier}-p{condition['phase_ticks']:04}"
                    for family in ('alignment', 'phase')}):
            raise ValueError('original DR3 phase source identity differs')
    except (KeyError, TypeError, ValueError) as exc:
        raise ReleaseContractError('source-qualified DR3 phase identity required') from exc
    return task, tier, seed, condition['source_revision'], condition['phase_ticks']


def _document(root, descriptor):
    path = file_descriptor(root, descriptor, verify=True)
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError) as exc:
        raise ReleaseContractError('cannot read task-shard document') from exc
    if not isinstance(value, dict):
        raise ReleaseContractError('task-shard document must be an object')
    return value


def load_source_index(root, descriptor, *, tasks):
    data = _document(root, descriptor)
    if (data.get('schema') != 'mind2act_eval.test_source_index.v1'
            or data.get('instance_set') != INSTANCE_SET
            or data.get('cases_per_cell') != 50 or data.get('coding_agent_per_cell') != 5
            or not isinstance(data.get('tasks'), list) or len(data['tasks']) != len(tasks)
            or set(data['tasks']) != set(tasks)):
        raise ReleaseContractError('source index must retain the exact full test task grid')
    rows = data.get('instances')
    if not isinstance(rows, list) or len(rows) != len(tasks)*150:
        raise ReleaseContractError('source index must retain the full test campaign denominator')
    ids, seeds, cells, coding, result = set(), set(), Counter(), Counter(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ReleaseContractError('source index instance must be an object')
        case, task, tier = row.get('case_id'), row.get('task_id'), row.get('difficulty')
        seed = row.get('seed')
        seed_identity = _seed_identity(root, data, row)
        if (not isinstance(case, str) or not case or case in ids
                or task not in tasks or tier not in TIERS or type(seed) is not int or seed < 0
                or seed_identity in seeds or type(row.get('coding_agent_selected')) is not bool
                or any(not isinstance(row.get(k), str) or not row[k] for k in ('logical_slot','native_case_id'))):
            raise ReleaseContractError('source index has an invalid or duplicate case/seed identity')
        digest_field(row.get('source_configuration_sha256'))
        limits = row.get('limits')
        try:
            from .budgets import validate_e2e_fields
            validate_e2e_fields(row)
        except ValueError as exc:
            raise ReleaseContractError(str(exc)) from exc
        if (not isinstance(limits, dict) or set(limits) != {'physics_ticks','wall_s','retries'}
                or type(limits['retries']) is not int or limits['retries'] != 0
                or any(type(limits[k]) is not int or limits[k] <= 0 for k in ('physics_ticks','wall_s'))):
            raise ReleaseContractError('source index requires explicit original budgets and zero retries')
        ids.add(case); seeds.add(seed_identity); cells[task,tier] += 1
        coding[task,tier] += row['coding_agent_selected']
        result.append(MappingProxyType(dict(row)))
    expected = {(t,d) for t in tasks for d in TIERS}
    if set(cells) != expected or set(cells.values()) != {50} or set(coding.values()) != {5}:
        raise ReleaseContractError('source index requires 50 cases and five coding cases per cell')
    return tuple(result)


def load_task_shards(root, descriptor, *, tasks):
    data = _document(root, descriptor)
    states = data.get('tasks')
    if (data.get('schema') != 'mind2act_eval.task_shards.v1'
            or data.get('instance_set') != INSTANCE_SET
            or data.get('expected_case_count') != len(tasks)*150
            or not isinstance(states, dict) or set(states) != set(tasks)):
        raise ReleaseContractError('task shards must declare the full campaign and every task status')
    source = load_source_index(root, data.get('source_index'), tasks=tasks)
    by_id = {r['case_id']:r for r in source}
    ready, statuses = [], {}
    for task, state in states.items():
        if not isinstance(state, dict) or state.get('status') not in {'ready','partial','pending','hold'}:
            raise ReleaseContractError('task shard requires explicit ready/partial/pending/hold status')
        statuses[task] = state['status']
        if state['status'] not in {'ready','partial'}:
            if state.get('catalog') is not None:
                raise ReleaseContractError('pending/hold task cannot publish runnable cases')
            continue
        rows = load_ready_catalog(root, state.get('catalog'), task=task)
        for row in rows:
            original = by_id.get(row['case_id'])
            keys = ('task_id','difficulty','seed','logical_slot','native_case_id',
                    'source_configuration_sha256','coding_agent_selected','limits',
                    'source_condition','source_document_id','limits_e2e','budget_version_e2e')
            if original is None or any(row.get(k) != original.get(k) for k in keys):
                raise ReleaseContractError('ready task shard differs from its full source index')
        ready.extend(rows)
    return tuple(ready), source, MappingProxyType(statuses)


def load_ready_catalog(root, descriptor, *, task):
    """Parse only published ready rows; the source index owns the full 50/5 grid."""
    data = _document(root, descriptor)
    rows = data.get('instances')
    if (data.get('schema') != 'mind2act_eval.ready_case_catalog.v1'
            or data.get('instance_set') != INSTANCE_SET or data.get('task_id') != task
            or data.get('expected_task_case_count') != 150
            or not isinstance(rows, list) or not 1 <= len(rows) <= 150):
        raise ReleaseContractError('ready catalog must declare a nonempty subset of its planned 150 cases')
    ids, result = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ReleaseContractError('ready catalog row must be an object')
        case = row.get('case_id')
        if (not isinstance(case,str) or not case or case in ids or row.get('task_id') != task
                or row.get('difficulty') not in TIERS or row.get('instance_id',case) != case
                or type(row.get('coding_agent_selected')) is not bool):
            raise ReleaseContractError('ready catalog has an invalid or duplicate case identity')
        ids.add(case)
        try:
            from .budgets import validate_e2e_fields
            validate_e2e_fields(row)
        except ValueError as exc:
            raise ReleaseContractError(str(exc)) from exc
        file_descriptor(root,row.get('spec'))
        digest_field(row['spec'].get('semantic_sha256'))
        assets = row.get('assets')
        if not isinstance(assets,dict) or any(not isinstance(k,str) or not k for k in assets):
            raise ReleaseContractError('ready case requires explicit asset roles')
        for asset in assets.values():
            asset_descriptor(root,asset)
        file_descriptor(root,row.get('asset_closure'))
        metadata = row.get('replay_metadata')
        if metadata is not None:
            file_descriptor(root,metadata)
        result.append(MappingProxyType(dict(row,instance_id=case,replay_metadata=metadata)))
    return tuple(result)
