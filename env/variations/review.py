"""Select a released overlay while retaining the exact clean replay identity."""
import json
from pathlib import Path

from mind2act_eval.contracts import InstanceSpec, to_json
from mind2act_eval.instance_catalog import file_descriptor,relative_path
from mind2act_eval.sim import LocalAssets
from . import base_spec_digest, clean_spec, resolve


class RandomPending(ValueError):
    pass


def _select_ready_random(root, row, *, layout_slot):
    """A fixed50 case selects only its catalog-pinned same-slot overlay."""
    if row.get('random_spec') is None and row.get('variation_manifest') is None:
        raise RandomPending(row['task_id'] + ' fixed50 random 尚未发布')
    case, slot = row['case_id'], row.get('layout_slot')
    if (row.get('instance_id', case) != case or row.get('logical_slot') != case
            or not isinstance(slot, str) or not slot
            or slot != case and slot != row.get('native_case_id')
            or layout_slot is not None and layout_slot != slot):
        raise ValueError('fixed50 random layout slot differs from its source case')
    clean_path = file_descriptor(root, row['spec'], verify=False)
    random_path = file_descriptor(root, row.get('random_spec'), verify=False)
    manifest = file_descriptor(root, row.get('variation_manifest'), verify=False)
    layout = file_descriptor(root, row.get('source_layout'), verify=False)
    clean = InstanceSpec.from_dict(json.loads(clean_path.read_text()))
    random = InstanceSpec.from_dict(json.loads(random_path.read_text()))
    from mind2act_eval.spec_identity import spec_digest
    source = clean.config.get('test_source', {})
    if (clean.source_id == 'cp07-training-layout-v1'
            and clean.config.get('cp07_mode') == 'train_layout_test'):
        from mind2act_eval.tasks.CP07.layout_identity import validate_identity
        validate_identity(clean)
        source = clean.config['fixed50_origin']
    if (clean.instance_id != case or clean.task_id != row['task_id']
            or clean.difficulty != row['difficulty'] or clean.seed != row['seed']
            or clean.variation.mode != 'clean'
            or source.get('instance_set') != 'test50-r5' or source.get('split') != 'test'
            or any(source.get(key) != row.get(key) for key in
                   ('logical_slot', 'native_case_id'))):
        raise ValueError('fixed50 clean spec differs from its catalog source identity')
    if random.variation.mode != 'random' or to_json(clean_spec(random)) != to_json(clean):
        raise ValueError('fixed50 random must preserve every clean spec field')
    if random.variation.layout_id != case:
        raise ValueError('fixed50 random spec differs from its catalog input pins')
    relative = str(manifest.relative_to(root))
    plan = resolve(random, LocalAssets(root, dict(random.assets, variation_manifest=relative)))
    if (plan.layout.slot != slot
            or (manifest.parent / plan.binding['path']).resolve() != layout):
        raise ValueError('fixed50 random plan differs from its original layout slot/pin')
    return random_path, row['random_spec']['sha256'], relative, slot


def select_random(root, release, clean_row, *, layout_slot=None):
    root = Path(root).resolve()
    if 'case_id' in clean_row:
        return _select_ready_random(root, clean_row, layout_slot=layout_slot)
    descriptor = release.get('random')
    if descriptor is None and release.get('asset_name') == 'old15-release-v1':
        manifest = json.loads(relative_path(root,release['manifest']).read_text())
        descriptor = next((dict(path=r['path'],sha256=r['sha256']) for r in manifest['files']
                           if r['path']=='random-specs/index.json'),None)
    if descriptor is None:
        raise RandomPending('random 尚未发布')
    index_path = file_descriptor(root, descriptor, verify=False)
    index = json.loads(index_path.read_text())
    task, tier = clean_row['task_id'], clean_row['difficulty']
    if index.get('schema') == 'mind2act.old45.random_specs.v1':
        row=index['rows'].get(task+'-'+tier)
        slot=row.get('layout_slot') if row else None
        if layout_slot is not None and layout_slot!=slot:
            raise ValueError('unknown released random layout slot: '+str(layout_slot))
    elif index.get('schema') == 'mind2act_eval.random_review.v1':
        status = index['tasks'].get(task)
        if status != 'READY':
            raise RandomPending(task + ' random 待交付' if status == 'PENDING_DELIVERY'
                                else task + ' random 输入未就绪')
        slot = layout_slot or index['defaults'].get(task + '-' + tier)
        row = index['layouts'].get(slot)
    else:
        raise ValueError('unsupported released random review index')
    if row is None:
        raise ValueError('unknown released random layout slot: ' + str(slot))
    if index['schema'] == 'mind2act_eval.random_review.v1':
        if row.get('logical_slot') != slot:
            raise ValueError('random index row differs from selected layout slot')
        layout_id = slot
    else:
        layout_id = row.get('layout_id')
    if any(row.get(key) != clean_row.get(key) for key in ('task_id', 'difficulty', 'instance_id')):
        raise ValueError('random layout belongs to another clean task/tier/instance')
    clean_path = file_descriptor(root, clean_row['spec'], verify=False)
    random_path = file_descriptor(root, row['spec'], verify=False)
    clean = InstanceSpec.from_dict(json.loads(clean_path.read_text()))
    random = InstanceSpec.from_dict(json.loads(random_path.read_text()))
    if random.variation.mode != 'random' or to_json(clean_spec(random)) != to_json(clean):
        raise ValueError('random must preserve every clean spec field')
    if random.variation.layout_id != layout_id:
        raise ValueError('random spec differs from selected layout slot')
    manifest = file_descriptor(root, index['variation_manifest'], verify=False)
    relative = str(manifest.relative_to(root))
    plan = resolve(random, LocalAssets(root, dict(random.assets, variation_manifest=relative)))
    if plan.layout.slot != slot:
        raise ValueError('resolved random layout slot differs from selected slot')
    return random_path, row['spec']['sha256'], relative, slot
