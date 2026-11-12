"""Build a data-only overlay payload for a later immutable release generation."""
from collections import Counter
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil

from mind2act_eval.contracts import InstanceSpec, VariationSpec, to_json
from mind2act_eval.sim import LocalAssets
from . import base_spec_digest, clean_spec, digest, file_digest, resolve
from .layouts import read_layout

TASKS = ('E2','S2','P2','F2','PR4','PR5','PR6','DR3','DR4','CP06','CP07','CP08','CP09','CP10','M2')
PENDING = frozenset(('CP09','DR4'))
COORDINATES = dict(settled_frame='world',units='m',quaternion='wxyz')


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(to_json(value),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def _link(source, target, expected):
    if file_digest(source)!=expected:
        raise ValueError('source input SHA changed: '+str(source))
    if target.exists():
        if file_digest(target)!=expected:raise ValueError('payload path collision: '+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copyfile(source,target)


def build(clean_root, inventory, pool_root, output, *, allow_missing=False):
    """Retain all source slots and pair each overlay with an unchanged clean row.

    Logical layout slots are overlay identities, never substituted instance
    seeds or claimed fixed50 clean cases. This writes no RELEASE descriptor.
    """
    clean_root,pool_root,output=map(lambda p:Path(p).resolve(),(clean_root,pool_root,output))
    release=_read(clean_root/'RELEASE.json')
    pool=_read(pool_root/'variation-pool.json')
    records=inventory['layouts']
    counts=Counter((r['task_id'],r['difficulty']) for r in records)
    delivered={r['task_id'] for r in records}
    if delivered-set(TASKS) or delivered&PENDING:raise ValueError('unexpected delivered task set')
    for task in delivered:
        if any(counts[task,tier]!=50 for tier in ('easy','medium','hard')):
            raise ValueError('delivery needs exactly 50 final layouts in every tier: '+task)
    missing=set(TASKS)-PENDING-delivered
    if missing and not allow_missing:raise ValueError('final delivery input missing: '+','.join(sorted(missing)))
    output.mkdir(parents=True,exist_ok=False);files=output/'files';files.mkdir()
    manifest=dict(schema='mind2act_eval.variation_manifest.v1',layouts={},assets=pool['assets'],
                  mdl_search_paths=pool.get('mdl_search_paths',[]),mdl_files=pool.get('mdl_files',[]))
    for pin in manifest['assets'].values():
        for resource in [pin,*pin['dependencies']]:
            _link(pool_root/resource['path'],files/resource['path'],resource['sha256'])
    for resource in manifest['mdl_files']:
        _link(pool_root/resource['path'],files/resource['path'],resource['sha256'])
    clean_rows={(r['task_id'],r['difficulty']):r for r in release['instances']}
    if len(clean_rows)!=45:raise ValueError('requires the exact 45 clean replay rows')
    bases={};prepared=[];slots=set();missing_cards=set()
    for record in sorted(records,key=lambda r:r['logical_slot']):
        task,tier,slot=(record[k] for k in ('task_id','difficulty','logical_slot'))
        if slot in slots or Path(slot).name!=slot or '\\' in slot:raise ValueError('duplicate/unsafe layout slot')
        slots.add(slot);cell=task+'-'+tier;row=clean_rows[task,tier]
        if cell not in bases:
            path=clean_root/row['spec']['path']
            if file_digest(path)!=row['spec']['sha256']:raise ValueError('clean spec changed')
            clean=InstanceSpec.from_dict(_read(path))
            if clean.variation.mode!='clean' or any(getattr(clean,k)!=row[k] for k in ('task_id','instance_id','difficulty','seed')):
                raise ValueError('clean row identity differs')
            bases[cell]=clean
        clean=bases[cell];source=Path(record['path'])
        if file_digest(source)!=record['sha256']:raise ValueError('archived layout changed')
        raw=_read(source);layout=read_layout(raw,COORDINATES)
        if (layout.task,layout.tier,layout.slot)!=(task,tier,slot):raise ValueError('archived layout identity differs')
        keys={o.asset_key for o in layout.objects}|{layout.ground}
        if layout.appearance['table_enabled']:keys.add(layout.appearance['material']['id'])
        if keys-set(manifest['assets']):raise ValueError('layout resources absent from pinned pool')
        relative='random-layouts/'+task+'/'+tier+'/'+slot+'.json'
        _link(source,files/relative,record['sha256'])
        excluded=record.get('forbidden_aabbs',[])
        if record.get('card_path'):
            _link(Path(record['card_path']),files/('random-layouts/'+task+'/'+tier+'/clutter.yml'),raw['card_sha256'])
        else:missing_cards.add((task,tier))
        thresholds=raw['settle']['thresholds']
        binding=dict(path=relative,base_spec_sha256=base_spec_digest(clean),task_id=task,
            instance_id=clean.instance_id,difficulty=tier,seed=clean.seed,layout_task=task,layout_slot=slot,
            coordinates=COORDINATES,generator_version=record['generator_version'],baseline_sha256=digest(clean.config),
            settled_receipt_sha256=digest(layout.provenance['settle']),forbidden_aabbs=excluded,
            exclusion_receipt_sha256=digest(excluded),source_card_sha256=raw['card_sha256'],
            source_public_config_sha256=raw['public_config_sha256'],
            exclusion_source='delivered_card' if record.get('card_path') else 'card_not_in_compact_delivery; advisory_zones_unavailable',
            initialization_pose_tolerance_m=thresholds['window_translation_m'],
            initialization_pose_tolerance_rad=math.radians(thresholds['window_angle_deg']),
            tolerance_source='archived layout.settle.thresholds')
        manifest['layouts'][slot]=binding
        prepared.append((slot,cell,clean,row,record))
    _write(files/'variation-manifest.json',manifest)
    manifest_ref=dict(path='variation-manifest.json',sha256=file_digest(files/'variation-manifest.json'))
    index=dict(schema='mind2act_eval.random_review.v1',variation_manifest=manifest_ref,
        tasks={t:'READY' if t in delivered else 'PENDING_DELIVERY' if t in PENDING else 'FINAL_INPUT_MISSING' for t in TASKS},
        defaults={},layouts={},pairing='exact clean RELEASE replay instance; no seed/slot synthesis')
    plans=[]
    for slot,cell,clean,row,record in prepared:
        random=replace(clean,variation=VariationSpec('random',slot,record['sha256'],base_spec_digest(clean),manifest_ref['sha256']))
        assert to_json(clean_spec(random))==to_json(clean)
        relative='random-specs/'+clean.task_id+'/'+clean.difficulty+'/'+slot+'.json'
        _write(files/relative,random)
        index['defaults'].setdefault(cell,slot)
        index['layouts'][slot]=dict(task_id=clean.task_id,difficulty=clean.difficulty,instance_id=clean.instance_id,
            seed=clean.seed,base_spec_sha256=base_spec_digest(clean),logical_slot=slot,
            spec=dict(path=relative,sha256=file_digest(files/relative)),clean_spec=dict(row['spec']))
        # Every layout is parsed and every pool byte was checked above. Resolve
        # each default against the real adapter once per clean cell; repeating
        # the same multi-GB pool hashes for all 50 overlays adds no coverage.
        if index['defaults'][cell]==slot:
            plan=resolve(random,LocalAssets(files,dict(variation_manifest='variation-manifest.json')))
            plans.append(dict(slot=slot,plan_sha256=plan.sha256))
    _write(files/'random-specs/index.json',index)
    additions=dict(random=dict(path='random-specs/index.json',sha256=file_digest(files/'random-specs/index.json')),
                   variation_manifest=manifest_ref)
    _write(output/'release-additions.json',additions)
    receipt=dict(schema='mind2act_eval.new15_random_payload.v1',parsed=len(records),resolved=len(plans),
        clean_release_sha256=file_digest(clean_root/'RELEASE.json'),clean_specs_unchanged=True,
        tasks=index['tasks'],missing_advisory_card_bytes=[list(v) for v in sorted(missing_cards)],
        resources=len(manifest['assets']),mdl_files=len(manifest['mdl_files']),
        native_completed=0,source_files_modified=False,published=False,plans=plans,
        files=[dict(path=str(p.relative_to(files)),sha256=file_digest(p),bytes=p.stat().st_size)
               for p in sorted(files.rglob('*')) if p.is_file()])
    _write(output/'payload-receipt.json',receipt)
    return receipt


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean-root',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--pool-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-missing-deliveries',action='store_true')
    args=parser.parse_args()
    receipt=build(args.clean_root,_read(args.inventory),args.pool_root,args.output,
                  allow_missing=args.allow_missing_deliveries)
    print(json.dumps({k:receipt[k] for k in ('parsed','resolved','tasks','resources','native_completed','published')}))


if __name__=='__main__':
    main()
