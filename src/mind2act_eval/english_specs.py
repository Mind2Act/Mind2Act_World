"""Exact English derivatives of the reviewed runnable source specs.

Source prose and its hashes remain private provenance. The code-owned catalogue
binds the complete clean source and derivative, not a caller-selected field mask.
Random overlays must separately bind the new clean digest and manifest bytes.
"""
from dataclasses import replace
from functools import lru_cache
from importlib import resources
import json
import re

from .contracts import to_json
from .scene_probe import spec_digest
from .variations import clean_spec, digest

BASELINE_SHA256 = '0cfee6abd002c1892b13d7c63503ea6e545fc28a92258f66a8260f25809f06de'
EXTRA_SOURCES_SHA256 = '95b6f215286854e8bfb8373332d87d1bc3ef4585da29293c8de8087b8d9af5cb'
CP09_EXTRA_SOURCES_SHA256 = 'f0cb15597dff502ea7b97f3ab2b0f759780c74866aa9cffbfe23cc3e2f9d22fe'
CJK = re.compile('[\u2e80-\u9fff\uf900-\ufaff\U00020000-\U000323af]')


def cjk_paths(value, path=''):
    """Inspect decoded JSON keys and values, including supplementary CJK."""
    if isinstance(value, str):
        return [path] if CJK.search(value) else []
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            pointer = path + '/' + key.replace('~', '~0').replace('/', '~1')
            if CJK.search(key):
                result.append(pointer + ' (key)')
            result.extend(cjk_paths(item, pointer))
        return result
    if isinstance(value, (list, tuple)):
        return [hit for i, item in enumerate(value) for hit in cjk_paths(item, path + '/' + str(i))]
    return []


def _project_fields(source):
    """Prospective text only; admission additionally requires the frozen hashes."""
    config = to_json(source.config)
    if source.task_id in ('CP01','CP02','CP03','CP04','CP05','DR1','DR2',
                           'E1','F1','M1','P1','PR3','S1'):
        from .tasks._instructions import public_instruction
        config['instruction'] = public_instruction(source)
    elif source.task_id in ('PR1', 'PR2'):
        task = source.task_id
        layout = json.loads(resources.files('mind2act_eval.tasks.' + task)
                            .joinpath('config/layout.json').read_text())
        target = config['target_ohm' if task == 'PR1' else 'target_volt']
        config['instruction'] = layout['instruction'].replace('{TARGET}', f'{target:g}')
    if source.task_id == 'CP09':
        from .robot.task_collision_overlay import english_geometry_decision
        config['geometry_decision'] = english_geometry_decision(config['geometry_decision'])
    result = replace(source, config=config)
    if cjk_paths(to_json(result)):
        raise ValueError('English spec contains an unapproved CJK key or value')
    return result


@lru_cache(maxsize=1)
def _catalogue():
    data = json.loads(resources.files('mind2act_eval.tasks._instructions')
                      .joinpath('config/spec-projections.json').read_text())
    if (data.get('schema') != 'mind2act_eval.english_spec_authority.v1'
            or data.get('source_baseline_sha256') != BASELINE_SHA256
            or data.get('source_extra_sha256') != EXTRA_SOURCES_SHA256
            or data.get('source_cp09_extra_sha256') != CP09_EXTRA_SOURCES_SHA256
            or data.get('fields') != ['/config/instruction', '/config/geometry_decision']
            or sum(len(rows) for rows in data.get('cases', {}).values()) != 4598):
        raise ValueError('English spec authority catalogue differs')
    runtime_sources = {}
    for case, rows in data['cases'].items():
        if not isinstance(case, str) or not case or not isinstance(rows, list) or not rows:
            raise ValueError('English spec authority requires exact case identities')
        seen = set()
        for row in rows:
            if (not isinstance(row, list) or len(row) != 3
                    or any(not isinstance(pin, str) or re.fullmatch('[0-9a-f]{64}', pin) is None for pin in row)
                    or row[0] in seen or runtime_sources.get(row[1], row[2]) != row[2]):
                raise ValueError('English spec authority requires unambiguous source/runtime/config digests')
            seen.add(row[0])
            runtime_sources[row[1]] = row[2]
    return data['cases']


def project_spec(source):
    """Compile one exact reviewed clean spec; arbitrary source hashes cannot opt in."""
    source_digest = spec_digest(source)
    row = next((r for r in _catalogue().get(source.instance_id, []) if r[0] == source_digest), None)
    if source.variation.mode != 'clean' or row is None:
        raise ValueError('English projection requires the exact code-owned clean source spec')
    result = _project_fields(source)
    if spec_digest(result) != row[1] or digest(source.config) != row[2]:
        raise ValueError('English projection differs from its frozen text/source authority')
    return result


def canonical_source_view(runtime, source):
    """Return the authenticated original only for private source checks, never model input."""
    if to_json(project_spec(source)) != to_json(runtime):
        raise ValueError('English derivative changed fields outside its exact text authority')
    return source


def approved_source_config_sha256(spec):
    """Bridge resource approval identity only for an exact complete English derivative."""
    if spec is None:
        return None
    rows = _catalogue().get(spec.instance_id, [])
    if rows:
        runtime_digest = spec_digest(clean_spec(spec))
        row = next((r for r in rows if r[1] == runtime_digest), None)
        if row is not None:
            return row[2]
    return None


def rebind_manifest(manifest, sources):
    """Change only clean/config digest references; keep layout and physical evidence."""
    result = to_json(manifest)
    if result.get('schema') != 'mind2act_eval.variation_manifest.v1':
        raise ValueError('English projection requires a frozen variation manifest')
    for slot, binding in result['layouts'].items():
        source = sources.get(binding['instance_id'])
        if source is None:
            raise ValueError('variation manifest has no reviewed English source: ' + slot)
        if (any(binding[key] != getattr(source, key) for key in ('task_id','instance_id','difficulty','seed'))
                or binding['base_spec_sha256'] != spec_digest(source)
                or binding['baseline_sha256'] != digest(source.config)):
            raise ValueError('variation manifest differs from its exact original source')
        runtime = project_spec(source)
        binding['base_spec_sha256'] = spec_digest(runtime)
        binding['baseline_sha256'] = digest(runtime.config)
    return result


def project_random(source, runtime_clean, *, source_clean, manifest_sha256):
    """Retain every original random field except its paired clean/manifest digests."""
    canonical_source_view(runtime_clean, source_clean)
    if (source.variation.mode != 'random' or to_json(clean_spec(source)) != to_json(source_clean)
            or source.variation.base_spec_sha256 != spec_digest(source_clean)):
        raise ValueError('random source must preserve every original clean spec field')
    variation = replace(source.variation, base_spec_sha256=spec_digest(runtime_clean),
                        asset_manifest_sha256=manifest_sha256)
    return replace(runtime_clean, variation=variation)
