"""Offline fixed-denominator inspection and prospective smoke selection.

This reports input inspection separately from resolve/apply/native acceptance.
Callers provide the frozen expected slots, not a count discovered from successes.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from . import file_digest
from .layouts import read_layout


def inspect(paths, expected, *, coordinates):
    rows, seen = [], Counter()
    for path in sorted(map(Path, paths)):
        row = {"path": str(path), "file_sha256": file_digest(path), "cpu_layout_parse": "failed"}
        try:
            data = json.loads(path.read_text())
            key = (data['task'], data['difficulty'], data['logical_slot'])
            row.update(task=key[0], tier=key[1], slot=key[2])
            seen[key] += 1
            layout = read_layout(data, coordinates)
            errors = []
            if key not in expected:
                errors.append('slot absent from frozen expected index')
            else:
                for name in ('native_seed', 'release_seed'):
                    if name in expected[key] and data.get(name) != expected[key][name]:
                        errors.append(name + ' differs from frozen index')
            row.update(cpu_layout_parse='passed' if not errors else 'failed', errors=errors,
                       objects=len(layout.objects), archived_settle=layout.provenance['settle'] is not None,
                       generation_seed=layout.provenance['env_seed'],
                       native_status='not_run', product_admission='held_missing_bound_manifest')
        except (KeyError, TypeError, ValueError) as exc:
            row['errors'] = [f'{type(exc).__name__}: {exc}']
        rows.append(row)
    missing = sorted(set(expected) - set(seen))
    extra = sorted(set(seen) - set(expected))
    duplicates = sorted(k for k, count in seen.items() if count != 1)
    return {'schema': 'mind2act_eval.variation_input_audit.v1', 'denominator': len(expected),
            'examined': len(rows), 'parsed': sum(r['cpu_layout_parse'] == 'passed' for r in rows),
            'missing': missing, 'extra': extra, 'duplicates': duplicates, 'rows': rows,
            'resolved_plans': 0, 'native_completed': 0,
            'qualification': 'input schema/slot inspection only; no asset closure or native acceptance'}


def smoke(rows, expected_order):
    by_tier = defaultdict(dict)
    for row in rows:
        if 'slot' in row:
            by_tier[(row['task'], row['tier'])][row['slot']] = row
    result = []
    for task, tier in sorted(expected_order):
        slots = expected_order[(task, tier)]
        # First frozen slot is reserved for prospective agentic review; two
        # independent digest-ranked samples are frozen before any GPU outcome.
        agentic = slots[0]
        ranked = sorted((s for s in slots if s != agentic),
                        key=lambda s: hashlib.sha256(s.encode()).hexdigest())
        selected = [agentic, *ranked[:2]]
        if len(selected) != 3:
            raise ValueError('each frozen tier needs three distinct smoke slots')
        cases = []
        for i, slot in enumerate(selected):
            row = by_tier[(task, tier)].get(slot)
            cases.append({'task': task, 'tier': tier, 'slot': slot,
                          'selection_role': 'prospective_agentic_review' if i == 0 else 'slot_sha256_order',
                          'layout': row['path'] if row else None,
                          'layout_sha256': row['file_sha256'] if row else None,
                          'first_frame': 'not_run', 'reviewer': None,
                          'expert_replay_selected': i == 0,
                          'expert_replay': 'missing_exact_instance_h5' if i == 0 else 'not_selected',
                          'clean_gate': 'not_required_for_viewing_smoke',
                          'product_admission': 'held_missing_bound_manifest'})
        result.extend(cases)
    return {'schema': 'mind2act_eval.variation_smoke_plan.v1', 'selection': 'frozen_index_first_plus_two_sha256',
            'first_frame_denominator': len(result),
            'expert_replay_denominator': sum(c['expert_replay_selected'] for c in result),
            'completed': 0, 'cases': result}
