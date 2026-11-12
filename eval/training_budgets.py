"""Derive control-aligned P90 budgets from manifest-selected H5 metadata."""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
from math import ceil

from .budgets import E2E_BUDGET_VERSION


def scan_training_root(root):
    """Read active manifest paths only; retired/rewritten copies are not samples."""
    from pathlib import Path
    import math
    import h5py
    root = Path(root).resolve()
    index_bytes = (root / 'INDEX.json').read_bytes()
    index = json.loads(index_bytes)
    output = dict(schema='mind2act.training_lengths.v1', root=str(root),
                  index_sha256=hashlib.sha256(index_bytes).hexdigest(), tasks=[], episodes=[])
    for task in index['tasks']:
        manifest_path = (Path(task['current_manifest']) if 'current_manifest' in task
                         else root / task['manifest_relative_path'])
        if not manifest_path.resolve().is_relative_to(root):
            raise ValueError('training manifest escapes selected root')
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
        public = task.get('public_task', task['task'])
        output['tasks'].append(dict(task_id=public, directory_task=task['task'],
            manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(raw).hexdigest(),
            index_manifest_sha256=task.get('current_manifest_sha256', task.get('manifest_sha256'))))
        paths = set()
        for row in manifest['rows']:
            path = Path(row['destination']) if 'destination' in row else root / row['relative_path']
            item = dict(task_id=public, difficulty=row['difficulty'], episode_id=row['id'] if 'id' in row else row['identity'],
                        path=str(path), expected_sha256=row['sha256'], expected_bytes=row.get('bytes'))
            try:
                if path in paths or not path.resolve().is_relative_to(root):
                    raise ValueError('duplicate or out-of-root active H5 path')
                paths.add(path)
                item['actual_bytes'] = path.stat().st_size
                with h5py.File(path, 'r') as stream:
                    shapes = {key: list(value.shape) for key, value in stream['action'].items()}
                    lengths = {shape[0] for shape in shapes.values()}
                    if len(lengths) != 1 or min(lengths) <= 0:
                        raise ValueError('expert action streams have inconsistent/empty lengths')
                    hz = float(stream['additional_info/frequency'][()])
                    if not math.isfinite(hz) or hz <= 0:
                        raise ValueError('invalid expert frequency')
                    count = lengths.pop()
                    item.update(rows=count, frequency_hz=hz, seconds=count/hz, action_shapes=shapes)
                item['size_matches_manifest'] = row.get('bytes') == item['actual_bytes']
            except Exception as exc:
                item['error'] = type(exc).__name__ + ': ' + str(exc)
            output['episodes'].append(item)
    return output


def quantile(values, numerator, denominator):
    """Linear interpolation at (n-1)*q, using exact rational arithmetic."""
    values = sorted(values)
    if not values:
        raise ValueError('quantile requires samples')
    at = Fraction((len(values) - 1) * numerator, denominator)
    low = at.numerator // at.denominator
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (at - low)


def summarize(inventory, *, old_budgets=None, base_budgets=None, wall_slowdown=None):
    if wall_slowdown is not None and Fraction(str(wall_slowdown)) <= 0:
        raise ValueError('wall slowdown must be positive')
    groups = defaultdict(list)
    manifests = {t['task_id']: t for t in inventory['tasks']}
    for row in inventory['episodes']:
        groups[row['task_id'], row['difficulty']].append(row)
    cells = []
    for (task, tier), episodes in sorted(groups.items()):
        errors = [r for r in episodes if 'error' in r]
        valid = [r for r in episodes if 'error' not in r]
        flags = []
        if len(valid) < 10:
            flags.append('fewer_than_10_experts')
        if errors:
            flags.append('unreadable_experts')
        if len(episodes) != 50:
            flags.append('sample_count_not_50')
        if any(not r['size_matches_manifest'] for r in valid):
            flags.append('h5_size_differs_from_manifest')
        manifest = manifests[task]
        if (manifest.get('index_manifest_sha256') is not None
                and manifest['index_manifest_sha256'] != manifest['manifest_sha256']):
            flags.append('index_manifest_sha_changed')
        values = [Fraction(r['rows']) / Fraction(str(r['frequency_hz'])) for r in valid]
        cell = dict(task_id=task, difficulty=tier, sample_count=len(valid),
                    declared_sample_count=len(episodes), flags=flags,
                    training_manifest_sha256=manifest['manifest_sha256'],
                    failed_sample_count=len(errors))
        if values:
            p50, p90 = quantile(values, 1, 2), quantile(values, 9, 10)
            if p90 > 3 * p50 or max(values) > 2 * p90:
                flags.append('large_duration_spread_review')
            if min(values) < 1:
                flags.append('expert_shorter_than_one_second')
            controls = ceil(p90 * Fraction(6, 5) * 30)
            seconds = Fraction(controls, 30)
            sample_set = sorted((r['episode_id'], r['expected_sha256'], r['rows'], r['frequency_hz'])
                                for r in valid)
            sample_sha = hashlib.sha256(json.dumps(sample_set, separators=(',', ':')).encode()).hexdigest()
            cell.update(p50_seconds=float(p50), p90_seconds=float(p90), max_seconds=float(max(values)),
                        min_seconds=float(min(values)), p90_seconds_exact=str(p90),
                        expert_sample_set_sha256=sample_sha,
                        physics_ticks=controls * 4, control_steps=controls,
                        simulated_seconds=float(seconds), simulated_seconds_exact=str(seconds),
                        wall_s=(max(1800, ceil(seconds * Fraction(str(wall_slowdown))))
                                if wall_slowdown is not None else None))
        key = task + '/' + tier
        old = (old_budgets or {}).get(key)
        base = (base_budgets or {}).get(key)
        cell['published_limits'] = old
        cell['old_base_limits'] = base
        if values and old:
            cell['new_over_published_physics'] = cell['physics_ticks'] / old['physics_ticks']
        cells.append(cell)
    return dict(schema='mind2act_eval.e2e_budget_table.v1', budget_version=E2E_BUDGET_VERSION,
                physics_hz=120, control_hz=30, quantile='linear:(n-1)*0.9',
                multiplier='6/5', rounding='ceil to 30Hz control step, then multiply by 4',
                training_index_sha256=inventory['index_sha256'],
                wall_slowdown=wall_slowdown, cells=cells)


def bind_case_budgets(document, table):
    """Prepare a new source index/catalog; never mutate the supplied release."""
    import copy
    if (table.get('budget_version') != E2E_BUDGET_VERSION
            or table.get('physics_hz') != 120 or table.get('control_hz') != 30
            or table.get('multiplier') != '6/5'):
        raise ValueError('wrong e2e budget table version')
    cells = {}
    for cell in table['cells']:
        key = cell['task_id'], cell['difficulty']
        if key in cells:
            raise ValueError('duplicate budget cell')
        cells[key] = cell
    output = copy.deepcopy(document)
    for row in output['instances']:
        cell = cells.get((row['task_id'], row['difficulty']))
        if cell is None:
            raise ValueError('source case has no training budget cell')
        if cell['sample_count'] < 10 or cell['flags'] or cell.get('wall_s') is None:
            raise ValueError('unqualified training statistics or unfrozen wall policy')
        controls = ceil(Fraction(cell['p90_seconds_exact']) * Fraction(6, 5) * 30)
        slowdown = Fraction(str(table['wall_slowdown']))
        if (slowdown <= 0 or cell['control_steps'] != controls or cell['physics_ticks'] != controls * 4
                or cell['wall_s'] != max(1800, ceil(Fraction(controls, 30) * slowdown))):
            raise ValueError('budget table differs from its declared P90 or wall formula')
        if row.get('limits_e2e') is not None or row.get('budget_version_e2e') is not None:
            raise ValueError('refusing to overwrite an existing e2e budget binding')
        row['limits_e2e'] = dict(physics_ticks=cell['physics_ticks'], wall_s=cell['wall_s'], retries=0)
        row['budget_version_e2e'] = E2E_BUDGET_VERSION
        from .budgets import validate_e2e_fields
        validate_e2e_fields(row)
    output['e2e_budget_source'] = dict(budget_version=E2E_BUDGET_VERSION,
        table_semantic_sha256=hashlib.sha256(json.dumps(table, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    return output



def rebind_cp01_case_budgets(document, *, previous_table, table):
    """Return a fresh CP01 budget revision; inputs and immutable releases stay intact.

    This explicit migration accepts only the approved restored CP01 sample source.
    Default binding still refuses every pre-existing binding. All non-CP01 cells,
    CP01 Easy statistics, instance identities, instructions and coding limits are
    preserved; only CP01 Medium/Hard e2e limits and the table digest may change.
    """
    import copy
    def semantic(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    def indexed(value):
        result = {(c['task_id'], c['difficulty']): c for c in value['cells']}
        if len(result) != len(value['cells']):
            raise ValueError('duplicate budget cell')
        return result
    # The exception is deliberately tied to the two reviewed immutable tables;
    # arbitrary changes under the unchanged budget-version string are rejected.
    if (semantic(previous_table) != 'a58d0183ef5c2ba666d9f2adde394fb037144a5eac7aa90c8f6927c461c68bb4'
            or semantic(table) != 'a81ba6f25f08d700145b581dd946c00610cdb38f98d2023a47a3bb23c03d5af7'):
        raise ValueError('CP01 rebinding requires the approved previous and restored source tables')
    old, new = indexed(previous_table), indexed(table)
    if set(old) != set(new):
        raise ValueError('CP01 rebinding cannot change the budget cell population')
    top_allowed = {'cells', 'training_index_sha256', 'training_source'}
    if ({k:v for k,v in previous_table.items() if k not in top_allowed}
            != {k:v for k,v in table.items() if k not in top_allowed}):
        raise ValueError('CP01 rebinding cannot change the budget protocol or wall policy')
    restored_manifest = 'e9bf08fd2f4229da8d060c57e6e49f891d2c7fa56f2c27ab31fc039e9ac1d37b'
    cp01_variable = {'training_manifest_sha256', 'p50_seconds', 'p90_seconds',
        'max_seconds', 'min_seconds', 'p90_seconds_exact', 'expert_sample_set_sha256',
        'physics_ticks', 'control_steps', 'simulated_seconds', 'simulated_seconds_exact',
        'wall_s', 'new_over_published_physics'}
    expected = {'easy': ('16081/300', 7720),
                'medium': ('10551/100', 15196), 'hard': ('3097/20', 22300)}
    for key, cell in new.items():
        before = old[key]
        if key[0] != 'CP01':
            if cell != before:
                raise ValueError('CP01 rebinding cannot change another task cell')
            continue
        tier = key[1]
        if tier not in expected:
            raise ValueError('unexpected CP01 difficulty')
        allowed = cp01_variable if tier != 'easy' else {'training_manifest_sha256'}
        if ({k:v for k,v in before.items() if k not in allowed}
                != {k:v for k,v in cell.items() if k not in allowed}):
            raise ValueError('unapproved CP01 budget cell change')
        exact, ticks = expected[tier]
        if (cell['training_manifest_sha256'] != restored_manifest
                or cell['sample_count'] != 50 or cell['flags']
                or cell['p90_seconds_exact'] != exact
                or cell['physics_ticks'] != ticks or cell['wall_s'] != ticks):
            raise ValueError('CP01 budget differs from the approved restored source')
    if set(k[1] for k in new if k[0] == 'CP01') != set(expected):
        raise ValueError('all three CP01 cells are required')
    source = document.get('e2e_budget_source', {})
    if (source.get('budget_version') != E2E_BUDGET_VERSION
            or source.get('table_semantic_sha256') != semantic(previous_table)):
        raise ValueError('source document does not bind the supplied previous table')
    fresh = copy.deepcopy(document)
    for row in fresh['instances']:
        cell = old.get((row['task_id'], row['difficulty']))
        if cell is None:
            raise ValueError('source case has no previous training budget cell')
        expected_limits = dict(physics_ticks=cell['physics_ticks'], wall_s=cell['wall_s'], retries=0)
        if row.get('budget_version_e2e') != E2E_BUDGET_VERSION or row.get('limits_e2e') != expected_limits:
            raise ValueError('source row differs from its previous budget binding')
        del row['limits_e2e']
        del row['budget_version_e2e']
    return bind_case_budgets(fresh, table)


def write_budget_csv(path, table):
    """Write the published budget columns with deterministic LF line endings."""
    import csv
    fields = ('task_id', 'difficulty', 'sample_count', 'min_seconds', 'p50_seconds',
              'p90_seconds', 'max_seconds', 'control_steps', 'physics_ticks',
              'simulated_seconds', 'wall_s', 'published_physics_ticks',
              'old_base_physics_ticks', 'flags')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for cell in table['cells']:
            row = dict(cell, flags=';'.join(cell['flags']),
                       published_physics_ticks=(cell.get('published_limits') or {}).get('physics_ticks'),
                       old_base_physics_ticks=(cell.get('old_base_limits') or {}).get('physics_ticks'))
            writer.writerow(row)

def main(argv=None):
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--inventory', type=Path)
    inputs.add_argument('--training-root', type=Path)
    inputs.add_argument('--document', type=Path, help='Prepare an enriched source index or ready catalog')
    parser.add_argument('--table', type=Path, help='Frozen e2e budget table for --document')
    parser.add_argument('--rebind-cp01-from', type=Path, help='Explicit previous table for a fresh CP01-only revision')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-slowdown', type=str)
    args = parser.parse_args(argv)
    if args.wall_slowdown is not None and Fraction(args.wall_slowdown) <= 0:
        parser.error('wall slowdown must be positive')
    if args.output.exists():
        parser.error('output exists; use a fresh producer-owned path')
    if not args.document and args.output.with_suffix('.csv').exists():
        parser.error('CSV output exists; use a fresh producer-owned path')
    if args.document:
        if not args.table or args.wall_slowdown is not None:
            parser.error('--document requires --table and its already frozen wall policy')
        document, table = json.loads(args.document.read_text()), json.loads(args.table.read_text())
        value = (rebind_cp01_case_budgets(document, previous_table=json.loads(args.rebind_cp01_from.read_text()), table=table)
                 if args.rebind_cp01_from else bind_case_budgets(document, table))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        return
    if args.table or args.rebind_cp01_from:
        parser.error('--table and --rebind-cp01-from require --document')
    inventory = (scan_training_root(args.training_root) if args.training_root
                 else json.loads(args.inventory.read_text()))
    table = summarize(inventory, wall_slowdown=args.wall_slowdown)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, allow_nan=False) + '\n')
    write_budget_csv(args.output.with_suffix('.csv'), table)


if __name__ == '__main__':
    main()
