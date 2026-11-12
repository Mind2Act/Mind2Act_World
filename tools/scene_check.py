#!/usr/bin/env python3
"""Load one explicitly selected clean instance in each of the 90 task/tier cells.

The manifest is a JSON list of {spec, asset_root} objects. Paths are relative to
its directory. Every instance is loaded unchanged in a fresh process, without
a model or score. Use --plan-only to validate selections on CPU.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from mind2act_eval.contracts import InstanceSpec
from mind2act_eval.scene_probe import SOURCE_CLEAN_PREVIEW, spec_digest
from mind2act_eval.tasks.catalog import TASK_MODULES

TIERS = ('easy', 'medium', 'hard')


def plan(manifest):
    manifest = Path(manifest).resolve()
    expected = {(task, tier) for task in TASK_MODULES if task != 'demo' for tier in TIERS}
    selected = {}
    for entry in json.loads(manifest.read_text()):
        path = (manifest.parent / entry['spec']).resolve(strict=True)
        spec = InstanceSpec.from_dict(json.loads(path.read_text()))
        key = spec.task_id, spec.difficulty
        if key not in expected or key in selected or spec.variation.mode != 'clean':
            raise ValueError('each cell requires one distinct clean instance: ' + str(key))
        root = (manifest.parent / entry['asset_root']).resolve(strict=True)
        selected[key] = {'spec': str(path), 'asset_root': str(root), 'task': spec.task_id,
                         'tier': spec.difficulty, 'instance_id': spec.instance_id,
                         'probe': {'schema': 'mind2act_eval.scene_probe.v1',
                                   'projection': SOURCE_CLEAN_PREVIEW, 'task_id': spec.task_id,
                                   'difficulty': spec.difficulty, 'source_spec_sha256': spec_digest(spec)}}
    if selected.keys() != expected:
        raise ValueError('missing cells: ' + ', '.join('/'.join(k) for k in sorted(expected - selected.keys())))
    return [selected[key] for key in sorted(selected, key=lambda k: (k[0], TIERS.index(k[1])))]


def contact_sheet(rows, out):
    from PIL import Image, ImageDraw
    width, height = 320, 265
    canvas = Image.new('RGB', (width * 3, height * 30), '#202020')
    draw = ImageDraw.Draw(canvas)
    for i, row in enumerate(rows):
        x, y = (i % 3) * width, (i // 3) * height
        frame = row.get('first_frame')
        if frame:
            with Image.open(frame) as source:
                picture = source.convert('RGB'); picture.thumbnail((width, 240))
                canvas.paste(picture, (x, y + 25))
        draw.text((x + 5, y + 5), f"{row['task']}/{row['tier']}: {row['status']}", fill='white')
    canvas.save(out / 'first-frames.jpg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--wall-s', type=float, default=1800)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not args.plan_only and (args.gpu is None or args.gpu < 0):
        parser.error('an explicit nonnegative --gpu is required')
    selections = plan(args.manifest)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'selection.json').write_text(json.dumps(selections, indent=2))
    if args.plan_only:
        print('90 fixed clean instances selected; no scene was loaded.')
        return 0
    rows = []
    for selected in selections:
        cell = args.out / (selected['task'] + '-' + selected['tier']); cell.mkdir()
        config = cell / 'probe.json'
        raw = (json.dumps(selected['probe'], sort_keys=True) + '\n').encode(); config.write_bytes(raw)
        command = [sys.executable, '-m', 'mind2act_eval.cli', 'run', '--spec', selected['spec'],
                   '--asset-root', selected['asset_root'], '--output-root', str(cell / 'runs'),
                   '--backend', 'isaac', '--consumer', 'scene_probe', '--gpu', str(args.gpu),
                   '--probe-config', str(config), '--probe-config-sha256', sha256(raw).hexdigest(),
                   '--physics-ticks', '600', '--wall-s', str(args.wall_s), '--retries', '0']
        with (cell / 'launch.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        receipts = list((cell / 'runs').glob('batch-*/episode-*/scene-probe-receipt.json'))
        receipt = json.loads(receipts[0].read_text()) if len(receipts) == 1 else {}
        frame = receipts[0].parent / 'before-head.png' if receipts else None
        passed = result.returncode == 0 and receipt.get('infra_valid') is True
        rows.append({'task': selected['task'], 'tier': selected['tier'], 'instance_id': selected['instance_id'],
                     'status': 'loaded' if passed else 'failed', 'exit_code': result.returncode,
                     'first_frame': str(frame) if frame and frame.is_file() else None,
                     'error': receipt.get('error'), 'log': str(cell / 'launch.log')})
        (args.out / 'results.json').write_text(json.dumps(rows, indent=2))
    failures = [row for row in rows if row['status'] != 'loaded']
    (args.out / 'failures.json').write_text(json.dumps(failures, indent=2))
    contact_sheet(rows, args.out)
    print(f'{90-len(failures)}/90 loaded; inspect first-frames.jpg manually. No benchmark score was produced.')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
