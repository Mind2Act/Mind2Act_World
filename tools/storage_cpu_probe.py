"""Measure production post-seal cleanup on new CPU-only storage fixtures.

Not native qualification: fixture media/scoring bytes are synthetic. Writes
only the explicitly supplied, fresh report directory; never adopts old runs.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from mind2act_eval.contracts import ConsumerKind, InstanceSpec, Result, Termination, to_json
from mind2act_eval.episode_storage import durable_json

MIB = 1024 ** 2


def fill(path, mib):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        for _ in range(mib):
            stream.write(b'x' * MIB)


def worker(args):
    from mind2act_eval.sealing import seal_worker_outputs
    root = args.output
    fill(root / 'cache/fixture.bin', 200)
    fill(root / 'kit/data/fixture.bin', 240)
    for name in ('head', 'left_wrist', 'right_wrist'):
        fill(root / (name + '-media-fixture.bin'), 1)
    if args.large_evidence:
        fill(root / 'PR4/private-scoring-fixture.npz', 35)
    durable_json(root / 'evidence.json', dict(fixture=True))
    durable_json(root / 'score.json', dict(value=1.))
    result = Result('demo', 'storage-cpu-fixture', 1., Termination.PHYSICS_BUDGET,
        True, ConsumerKind.DIAGNOSTIC, False, 1,
        artifacts={**{name: name + '-media-fixture.bin' for name in ('head', 'left_wrist', 'right_wrist')},
                   **({'private_scoring': 'PR4/private-scoring-fixture.npz'} if args.large_evidence else {})})
    durable_json(root / 'result.json', to_json(result))
    for name, stage, details in (
        ('run-episode-returned.json', 'run_episode_returned', dict(infra_valid=True)),
        ('worker-returned.json', 'worker_returned', dict(exit_code=0)),
        ('cli-returned.json', 'cli_returned', dict(exit_code=0)),
    ):
        durable_json(root / name, dict(schema='mind2act_eval.worker_lifecycle.v1', stage=stage,
                                      pid=os.getpid(), **details))
    print('CPU storage fixture: retained diagnostic and exact scoring/media bytes', flush=True)
    seal_worker_outputs(root)
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--large-evidence', action='store_true')
    parser.add_argument('--spec')  # Supervisor freezes this standard worker argument.
    args = parser.parse_args()
    if args.worker:
        worker(args)
    from mind2act_eval.batch import EpisodeJob, run_attempt
    from mind2act_eval.sealing import validate_worker_seal
    # Children run in their fresh attempt directory, so a relative PYTHONPATH
    # cannot identify the source checkout being measured.
    os.environ['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
    args.output.mkdir(parents=True, exist_ok=False)
    spec = InstanceSpec('demo', 'storage-cpu-fixture', 'easy', 'synthetic-public-v1', 'v1', 'none', 'v1')
    rows = []
    for large in (False, True):
        output = args.output / ('private-evidence' if large else 'ordinary')
        argv = (sys.executable, str(Path(__file__).resolve()), '--worker', *(('--large-evidence',) if large else ()))
        result = run_attempt(EpisodeJob(spec, argv, ConsumerKind.DIAGNOSTIC, 180., 0), output)
        if not result.infra_valid:
            raise RuntimeError(result.error)
        validate_worker_seal(output, durable=True)
        receipt = json.loads((output / 'scratch-cleanup.json').read_text())
        assert receipt['status'] == 'complete'
        assert receipt['deleted_bytes'] == 440 * MIB
        assert (receipt['after_bytes'] > 30 * MIB) is large
        rows.append(dict(case=output.name, synthetic=True, seal_valid_after_cleanup=True, **receipt))
    report = dict(native_gpu_run=False, measurements=rows)
    durable_json(args.output / 'measurements.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
