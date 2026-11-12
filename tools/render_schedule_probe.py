"""Offline scheduling comparison using an existing, identity-bound expert H5.

Runs both shared runners with counted synthetic physics/rendering and a trace-only
scene. It does not start Isaac, a GPU, a model, or the native task/scorer. Output
proves scheduling/action/evidence-cadence equivalence, never native task success.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from mind2act_eval.contracts import BuildContext, ConsumerKind, Health, InstanceSpec, NativeEvidence, Observation, Score
from mind2act_eval.replay import H5Replay, ReplayMetadata, file_sha256
from mind2act_eval.runner import EpisodeLimits, run_episode
from mind2act_eval.tasks import TaskDefinition

CAMERAS = ('head', 'left_wrist', 'right_wrist')


class CountedSimulator:
    def __init__(self, video_hz):
        self.world = self
        self.video_hz = video_hz
        self.tick = 0
        self.rendered = self.sampled = None
        self.render_calls = self.capture_calls = 0
        self.required = True
        self.recorder = SimpleNamespace(frame_counts={name: 0 for name in CAMERAS})
        self.artifacts = {}
    def _refresh(self):
        if self.rendered != self.tick:
            self.rendered = self.tick
            self.render_calls += 1
        if self.sampled != self.tick:
            self.sampled = self.tick
            self.capture_calls += 1
    def ready(self):
        self._refresh()
        self.capture()
    def prepare_observation(self, requires_rgb):
        self.required = requires_rgb
        if requires_rgb: self._refresh()
    def rgb(self):
        return ({name: np.zeros((2, 2, 3), dtype=np.uint8) for name in CAMERAS} if self.required else {})
    def step(self, *, render):
        self.tick += 1
        if render:
            self.rendered = self.tick
            self.render_calls += 1
    def capture(self):
        self._refresh()
        for name in CAMERAS: self.recorder.frame_counts[name] += 1
    def close(self): pass


class CountedRobot:
    def __init__(self, sim):
        self.sim = sim
        self.hash = hashlib.sha256()
        self.applied = 0
    def apply(self, action):
        self.hash.update(struct.pack('<14d', *action.left_arm, action.left_gripper,
                                     *action.right_arm, action.right_gripper))
        self.applied += 1
    def observe(self):
        return dict(left_arm=(0.,) * 6, right_arm=(0.,) * 6, left_gripper=1., right_gripper=1., rgb=self.sim.rgb())


def compare_run(runner, spec, metadata, h5, expected, output, video_hz, limits):
    sim = CountedSimulator(video_hz)
    robot = CountedRobot(sim)
    evidence_hash = hashlib.sha256()
    class TraceScene:
        def health(self): return Health(True)
        def is_complete(self): return False
        def observe(self, tick): return Observation(tick, tick / spec.clock.physics_hz, **robot.observe())
        def evidence(self, tick):
            evidence_hash.update(struct.pack('<Q', tick))
            return NativeEvidence(spec.task_id, spec.instance_id, tick, {})
        def close(self): pass
    # Trace-only scorer; do not invoke or claim native task scoring.
    task = TaskDefinition(spec.task_id, lambda spec, ctx: TraceScene(), lambda e: Score(0.))
    adapter = H5Replay(h5, metadata, expected_sha256=expected)
    adapter.assert_spec(spec)
    rows = adapter.coverage_receipt()['rows_total']
    result = runner(spec, task, BuildContext(sim, robot, None, output), adapter,
                    consumer_kind=ConsumerKind.EXPERT_REPLAY,
                    limits=limits)
    if not result.infra_valid: raise RuntimeError(result.error)
    coverage = json.loads((output/'replay-coverage.json').read_text())
    assert coverage['clock_coverage_complete']
    return dict(rows=rows, physics_ticks=result.ticks, action_apply_calls=robot.applied,
                action_targets_sha256=robot.hash.hexdigest(), evidence_tick_sha256=evidence_hash.hexdigest(),
                render_calls=sim.render_calls, synchronized_capture_calls=sim.capture_calls,
                video_hz=video_hz, frame_counts=sim.recorder.frame_counts, coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'metadata', 'h5', 'baseline-runner', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--h5-sha256')
    parser.add_argument('--physics-ticks', required=True, type=int)
    parser.add_argument('--wall-s', required=True, type=float)
    args = parser.parse_args()
    spec = InstanceSpec.from_dict(json.loads(args.spec.read_text()))
    metadata = ReplayMetadata(**json.loads(args.metadata.read_text()))
    loader = importlib.util.spec_from_file_location('mind2act_eval.runner._baseline', args.baseline_runner)
    baseline = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = baseline
    loader.loader.exec_module(baseline)
    expected = args.h5_sha256 or file_sha256(args.h5)
    limits = EpisodeLimits(args.physics_ticks, args.wall_s)
    with tempfile.TemporaryDirectory(prefix='render-schedule-probe-') as directory:
        root = Path(directory)
        before_dir, after_dir = root/'before', root/'after'
        before_dir.mkdir(); after_dir.mkdir()
        before = compare_run(baseline.run_episode, spec, metadata, args.h5, expected, before_dir, 30, limits)
        after = compare_run(run_episode, spec, metadata, args.h5, expected, after_dir, 6, limits)
    for key in ('rows', 'physics_ticks', 'action_apply_calls', 'action_targets_sha256', 'evidence_tick_sha256', 'coverage'):
        assert before[key] == after[key], key
    result = dict(schema='mind2act_eval.offline_render_probe.v1',
                  native_physics_run=False, native_task_scoring_run=False, model_run=False,
                  task_id=spec.task_id, instance_id=spec.instance_id, seed=spec.seed,
                  limits=dict(physics_ticks=limits.physics_ticks, wall_s=limits.wall_s),
                  h5_sha256=expected, spec_sha256=file_sha256(args.spec),
                  metadata_sha256=file_sha256(args.metadata), baseline_runner_sha256=file_sha256(args.baseline_runner),
                  before=before, after=after,
                  render_reduction_fraction=1 - after['render_calls'] / before['render_calls'])
    with args.output.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
