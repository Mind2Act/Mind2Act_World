"""CPU encoder startup before the PRO native heap grows; identical RGB output."""
from contextlib import contextmanager, nullcontext
import io
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.sim.video import RGBRecorder


def test_prepared_encoders_never_spawn_on_first_frame(tmp_path, monkeypatch):
    phase = ['small-heap']
    children = []
    def popen(argv, **options):
        assert phase[0] == 'small-heap', 'encoder fork after native heap grew'
        child = SimpleNamespace(stdin=io.BytesIO(), wait=lambda **kw: 0)
        children.append((argv, child))
        return child
    monkeypatch.setattr(subprocess, 'Popen', popen)
    recorder = RGBRecorder(tmp_path, 6)
    sizes = {'head': (32, 24), 'left_wrist': (16, 16), 'right_wrist': (16, 16)}
    recorder.prepare(sizes)
    phase[0] = 'native-heap'
    frames = {name: np.full((h, w, 3), i, np.uint8)
              for i, (name, (w, h)) in enumerate(sizes.items())}
    try:
        recorder.capture(frames)
        assert len(children) == 3
        for (argv, child), (name, frame) in zip(children, frames.items()):
            assert child.stdin.getvalue() == frame.tobytes()
            assert argv[argv.index('-r') + 1] == '6'
            assert argv[argv.index('-s') + 1] == f'{frame.shape[1]}x{frame.shape[0]}'
        assert recorder.frame_counts == {name: 1 for name in sizes}
    finally:
        recorder.close()


def test_public_runtime_starts_encoders_on_capture(tmp_path, monkeypatch):
    from mind2act_eval import sim, startup_gate
    from mind2act_eval.contracts import ClockSpec
    calls = []
    sizes = {'head': (32, 24)}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setenv('MIND2ACT_EVAL_LOCK_ROOT', str(tmp_path / '.locks'))
    monkeypatch.setattr(sim, 'find_spec', lambda _: object())
    monkeypatch.setattr(startup_gate, 'simulator_gates', lambda *a, **kw:
                        (nullcontext(lambda: None), nullcontext(lambda: None)))
    def prepare(self, configured):
        assert configured == sizes
        calls.append('encoders')
    monkeypatch.setattr(RGBRecorder, 'prepare', prepare, raising=False)
    def app(config):
        assert calls == []
        calls.append('kit')
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setitem(sys.modules, 'isaacsim', SimpleNamespace(SimulationApp=app))
    instance = sim.IsaacSimulator(ClockSpec(), tmp_path / 'attempt', physical_gpu=4,
        cameras={'head': {'path': '/World/HeadCamera', 'resolution': [32, 24]}})
    instance.close()
    assert calls == ['kit']


def test_prepared_video_preserves_decoded_frames_and_resolution_guard(tmp_path):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('explicit local ffmpeg required')
    frames = []
    for i in range(4):
        frame = np.zeros((32, 32, 3), np.uint8)
        frame[..., 1] = 160 + i * 20
        frames.append(frame)
    decoded = []
    for eager in [False, True]:
        output = tmp_path / str(eager); output.mkdir()
        recorder = RGBRecorder(output, 6, ffmpeg=ffmpeg)
        if eager:
            recorder.prepare({'head': (32, 32)})
        try:
            for frame in frames:
                recorder.capture({'head': frame})
            with pytest.raises(ValueError, match='resolution changed'):
                recorder.capture({'head': np.zeros((16, 16, 3), np.uint8)})
        finally:
            recorder.close()
        assert recorder.frame_counts == {'head': 4}
        decoded.append(subprocess.run([ffmpeg, '-v', 'error', '-i', str(output / 'head.mp4'),
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], check=True, capture_output=True).stdout)
    assert decoded[0] == decoded[1]
    assert len(decoded[0]) == 4 * 32 * 32 * 3


def test_denied_startup_never_starts_encoder(tmp_path, monkeypatch):
    from mind2act_eval import sim, startup_gate
    from mind2act_eval.contracts import ClockSpec
    @contextmanager
    def denied():
        raise PermissionError('startup admission denied')
        yield  # pragma: no cover
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4')
    monkeypatch.setenv('MIND2ACT_EVAL_LOCK_ROOT', str(tmp_path / '.locks'))
    monkeypatch.setattr(sim, 'find_spec', lambda _: object())
    monkeypatch.setattr(startup_gate, 'simulator_gates', lambda *a, **kw:
                        (nullcontext(lambda: None), denied()))
    monkeypatch.setattr(RGBRecorder, 'prepare', lambda *a: pytest.fail('encoder before admission'))
    monkeypatch.setitem(sys.modules, 'isaacsim', SimpleNamespace(
        SimulationApp=lambda config: pytest.fail('Kit before admission')))
    with pytest.raises(PermissionError, match='startup admission denied'):
        sim.IsaacSimulator(ClockSpec(), tmp_path / 'attempt', physical_gpu=4)
