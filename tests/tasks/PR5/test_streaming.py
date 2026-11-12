from copy import deepcopy
import json
from pathlib import Path

import pytest

from mind2act_eval.contracts import to_json
from mind2act_eval.tasks.PR5.evidence import Recorder, reconstruct
from mind2act_eval.tasks.PR5.mechanism.imprint import Frame, ImprintTracker
from mind2act_eval.tasks.PR5.mechanism.streaming import StreamingImprints
from .test_evidence import case, measured


def assert_equivalent(oracle, online):
    expected = deepcopy(oracle).finish()
    count, marks = online.preview()
    assert count == len(expected)
    assert [vars(mark) for mark in marks] == [vars(mark) for mark in expected[:2]]


def test_incremental_marks_equal_source_oracle_at_cutoffs_and_multiple_segments():
    oracle, online = ImprintTracker(), StreamingImprints()
    tick = 0
    for width in (1, 12, 13, 14, 60, 600, 2000):
        for i in range(width):
            frame = Frame(tick/120, 5.+i%3, .001*(i%5), .0004*(i%7),
                          (179.+i*.3+180.)%360.-180., 10.)
            oracle.update(frame)
            online.update(frame)
            tick += 1
            if i % 71 == 0 or i == width-1:
                assert_equivalent(oracle, online)
        frame = Frame(tick/120, 4.999, 0., 0., 0., 0.)
        oracle.update(frame)
        online.update(frame)
        tick += 1
        assert_equivalent(oracle, online)


def test_equal_force_peaks_keep_the_first_source_pose_and_twist_boundary():
    oracle, online = ImprintTracker(), StreamingImprints()
    frames = [Frame(0., 7., 0., 0., 179., 10.), Frame(.1, 7., .003, 0., -176., 10.),
              Frame(.11, 8., 0., 0., 0., 10.001)]
    for frame in frames:
        oracle.update(frame)
        online.update(frame)
        assert_equivalent(oracle, online)
    count, marks = online.preview()
    assert count == 1 and marks[0].x == 0. and not marks[0].smeared


def test_evidence_receipt_stays_bounded_as_the_complete_raw_trace_grows(tmp_path):
    rec = Recorder('PR5', 'synthetic', case(), trace_path=tmp_path/'raw-trace.jsonl')
    sizes = []
    for tick in range(1, 4001):
        rec.record(tick, [.02, -.2, .772], [1., 0., 0., 0.],
                   {'Left': [0., 0., 1.], 'Right': [0., 0., 1.]}, [])
        if tick in (1000, 2000, 4000):
            evidence = rec.evidence(tick)
            assert evidence.values['raw_trace']['frames'] == tick
            sizes.append(len(json.dumps(to_json(evidence))))
    assert max(sizes) < 2000 and max(sizes)-min(sizes) < 30
    assert len(rec.trace_path.read_bytes().splitlines()) == 4000  # no skipped physics samples
    assert reconstruct(rec.evidence(4000)) == rec.facts()
    rec.close()


@pytest.mark.parametrize('damage', ['edit', 'append', 'truncate'])
def test_raw_trace_hash_and_size_are_verified_before_a_score_can_be_returned(tmp_path, damage):
    rec = measured(tmp_path)
    evidence = rec.evidence(76)
    path = Path(evidence.values['raw_trace']['path'])
    data = path.read_bytes()
    if damage == 'edit':
        data = data.replace(b'0.772', b'0.773', 1)  # same size, different physical measurement
    elif damage == 'append':
        data += b'\n'
    else:
        data = data[:-1]
    path.write_bytes(data)
    with pytest.raises(ValueError, match='trace'):
        reconstruct(evidence)


@pytest.mark.parametrize('start', [1, 3, 7, 599, 1201, 100001])
def test_nominal_point_one_second_is_not_lost_to_subtraction(start):
    oracle, online = ImprintTracker(), StreamingImprints()
    for tick in range(start, start + 13):
        frame = Frame(tick / 120, 6., 0., 0., 0., 0.)
        oracle.update(frame)
        online.update(frame)
    assert len(oracle.finish()) == online.preview()[0] == 1
