"""Exercise the actual scene control/document path without an Isaac runtime."""
from types import SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.tasks.CP03.scene import CP03Scene


def scene(level):
    s = CP03Scene.__new__(CP03Scene)
    s._spec = SimpleNamespace(difficulty=level)
    mode = {'easy': 1, 'medium': 2, 'hard': 3}[level]
    angle = np.radians([18., 0., -18.][mode - 1])
    gear_q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0., 0.])
    s.depth, s.contact, s.gear_contact = 0., False, False
    s._controls = {'start': SimpleNamespace(get_world_pose=lambda: (np.array([0., 0., -s.depth]), [1., 0., 0., 0.]))}
    if level != 'easy':
        s._controls['gear'] = SimpleNamespace(get_world_pose=lambda: (np.zeros(3), gear_q))
    s._paths = {name: name for name in s._controls}
    s._origins = {'start': np.zeros(3)}
    s._finger = lambda path: s.contact if path == 'start' else s.gear_contact
    s._touch = lambda path, **kw: s._finger(path)
    s.S = {'mode': mode}
    s.C, s._relaxed = {}, False
    s.starts = []
    s._ns = {'stage': SimpleNamespace(GetPrimAtPath=lambda path: None),
             'rail_x': lambda t: 0., 'start_motor': s.starts.append}
    attr = SimpleNamespace(Set=lambda value: None)
    drive = SimpleNamespace(GetTargetPositionAttr=lambda: attr, GetStiffnessAttr=lambda: attr,
                            GetMaxForceAttr=lambda: attr)
    s._UsdPhysics = SimpleNamespace(DriveAPI=SimpleNamespace(Get=lambda *args: drive))
    s._motor = dict(started=False, gear_touched=True, mode_touched=False, speed_touched=False,
                    sent=False, start_down=False, start_released=False)
    s._stage_doc = dict(valid_start=False, gear_qualified=False, gear_consecutive_ticks=0)
    s._rail = dict(x=0., sign=0, reversals=0)
    s._failures, s._preloaded, s._sources = set(), [], []
    s._bodies, s._assigned = {}, {}
    s._start_tick, s._gear_set_tick, s._tick = None, None, 0
    return s


def tick(s, *, down=False, gear_contact=False):
    s._tick += 1
    s.depth, s.contact, s.gear_contact = (.002, True, gear_contact) if down else (0., False, gear_contact)
    s._control_and_placement(s._tick)
    s._sample_document_stage()


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
def test_start_is_valid_at_physical_press_without_release(level):
    s = scene(level)
    for _ in range(120):
        tick(s)
    tick(s, down=True)
    assert s._motor['started'] is True
    assert s._stage_doc['valid_start'] is True
    tick(s, down=True)
    assert s._stage_doc['valid_start'] is True
    tick(s)
    assert s._stage_doc['valid_start'] is True
    assert len(s.starts) == 1


@pytest.mark.parametrize('level', ['medium', 'hard'])
def test_gear_correctness_does_not_require_a_one_second_window(level):
    s = scene(level)
    for _ in range(119):
        tick(s)
    tick(s, down=True)
    for _ in range(130):
        tick(s, down=True)
    assert s._motor['started'] is True
    assert s._stage_doc['valid_start'] is True
    assert len(s.starts) == 1


@pytest.mark.parametrize('level', ['medium', 'hard'])
def test_selector_contact_does_not_block_correct_gear_start(level):
    s = scene(level)
    for _ in range(120):
        tick(s)
    # Contact resets the full released window, including on the Start tick.
    tick(s, down=True, gear_contact=True)
    assert s._motor['started'] is True


@pytest.mark.parametrize('level', ['medium', 'hard'])
def test_final_gear_needs_no_release_or_one_second_before_depth_start(level):
    s=scene(level)
    tick(s,gear_contact=True)
    tick(s,down=True,gear_contact=True)
    assert s._motor['started']
    assert s._stage_doc['valid_start']
    assert len(s.starts)==1


def test_final_start_depth_accepts_nonfinger_contact_and_exact_threshold():
    s=scene('easy')
    s.depth=.0015;s.contact=False;s._tick=1
    s._control_and_placement(1);s._sample_document_stage()
    assert s._motor['started'] and s._stage_doc['valid_start']
