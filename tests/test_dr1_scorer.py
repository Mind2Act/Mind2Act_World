import json
from pathlib import Path

import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.DR1.scorer import score
from mind2act_eval.tasks.DR1.instances import spec_from_row

import dr1_factory

TRAY = [-.34, .34, -.275, -.155, .768, .784]
SPEED = .08


def parcel(x=0., y=-.215, z=.80, finger=False, tray=True, active=True, v=0.):
    return dict(active=active, position=[x, y, z], extent=[.05, .05, .05], velocity=[v, 0, 0],
                finger_contact=finger, belt_contact=False, tray_contact=tray, tray_supported=tray)


def evidence(rows_of_parcels, speed=SPEED, belt=None):
    ticks = [dict(tick=i, belt_velocity=belt(i) if belt else [SPEED, 0., 0.], parcels=ps)
             for i, ps in enumerate(rows_of_parcels, 1)]
    return NativeEvidence('DR1', 'FR1/easy/x', len(ticks),
                          dict(ticks=ticks, count=len(rows_of_parcels[0]), tray_bounds_m=TRAY, belt_speed_m_s=speed))


def episode(final, length=200):
    return evidence([final] * length)


def test_all_placed_scores_100():
    s = score(episode([parcel(), parcel(x=.1), parcel(x=-.1)]))
    assert s.value == 100. and s.components['completion']['k'] == 3 and s.components['task_success']


def test_partial_is_twenty_times_fraction():
    s = score(episode([parcel(), parcel(x=.9, tray=False), parcel(x=.1), parcel(x=-.1, finger=True), parcel(x=.2)]))
    assert s.components['completion']['k'] == 4 and s.value == pytest.approx(17.6) and not s.components['task_success']


def test_hard_three_of_seven_is_8_57():
    ps = [parcel(x=-.3 + .1 * i) for i in range(3)] + [parcel(x=1., tray=False) for _ in range(4)]
    assert score(episode(ps)).value == pytest.approx(15 * 3 / 7)


def test_unsupported_parcel_not_counted():
    assert score(episode([parcel(tray=False)])).components['completion']['k'] == 0


def test_still_held_by_finger_in_last_second_not_counted():
    s = score(episode([parcel(finger=True)]))
    assert s.components['completion']['k'] == 1 and s.value == 100.


def test_moving_parcel_not_stable():
    assert score(episode([parcel(v=.1)])).components['completion']['k'] == 1


def test_belt_stop_is_reported_but_does_not_block_success():
    s = score(evidence([[parcel()]] * 200, belt=lambda i: [0., 0., 0.] if i == 50 else [SPEED, 0., 0.]))
    assert s.components['completion']['k'] == 1 and s.value == 100. and not s.components['belt_continuous']


def test_round_does_not_end_while_a_parcel_is_missing_or_moving():
    from mind2act_eval.tasks.DR1.scorer import round_over
    still = [parcel(), parcel(x=.1)]
    assert not round_over([dict(tick=i, parcels=still) for i in range(1, 80)], 2)
    assert not round_over([dict(tick=i, parcels=still) for i in range(1, 121)], 2)
    moving = [parcel(), parcel(x=-1., tray=False, v=.08)]
    assert not round_over([dict(tick=i, parcels=moving) for i in range(1, 80)], 2)
    riding = [parcel(), dict(parcel(x=-1., tray=False), belt_contact=True)]
    assert not round_over([dict(tick=i, parcels=riding) for i in range(1, 80)], 2)


def test_lost_parcel_lying_still_off_the_belt_lets_the_round_end_with_partial_score():
    from mind2act_eval.tasks.DR1.scorer import round_over
    rows = [parcel(), parcel(x=2., z=.1, tray=False)]
    assert not round_over([dict(tick=i, parcels=rows) for i in range(1, 121)], 2)
    assert score(episode(rows)).value == 15.


def test_parcel_never_appeared_counts_zero_not_error():
    assert score(episode([parcel(active=False)])).value == 0.


def test_short_episode_cannot_pass():
    assert score(evidence([[parcel()]] * 10)).value == 100.


def test_missing_ticks_rejected():
    with pytest.raises(ValueError):
        score(NativeEvidence('DR1', 'x', 2, dict(ticks=[dict(tick=2, belt_velocity=[SPEED, 0, 0], parcels=[parcel()])],
                                                  count=1, tray_bounds_m=TRAY, belt_speed_m_s=SPEED)))


def test_scorer_ignores_expert_labels():
    ps = [dict(parcel(), arm_states={'Left': {'phase': 'release'}})]
    assert score(episode(ps)).value == 100.


def test_instances_resolve_per_difficulty_without_expert_fields():
    row = dr1_factory.row()
    spec = spec_from_row(row, asset_version='t')
    assert spec.difficulty == 'easy' and len(spec.config['source_recipe']['items']) == 3
    assert all('closed_joint_target_m' not in i for i in spec.config['source_recipe']['items'])
    row['difficulty'] = 'hard'
    with pytest.raises(ValueError):
        spec_from_row(row, asset_version='t')


class Body:
    def __init__(self, pos): self.pos = pos
    def get_world_pose(self): return self.pos, [1, 0, 0, 0]


class Rb:
    def __init__(self, value): self.value = value
    def GetKinematicEnabledAttr(self): return self
    def Get(self): return self.value


def fake_scene(kinematic, active, pos=(0., 0., .8)):
    from types import SimpleNamespace
    from mind2act_eval.tasks.DR1.scene import ConveyorScene
    h = SimpleNamespace(rbapis=[Rb(kinematic)])
    row = dr1_factory.row()
    spec = spec_from_row(row, asset_version='t')
    rec = SimpleNamespace(ticks=[])
    return ConveyorScene(spec, None, h, [Body(pos)], rec, None, [active], None, TRAY, None)


def test_health_returns_explicit_type_never_none():
    from mind2act_eval.contracts import Health
    assert fake_scene(True, False).health() == Health(True)
    assert fake_scene(False, True).health().ok
    assert not fake_scene(None, False).health().ok            # unreadable flag is reported, not swallowed
    assert not fake_scene(True, True).health().ok             # kinematic after activation
    assert not fake_scene(True, False, pos=(float('nan'), 0, 0)).health().ok


def test_non_baseline_physics_clock_is_rejected_at_entry_and_build():
    from mind2act_eval.contracts import ClockSpec, InstanceSpec, to_json
    from mind2act_eval.tasks.DR1.scene import validate
    with pytest.raises(ValueError):
        spec_from_row(dr1_factory.row(), asset_version='t', clock=ClockSpec(240, 30, 30))
    forged = InstanceSpec.from_dict(dict(to_json(spec_from_row(dr1_factory.row(), asset_version='t')),
                                         clock=dict(physics_hz=240, control_hz=30, rgb_hz=30)))
    with pytest.raises(ValueError):
        validate(forged)


def test_t0_supported_center_in_tray_counts_without_dwell_or_release():
    assert score(evidence([[parcel(x=.339, finger=True, v=.1)]])).value == 100.
    assert score(evidence([[parcel(x=.340001)]])).value == 0.
    assert score(evidence([[parcel(tray=False)]])).value == 0.


def test_contact_without_measured_support_is_infra():
    p = parcel()
    del p['tray_supported']
    with pytest.raises(ValueError, match='support'):
        score(evidence([[p]]))
