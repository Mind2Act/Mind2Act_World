"""CPU measurements through the scene hooks, evidence builder and scorer."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from mind2act_eval.tasks.CP01.evidence import Evidence
from mind2act_eval.tasks.CP01.scene import CP01Scene
from mind2act_eval.tasks.CP01.scorer import score


def _trace(difficulty, last_action):
    n = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
    centers = [0.4 * j for j in range(n)]
    setup = dict(difficulty=difficulty, items=[{"kind": k} for _ in range(n) for k in "abc"],
                 orders=[list("abc") for _ in range(n)], tray_centers=centers, tray_center_y=0.,
                 tray_spec={"tray_size_m": [0.36, 0.14]}, reference_batch={"duration_s": 0.})
    base = dict(quaternion=[1, 0, 0, 0], local_center=[0, 0, 0],
                half_extents=[.01, .01, .01], bilateral=False, touch_robot=False,
                touch_table=False, touch_trays=[], tilt_deg=0.)
    bodies = [dict(base, index=i, position=[-1., 0., .8]) for i in range(3 * n)]
    bodies += [dict(base, index=3 * n + j, position=[centers[-1], 0., .768 + .05 * (n - 1 - j)],
                    touch_table=j == n - 1) for j in range(n)]
    rows = [dict(bodies=deepcopy(bodies), reference_visible=True)]
    for j, body in enumerate(bodies[3 * n:]):
        if j != n - 1 or last_action == "moved_and_replaced":
            body.update(position=[centers[j] + .10, 0., .90], bilateral=True,
                        touch_robot=True, touch_table=False)
        elif last_action == "grasped_without_lift":
            body.update(bilateral=True, touch_robot=True)
    rows.append(dict(bodies=deepcopy(bodies), reference_visible=False))
    for j, body in enumerate(bodies[3 * n:]):
        body.update(position=[centers[j], 0., .768], bilateral=False,
                    touch_robot=False, touch_table=True)
    rows.extend(dict(bodies=deepcopy(bodies), reference_visible=False) for _ in range(120))
    for body in bodies[:3 * n]:
        body.update(position=[-1., 0., .9], bilateral=True, touch_robot=True)
    rows.append(dict(bodies=deepcopy(bodies), reference_visible=False))
    for i, body in enumerate(bodies[:3 * n]):
        j, k = divmod(i, 3)
        body.update(position=[centers[j] + (k - 1) * .04, 0., .814],
                    bilateral=False, touch_robot=False, touch_trays=[j])
    rows.extend(dict(bodies=deepcopy(bodies), reference_visible=False) for _ in range(120))
    return setup, rows


def _scene(setup):
    # Replace only the USD measurement boundary; execute the actual control hooks.
    scene = CP01Scene.__new__(CP01Scene)
    scene._setup = setup
    scene._spec = SimpleNamespace(instance_id="cpu-trays", difficulty=setup["difficulty"])
    scene._evidence = Evidence(setup, 30.)
    scene._grade = None
    scene._phase = "observation"
    scene._reference_seen = False
    scene._error = None
    return scene


def _sample(scene, row):
    tick = scene._evidence.last_tick + 1
    scene._measure = lambda: deepcopy(row)
    scene.after_step(tick)
    return scene.evidence(tick)


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
@pytest.mark.parametrize("last_action", ["untouched", "moved_and_replaced", "grasped_without_lift"])
def test_last_tray_may_stay_move_or_be_grasped(difficulty, last_action):
    setup, rows = _trace(difficulty, last_action)
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert scene.health().ok
    assert not native.values["protocol_violations"]
    assert all(native.values["tray_readiness"])
    assert native.values["tray_steps"] == len(setup["orders"])
    assert native.values["steps_satisfied"] == {"medium": 8, "hard": 12}[difficulty]
    assert scene.is_complete()
    assert score(native).value == 100


@pytest.mark.parametrize("defect", ["held", "unsupported", "tilted", "unstable"])
def test_terminal_tray_requires_support_without_release_or_stability(defect):
    setup, rows = _trace("medium", "untouched")
    scene = _scene(setup)
    for row in rows:
        _sample(scene, row)
    assert scene.is_complete()
    last = rows[-1]
    tray = last["bodies"][-1]
    if defect == "held":
        tray["touch_robot"] = True
    elif defect == "unsupported":
        tray["touch_table"] = False
    elif defect == "tilted":
        tray["quaternion"] = [np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0]
    else:
        tray["position"][0] += .004
    native = _sample(scene, last)
    assert native.values["tray_readiness"][-1] is (defect != "unsupported")
    if defect == "unsupported":
        assert not scene.is_complete()
        assert score(native).value < 100


def test_replaced_last_tray_counts_without_stability_window():
    setup, rows = _trace("medium", "moved_and_replaced")
    scene = _scene(setup)
    for row in rows[:121]:  # initial + move + 119 settled samples
        native = _sample(scene, row)
    assert native.values["tray_readiness"][-1]
    native = _sample(scene, rows[121])
    assert native.values["tray_readiness"][-1]


def test_transport_tilt_does_not_veto_terminal_completion():
    setup, rows = _trace("medium", "untouched")
    # Existing food-transport violation is independent of terminal completion.
    rows[122]["bodies"][0]["tilt_deg"] = 31.
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert native.values["steps_satisfied"] == native.values["required_steps"] == 8
    assert not native.values["protocol_violations"]
    assert scene.is_complete()
    assert score(native).value == 100.


def test_observation_tilt_does_not_poison_later_formal_completion():
    setup, rows = _trace('medium', 'untouched')
    rows[0]['bodies'][0].update(bilateral=True, touch_robot=True, tilt_deg=45.)
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert not native.values['protocol_violations']
    assert score(native).value == 100.


def test_easy_preplaced_tray_still_allows_three_food_success():
    setup, rows = _trace("easy", "untouched")
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert scene.is_complete()
    assert native.values["tray_steps"] == 0
    assert native.values["steps_satisfied"] == native.values["required_steps"] == 3
    assert score(native).value == 100


@pytest.mark.parametrize("difficulty", ["medium", "hard"])
def test_already_separated_trays_need_no_grasp_or_fixed_target(difficulty):
    setup, rows = _trace(difficulty, "untouched")
    count = len(setup["items"])
    # Start separated and stay there, away from every configured target.
    for row in rows:
        for j, tray in enumerate(row["bodies"][count:]):
            tray.update(position=[.4 * j, .25, .768], bilateral=False,
                        touch_robot=False, touch_table=True)
        for food in row["bodies"][:count]:
            if food["touch_trays"]:
                food["position"][1] += .25
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert scene.is_complete()
    assert score(native).value == 100


@pytest.mark.parametrize("offset", [0., .20])
def test_stacked_or_overlapping_trays_cannot_be_ready(offset):
    setup, rows = _trace("medium", "untouched")
    count = len(setup["items"])
    # Even apparently valid table contacts cannot substitute for separation.
    for row in rows:
        for j, tray in enumerate(row["bodies"][count:]):
            tray.update(position=[j * offset, 0., .768], bilateral=False,
                        touch_robot=False, touch_table=True)
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert not any(native.values["tray_readiness"])
    assert not scene.is_complete()


def test_relocated_last_tray_recovers_at_current_terminal_frame():
    setup, rows = _trace("medium", "moved_and_replaced")
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert scene.is_complete()
    relocated = deepcopy(rows[-1])
    relocated["bodies"][-1]["position"][1] += .25
    for food in relocated["bodies"][:len(setup["items"])]:
        if food["touch_trays"] == [1]:
            food["position"][1] += .25
    native = _sample(scene, relocated)
    assert scene.is_complete()
    for _ in range(119):
        native = _sample(scene, relocated)
    assert scene.is_complete()
    assert score(native).value == 100
    assert not native.values["protocol_violations"]


@pytest.mark.parametrize("overlap", [False, True])
def test_separation_uses_rotated_tray_footprints(overlap):
    setup, rows = _trace("medium", "untouched")
    count = len(setup["items"])
    for row in rows:
        for j, tray in enumerate(row["bodies"][count:]):
            # Long axes run along y: .20m is enough along x, but overlaps along y.
            tray.update(position=[0. if overlap else .20 * j, .20 * j if overlap else 0., .768],
                        quaternion=[np.sqrt(.5), 0, 0, np.sqrt(.5)],
                        bilateral=False, touch_robot=False, touch_table=True)
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)
    assert tuple(native.values["tray_readiness"]) == (not overlap, not overlap)


@pytest.mark.parametrize('difficulty', ['medium', 'hard'])
def test_food_ready_before_tray_release_counts_in_terminal_success(difficulty):
    setup, rows = _trace(difficulty, 'untouched')
    count = len(setup['items'])
    # Foods finish their real pickup/placement while another hand still
    # contacts each otherwise supported tray. Release trays only afterwards.
    for row in rows:
        for tray in row['bodies'][count:]:
            tray['touch_robot'] = True
    released = deepcopy(rows[-1])
    for tray in released['bodies'][count:]:
        tray['touch_robot'] = False
    rows.extend(deepcopy(released) for _ in range(120))
    scene = _scene(setup)
    for row in rows:
        native = _sample(scene, row)

    assert native.values['task_success'] and scene.is_complete()
    result = score(native)
    assert native.values['credited_foods'] == count
    assert native.values['steps_satisfied'] == native.values['required_steps']
    assert result.value == 100

    # Harmless robot contact does not remove actual tray support.
    held = deepcopy(rows[-1])
    held['bodies'][0]['touch_robot'] = True
    native = _sample(scene, held)
    assert scene.is_complete()
    assert native.values['steps_satisfied'] == native.values['required_steps']
    assert score(native).value == 100
