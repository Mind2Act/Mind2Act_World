"""The native scene must send partial occupants to H/X, independently of cup containment."""
from itertools import product
from types import SimpleNamespace as NS

import numpy as np
import pytest

from mind2act_eval.contracts import NativeEvidence
from mind2act_eval.tasks.P1.scene import PScene
from mind2act_eval.tasks.P1.scorer import score
from mind2act_eval.sim.contacts import ContactLedger


def box(lo, hi):
    return np.array(list(product(*zip(lo, hi))), dtype=float)


class BoxRays:
    def intersects_any(self, points, directions):
        return (np.abs(points[:, :2]) <= .05).all(axis=1)


def native_scene(monkeypatch, n):
    import mind2act_eval.tasks.P1.scene as module
    monkeypatch.setattr(module, 'BatchedCupRays', lambda mesh: mesh.ray)
    foods = {f'f{i}': {'id': f'f{i}', 'type': 'apple', 'prim_path': f'/World/Food{i}'} for i in range(n + 1)}
    # The extra object crosses the rim horizontally, rather than being fully in the cup.
    positions = {ident: np.array([0., 0., .04]) for ident in foods}
    positions[f'f{n}'] = np.array([.05, 0., .04])
    bodies = {ident: NS(get_world_pose=lambda ident=ident: (positions[ident], np.array([1., 0., 0., 0.]))) for ident in foods}
    mesh = NS(vertices=box([-.05, -.05, 0.], [.05, .05, .1]), ray=BoxRays())
    manifest = dict(foods=list(foods.values()), containers=[dict(id='cupA', prim_path='/World/CupA', world_bounds=[[-.05, -.05, 0.], [.05, .05, .1]])],
                    station=dict(world_support_z=0.))
    s = PScene(NS(), NS(sim=NS(contacts=ContactLedger())), None, manifest, bodies, {i: np.zeros(3) for i in foods}, {'cupA': mesh},
               {i: box([-.01, -.01, -.01], [.01, .01, .01]) for i in foods}, None, None, None)
    s._support = lambda food, *args: positions[food][2] <= .1
    return s, positions


@pytest.mark.parametrize('level,n', [('easy', 4), ('medium', 5), ('hard', 7)])
def test_boundary_extra_is_measured_and_costs_x(monkeypatch, level, n):
    s, positions = native_scene(monkeypatch, n)
    for tick in range(1, 122):
        # Give the first N their entry ticks before the boundary extra enters.
        if tick == 1:
            positions[f'f{n}'][0] = .2
        elif tick == 2:
            positions[f'f{n}'][0] = .05
        measured = s._measure()
        s.ledger.observe(tick, measured, published_demand={'cupA/apple': n}, all_orders_published=True)
    result = score(NativeEvidence('P1', 'synthetic', 121, dict(s.ledger.snapshot(), difficulty=level)))
    assert measured[f'f{n}']['occupied_containers'] == ['cupA']
    assert measured[f'f{n}']['inside'] is True
    assert result.components['credited_units'] == n
    assert result.components['extra'] == 1
    assert result.value == {'easy':15.,'medium':11.,'hard':7.5}[level]


def test_center_on_real_cup_boundary_counts_without_full_hull_containment(monkeypatch):
    s, _ = native_scene(monkeypatch, 4)
    for tick in range(1, 122):
        measured = s._measure()
        s.ledger.observe(tick, measured, published_demand={'cupA/apple': 4}, all_orders_published=True)
    partial = next(r for r in s.ledger.snapshot()['occupancy'] if r['object_id'] == 'f4')
    assert partial['inside'] is True


def test_disjoint_above_and_outside_food_do_not_occupy(monkeypatch):
    s, positions = native_scene(monkeypatch, 4)
    for position in ([.2, 0., .04], [0., 0., .2], [.07, .07, .04]):
        positions['f4'][:] = position
        assert s._measure()['f4']['occupied_containers'] == []


def test_overlapping_bounds_without_volume_intersection_is_not_occupancy():
    from mind2act_eval.tasks.P1.mechanism.cup_geometry import OccupancyVolume
    tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    cup = OccupancyVolume(tetra)
    food = OccupancyVolume(tetra * .1)
    assert not cup.intersects(food, [.8, .8, .8, 1., 0., 0., 0.])
    assert cup.intersects(food, [.1, .1, .1, 1., 0., 0., 0.])


def test_multi_cup_boundary_membership_remains_explicitly_ambiguous(monkeypatch):
    from mind2act_eval.tasks.P1.mechanism.cup_geometry import OccupancyVolume
    s, positions = native_scene(monkeypatch, 4)
    s.cups['cupB'] = dict(s.cups['cupA'], id='cupB')
    from scipy.spatial import ConvexHull
    s.cup_regions['cupB'] = ConvexHull(s.cup_meshes['cupA'].vertices[:, :2] + [.08, 0.]).equations
    positions['f4'][:] = [.04, 0., .04]
    measured = s._measure()
    assert measured['f4']['occupied_containers'] == ['cupA', 'cupB']
    assert measured['f4']['inside'] is False
    s.ledger.observe(1, measured, published_demand={}, all_orders_published=False)
    assert s.ledger.snapshot()['occupancy_complete'] is False


def test_measured_upward_support_not_side_touch_controls_membership(monkeypatch):
    s, positions = native_scene(monkeypatch, 4)
    s._support = PScene._support.__get__(s, PScene)
    ledger = s.ctx.sim.contacts
    own, cup = '/World/Food0/collision', '/World/CupA/collision'
    ledger.record(own, cup, active=True, impulse_ns=0., separation_m=0., normals_on_collider0=[(1.,0.,0.)])
    assert s._measure()['f0']['occupied_containers'] == []
    ledger.record(own, cup, active=True, impulse_ns=0., separation_m=0., normals_on_collider0=[(0.,0.,1.)])
    assert s._measure()['f0']['occupied_containers'] == ['cupA']
