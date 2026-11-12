"""Actual physics-tick geometry is distinct from an arbitrary-rotation envelope."""
import itertools
import pytest
from mind2act_eval.tasks.DR3.geometry import full_rotation_sweep, measured_aperture


def box(half):
    return list(itertools.product(*[(-v,v) for v in half]))


def gate():
    return dict(origin=[0,0,0],half_thickness_m=.007,
                aperture_low_yz=[-.15,0],aperture_high_yz=[.15,.26])


def pose(position):
    return dict(position=position,quaternion_wxyz=[1,0,0,0])


def test_stationary_far_link_is_not_measured_door_occupancy():
    # The native link2 counterexample has this large-radius/far-X pattern.
    vertices=box([.03325,.28,.02]);state=pose([-.143,-.325,.0845])
    assert full_rotation_sweep(vertices,state,state,gate())['corridor_intersects']
    result=measured_aperture(vertices,state,gate(),actual_vertices=True)
    assert result['enclosure_slab_intersects'] is False
    assert result['corridor_occupied'] is False
    assert result['aperture_safe'] is True
    assert 'continuous' not in result


def test_actual_point_in_closed_corridor_is_preserved_as_occupancy():
    result=measured_aperture(box([.005]*3),pose([0,0,.1]),gate(),actual_vertices=True)
    assert result['corridor_occupied'] is True
    assert result['aperture_safe'] is True


def test_measured_outside_aperture_point_is_a_real_violation_witness():
    result=measured_aperture(box([.005]*3),pose([0,.2,.1]),gate(),actual_vertices=True)
    assert result['aperture_safe'] is False
    assert result['outside_aperture_point'] is True
    assert result['corridor_occupied'] is False


def test_outer_hull_overlap_without_a_real_point_is_not_invented_occupancy():
    vertices=box([.005,.2,.005])
    result=measured_aperture(vertices,pose([0,0,.1]),gate(),actual_vertices=True)
    assert result['corridor_occupied'] is None
    assert result['aperture_safe'] is False
    assert result['outside_aperture_point'] is True


def test_enclosure_vertices_are_not_automatically_actual_collision_points():
    result=measured_aperture(box([.005]*3),pose([0,0,.1]),gate(),actual_vertices=False)
    assert result['corridor_occupied'] is None
    with pytest.raises(ValueError):
        measured_aperture([],pose([0,0,.1]),gate(),actual_vertices=True)


def test_declared_convex_shape_resolves_occupancy_between_its_vertices():
    vertices=box([.005,.2,.005])
    result=measured_aperture(vertices,pose([0,0,.1]),gate(),actual_vertices=True,
                             collision_parts=[dict(vertices=vertices,approximation='convexHull')])
    assert result['corridor_occupied'] is True


def test_separate_real_collision_shapes_do_not_fill_their_empty_gap():
    parts=[];vertices=[]
    for y in (-.2,.2):
        points=[[x,py+y,z] for x,py,z in box([.005]*3)]
        vertices.extend(points);parts.append(dict(vertices=points,approximation='convexHull'))
    result=measured_aperture(vertices,pose([0,0,.1]),gate(),actual_vertices=True,collision_parts=parts)
    assert result['corridor_occupied'] is False


def test_transition_preserves_unresolved_occupancy_as_none():
    from mind2act_eval.tasks.DR3.geometry import measured_physics_transition
    result=measured_physics_transition(box([.005,.2,.005]),pose([0,0,.1]),pose([0,0,.1]),gate(),actual_vertices=True)
    assert result['corridor_intersects'] is None
