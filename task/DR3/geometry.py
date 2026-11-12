# Source: fr3-replay-new15-20261031 / detector_geometry.py; see config/source.json.
"""Conservative collision-envelope certificates; uncertainty is never a legal pass.

Coordinates are metres in the frozen gate frame. An enclosure can prove safety.
An enclosure crossing a boundary does not prove a violation of the actual shape.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

MEASUREMENT_CONTRACT = 'dr3-measured-physics-substeps-v1'


def finite_array(value, shape=None):
    out = np.asarray(value, dtype=float)
    if shape is not None and out.shape != shape:
        raise ValueError("wrong geometry dimensions")
    if not np.isfinite(out).all():
        raise ValueError("nonfinite geometry")
    return out


def transform_vertices(vertices, pose):
    vertices = finite_array(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("empty collision enclosure")
    p = finite_array(pose["position"], (3,))
    q = finite_array(pose["quaternion_wxyz"], (4,))
    if abs(np.linalg.norm(q) - 1.) > 1e-5:
        raise ValueError("invalid rigid orientation")
    return vertices @ Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix().T + p


def slab_projection(vertices, low, high):
    """Enclose every point of a convex collision envelope inside a gate slab."""
    points = finite_array(vertices)
    if len(points) < 4:
        raise ValueError("incomplete three-dimensional collision enclosure")
    if np.max(points[:,0]) < low or np.min(points[:,0]) > high:
        return np.empty((0,3))
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        raise ValueError("degenerate collision enclosure") from exc
    result = [p for p in points if low <= p[0] <= high]
    edges = {tuple(sorted((int(face[i]), int(face[j]))))
             for face in hull.simplices for i in range(3) for j in range(i)}
    for i, j in edges:
        a, b = points[i], points[j]
        if a[0] == b[0]:
            continue
        for x in (low, high):
            fraction = (x - a[0]) / (b[0] - a[0])
            if 0 <= fraction <= 1:
                result.append(a + fraction * (b - a))
    return np.asarray(result).reshape((-1, 3))


def corridor_intersection(projection, low, high):
    """Intersect the projected convex envelope with the finite gate opening."""
    if not len(projection):return False
    points=np.unique(projection[:,1:],axis=0)
    try:polygon=list(points[ConvexHull(points).vertices])
    except QhullError:
        return bool(np.all(points.max(axis=0)>=low) and np.all(points.min(axis=0)<=high))
    for axis in range(2):
        for bound,above in ((low[axis],True),(high[axis],False)):
            if not polygon:return False
            out=[];previous=polygon[-1]
            previous_in=previous[axis]>=bound if above else previous[axis]<=bound
            for current in polygon:
                current_in=current[axis]>=bound if above else current[axis]<=bound
                if current_in!=previous_in:
                    fraction=(bound-previous[axis])/(current[axis]-previous[axis])
                    out.append(previous+fraction*(current-previous))
                if current_in:out.append(current)
                previous,previous_in=current,current_in
            polygon=out
    return bool(polygon)


def swept_aperture(vertices, before, after, gate, certificate, *, allow_test_certificates=False):
    """Evaluate the source-pinned native envelope or a bounded replay interval.

    Native capture selects the implemented full-rotation model. Test-only
    analytic motion records remain opt-in and cannot activate production scores.
    """
    if isinstance(certificate,dict) and certificate.get("contract")=="bounded-rigid-envelope-v1" and certificate.get("source")=="native_pose_full_rotation":
        return full_rotation_sweep(vertices,before,after,gate,actual_vertices=certificate.get("actual_collision_vertices") is True)
    if not isinstance(certificate, dict) or certificate.get("contract") != "bounded-rigid-envelope-v1":
        return dict(status="UNKNOWN", reason="missing authoritative interval sweep bound")
    endpoint_only = certificate.get("source") == "native_endpoint_enclosure"
    if not endpoint_only and certificate.get("source") != "native_physics_substeps" and not (allow_test_certificates and certificate.get("source") == "analytic_test_fixture"):
        return dict(status="UNKNOWN", reason="unqualified sweep source")
    error = 0. if endpoint_only else certificate.get("deviation_bound_m")
    if type(error) not in (int, float) or not np.isfinite(error) or error < 0:
        raise ValueError("invalid sweep deviation bound")
    a, b = transform_vertices(vertices, before), transform_vertices(vertices, after)
    origin = finite_array(gate["origin"], (3,))
    # Both end geometries and the supplied Hausdorff deviation bound enclose the
    # full swept trajectory. A mere endpoint distance is not a deviation bound.
    points = np.concatenate((a, b)) - origin
    radius = float(gate["half_thickness_m"])
    projection = slab_projection(points, -radius-error, radius+error)
    if not len(projection):
        return dict(status="UNKNOWN" if endpoint_only else "SAFE", intersects=False,corridor_intersects=False,
                    reason="endpoint-only trajectory lacks an intra-step motion bound" if endpoint_only else None)
    low, high = finite_array(gate["aperture_low_yz"], (2,)), finite_array(gate["aperture_high_yz"], (2,))
    corridor=corridor_intersection(projection,low-error,high+error)
    safe = bool(np.all(projection[:, 1:] - error > low) and np.all(projection[:, 1:] + error < high))
    qa = finite_array(before["quaternion_wxyz"], (4,))
    qb = finite_array(after["quaternion_wxyz"], (4,))
    constant_rotation = abs(abs(float(np.dot(qa, qb)))-1.) < 1e-8
    if endpoint_only:
        return dict(status="UNKNOWN",intersects=True,corridor_intersects=corridor,candidate_aperture_safe=safe,
                    reason="endpoint-only trajectory lacks an intra-step motion bound")
    exact = bool(error == 0 and constant_rotation and certificate.get("exact_convex_shape") is True)
    return dict(status="SAFE" if safe else "VIOLATION" if exact else "UNKNOWN", intersects=True,corridor_intersects=corridor,
                reason=None if safe else "swept geometry crosses aperture boundary" if exact else "swept enclosure intersects aperture boundary")


def side(vertices, pose, gate, clearance=0.):
    points = transform_vertices(vertices, pose)
    x = points[:, 0] - float(gate["origin"][0])
    if np.max(x) <= -clearance:
        return "OUTSIDE"
    if np.min(x) >= clearance:
        return "INSIDE"
    return "STRADDLING"


def clear_outside(vertices, pose, gate, clearance=.025):
    return bool(np.max(transform_vertices(vertices, pose)[:, 0]) <= float(gate["origin"][0])-clearance)


def measured_aperture(vertices, pose, gate, *, actual_vertices=False, collision_parts=None, _convex_part=False):
    """Read one measured pose without asserting any motion between ticks.

    An empty convex enclosure proves absence. An actual collision vertex in
    the corridor proves occupancy. An enclosure-only overlap remains None;
    a convex hull can bridge empty space between separate collision shapes.
    This observation is not an interval certificate or a legal-pass verdict.
    """
    if collision_parts is not None:
        if not collision_parts:raise ValueError('empty measured collision shape inventory')
        observations=[]
        for part in collision_parts:
            if part.get('approximation')!='convexHull':raise ValueError('unsupported measured collision shape')
            observations.append(measured_aperture(part['vertices'],pose,gate,
                actual_vertices=actual_vertices,_convex_part=True))
        occupied=True if any(o['corridor_occupied'] is True for o in observations) else False if all(o['corridor_occupied'] is False for o in observations) else None
        return dict(enclosure_slab_intersects=any(o['enclosure_slab_intersects'] for o in observations),
                    corridor_occupied=occupied,aperture_safe=all(o['aperture_safe'] for o in observations),
                    outside_aperture_point=any(o['outside_aperture_point'] is True for o in observations))
    points=transform_vertices(vertices,pose)-finite_array(gate['origin'],(3,))
    half=float(gate['half_thickness_m'])
    low=finite_array(gate['aperture_low_yz'],(2,))
    high=finite_array(gate['aperture_high_yz'],(2,))
    projection=slab_projection(points,-half,half)
    overlap=corridor_intersection(projection,low,high)
    safe=bool(not len(projection) or (np.all(projection[:,1:]>low) and np.all(projection[:,1:]<high)))
    in_slab=np.abs(points[:,0])<=half
    in_corridor=np.all(points[:,1:]>=low,axis=1)&np.all(points[:,1:]<=high,axis=1)
    outside=np.any((points[:,1:]<low)|(points[:,1:]>high),axis=1)
    occupied=bool(overlap) if _convex_part else False if not overlap else True if actual_vertices is True and np.any(in_slab&in_corridor) else None
    outside_point=bool(len(projection) and not safe) if _convex_part else True if actual_vertices is True and np.any(in_slab&outside) else False if safe or actual_vertices is True else None
    return dict(enclosure_slab_intersects=bool(len(projection)),corridor_occupied=occupied,
                aperture_safe=safe,outside_aperture_point=outside_point)


def measured_physics_transition(vertices, before, after, gate, *, actual_vertices=False, collision_parts=None):
    """Compare actual consecutive physics substeps, never interpolate a pose.

    This implements the 120 Hz measured-substep contract, not an intra-tick
    continuous trajectory certificate. Whole geometry and actual aperture
    samples determine passage; an arbitrary-rotation sphere is not consulted.
    """
    a=measured_aperture(vertices,before,gate,actual_vertices=actual_vertices,collision_parts=collision_parts)
    b=measured_aperture(vertices,after,gate,actual_vertices=actual_vertices,collision_parts=collision_parts)
    crosses={side(vertices,before,gate),side(vertices,after,gate)}=={'OUTSIDE','INSIDE'}
    safe=a['aperture_safe'] and b['aperture_safe']
    outside=a['outside_aperture_point'] is True or b['outside_aperture_point'] is True
    if crosses and not a['enclosure_slab_intersects'] and not b['enclosure_slab_intersects']:
        # Crossing between two real physics samples still examines the entire
        # sampled shapes, rather than only their origins. No path is generated.
        points=np.concatenate([transform_vertices(vertices,p) for p in (before,after)])
        yz=points[:,1:]-finite_array(gate['origin'],(3,))[1:]
        low=finite_array(gate['aperture_low_yz'],(2,));high=finite_array(gate['aperture_high_yz'],(2,))
        safe=bool(np.all(yz>low) and np.all(yz<high))
        outside=bool(actual_vertices is True and not safe)
    occupied=True if a['corridor_occupied'] is True or b['corridor_occupied'] is True or (crosses and safe) else False if a['corridor_occupied'] is False and b['corridor_occupied'] is False else None
    return dict(status='SAFE' if safe else 'VIOLATION' if outside else 'UNKNOWN',
        intersects=bool(crosses or a['enclosure_slab_intersects'] or b['enclosure_slab_intersects']),
        corridor_intersects=occupied,
        before_observation=a,after_observation=b,crosses_sampled_sides=crosses,
        reason=None if safe else 'measured collision points outside aperture' if outside else 'collision topology is insufficient to resolve measured aperture overlap')


def floor_region_covered(triangles, center_xy, half_xy, floor_z, tolerance=1e-5):
    """Subtract the actual coplanar triangle union from the required rectangle.

    Rounded source liners have a many-triangle fan. No one/two-face restriction
    or convex-hull shortcut can establish the coverage of that tessellation.
    """
    triangles=finite_array(triangles)
    center=finite_array(center_xy,(2,));half=finite_array(half_xy,(2,))
    floor=[t[:,:2] for t in triangles if np.all(np.abs(t[:,2]-floor_z)<=tolerance)]
    remaining=[np.asarray([center+np.asarray([x,y])*half for x,y in ((-1,-1),(1,-1),(1,1),(-1,1))])]
    def area(poly):
        if len(poly)<3:return 0.
        return abs(float(np.dot(poly[:,0],np.roll(poly[:,1],-1))-np.dot(poly[:,1],np.roll(poly[:,0],-1))))/2
    def clip(poly,normal,bound,inside):
        output=[]
        if not len(poly):return np.empty((0,2))
        prev=poly[-1];dp=float(prev@normal-bound)
        for current in poly:
            dc=float(current@normal-bound)
            p_in=dp<=0 if inside else dp>=0
            c_in=dc<=0 if inside else dc>=0
            if p_in!=c_in:output.append(prev+(current-prev)*(dp/(dp-dc)))
            if c_in:output.append(current)
            prev,dp=current,dc
        return np.asarray(output).reshape(-1,2)
    for triangle in floor:
        try:hull=ConvexHull(triangle)
        except QhullError:continue
        next_remaining=[]
        for poly in remaining:
            overlap=poly
            for nx,ny,offset in hull.equations:
                normal=np.asarray([nx,ny]);bound=-offset+tolerance
                outside=clip(overlap,normal,bound,False)
                if area(outside)>tolerance**2:next_remaining.append(outside)
                overlap=clip(overlap,normal,bound,True)
                if not len(overlap):break
        remaining=next_remaining
        if not remaining:return True
    return not remaining

def full_rotation_sweep(vertices, before, after, gate, *, actual_vertices=False):
    """Frozen per-physics-step origin-segment sweep with arbitrary rotation.

    The sphere encloses the entire collider in every orientation. Boundary
    overlap is uncertainty, rather than proof that the actual mesh bypassed.
    """
    local=finite_array(vertices)
    radius=float(np.max(np.linalg.norm(local,axis=1)))
    a=finite_array(before["position"],(3,))-finite_array(gate["origin"],(3,))
    b=finite_array(after["position"],(3,))-finite_array(gate["origin"],(3,))
    half=float(gate["half_thickness_m"])
    low_x,high_x=-half-radius,half+radius
    delta=b[0]-a[0]
    if delta==0:
        if a[0]<low_x or a[0]>high_x:
            return dict(status="SAFE",intersects=False,corridor_intersects=False)
        first,last=0.,1.
    else:
        cuts=sorted(((low_x-a[0])/delta,(high_x-a[0])/delta))
        first,last=max(0.,cuts[0]),min(1.,cuts[1])
        if first>last:
            return dict(status="SAFE",intersects=False,corridor_intersects=False)
    actual_before=transform_vertices(vertices,before)-finite_array(gate["origin"],(3,))
    actual_after=transform_vertices(vertices,after)-finite_array(gate["origin"],(3,))
    definite_before=bool(actual_vertices and np.any(np.abs(actual_before[:,0])<=half))
    definite_after=bool(actual_vertices and np.any(np.abs(actual_after[:,0])<=half))
    centers=np.array([a+first*(b-a),a+last*(b-a)])
    lower=centers[:,1:].min(axis=0)-radius
    upper=centers[:,1:].max(axis=0)+radius
    low=finite_array(gate["aperture_low_yz"],(2,))
    high=finite_array(gate["aperture_high_yz"],(2,))
    corridor=bool(np.all(upper>=low) and np.all(lower<=high))
    safe=bool(np.all(lower>low) and np.all(upper<high))
    return dict(status="SAFE" if safe else "UNKNOWN",intersects=True,corridor_intersects=corridor,
                reason=None if safe else "full-rotation collider enclosure overlaps aperture boundary",
                envelope_radius_m=radius,definite_before=definite_before,definite_after=definite_after)
