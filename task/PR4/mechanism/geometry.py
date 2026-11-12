"""Selected PR4 read-only contact/shape functions; source phase/controller checks excluded."""
import re
import numpy as np


def unexpected_pairs(rows):
    bad=set()
    for row in rows:
        for pair in row['rigid_contacts']:
            pair=[re.sub(r'/World/PR4/FingerSurfaces/(Left|Right)_(link[78])',r'/World/Robot\1/Asset/\2/collisions/Surface',p) for p in pair]
            arms=[p for p in pair if '/Robot' in p]
            if not arms:continue
            if len(arms)==1:
                if any('/Support/' in p or '/Worktable/' in p or '/CameraStand/' in p or '/Pedestal' in p for p in pair):bad.add(tuple(pair))
            elif len(arms)==2:
                side=[re.search(r'/Robot(Left|Right)/',p).group(1) for p in arms]
                links=[re.search(r'/link(\d+)(?:/|$)',p) for p in arms]
                if side[0]!=side[1]:bad.add(tuple(pair))
                elif all(links):
                    a,b=[int(x.group(1)) for x in links]
                    if a!=b and abs(a-b)>1 and {a,b}!={6,8}:bad.add(tuple(pair))
    return [list(p) for p in sorted(bad)]


def spatial_layer_order(front, rear, minimum_separation=.001):
    """Compare actual surfaces at the same XY, within their projected overlap.

    Material correspondence checks alignment separately. Comparing the heights
    of corresponding but displaced vertices is not a test of layer order.
    Use the highest rear triangle if the rear sheet itself overlaps locally.
    No convex hull or interpolated replacement mesh bridges gaps in the sheet.
    """
    a,b=rear[:-1,:-1],rear[:-1,1:]
    c,d=rear[1:,:-1],rear[1:,1:]
    triangles=np.concatenate([np.stack([a,b,c],axis=-2).reshape(-1,3,3),
                              np.stack([b,d,c],axis=-2).reshape(-1,3,3)])
    return surface_layer_order(front,triangles,minimum_separation)


def surface_layer_order(front,triangles,minimum_separation=.001):
    """Actual arbitrary material triangles at the queried XY positions."""
    triangles=np.asarray(triangles)
    queries=np.asarray(front).reshape(-1,3)
    v0=triangles[:,1,:2]-triangles[:,0,:2]
    v1=triangles[:,2,:2]-triangles[:,0,:2]
    determinant=v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0]
    valid=np.abs(determinant)>1e-12
    denominator=np.where(valid,determinant,1.)
    rear_heights=[]
    for start in range(0,len(queries),64):
        delta=queries[start:start+64,None,:2]-triangles[None,:,0,:2]
        u=(delta[:,:,0]*v1[None,:,1]-delta[:,:,1]*v1[None,:,0])/denominator
        v=(v0[None,:,0]*delta[:,:,1]-v0[None,:,1]*delta[:,:,0])/denominator
        inside=(u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&valid
        z=(triangles[None,:,0,2]+u*(triangles[None,:,1,2]-triangles[None,:,0,2])
           +v*(triangles[None,:,2,2]-triangles[None,:,0,2]))
        rear_heights.extend(np.max(np.where(inside,z,-np.inf),axis=1))
    rear_heights=np.asarray(rear_heights)
    covered=np.isfinite(rear_heights)
    separation=queries[covered,2]-rear_heights[covered]
    return dict(top_layer_fraction=float(np.mean(separation>minimum_separation)) if covered.any() else 0.,
                projected_front_overlap_fraction=float(np.mean(covered)),
                overlapping_front_samples=int(covered.sum()),front_samples=len(queries),
                geometry='actual rear material triangles at same XY; maximum rear height; layer order within overlap, material correspondence still gates alignment')


def basic_fold(points, center, width, top, nx, ny):
    """Material correspondence after release, not a commanded TCP endpoint.

    The front half must turn over onto the supported rear half. A compressed
    pile cannot pass the footprint/orientation checks; interleaved sheets
    cannot pass the layer-order check. These are initial component limits.
    """
    grid=np.asarray(points).reshape(ny,nx,3)
    half=ny//2
    front=grid[:half]
    rear=grid[:half:-1]
    errors=np.linalg.norm((front-rear)[...,:2],axis=2)
    # Exclude the two rows nearest the curved crease from the flat-layer test.
    separation=(front-rear)[:-2,:,2]
    layer_order=spatial_layer_order(front[:-2],grid[half+2:])
    flat=grid[...,2]
    xspan=float(np.quantile(grid[...,0],.99)-np.quantile(grid[...,0],.01))
    yspan=float(np.quantile(grid[...,1],.99)-np.quantile(grid[...,1],.01))
    target=(abs(grid[...,0]-center[0])<=width/2+.010)&(grid[...,1]>=center[1]-.010)&(grid[...,1]<=center[1]+width/2+.010)
    def orientation(part):
        dx=part[:-1,1:,:2]-part[:-1,:-1,:2]
        dy=part[1:,:-1,:2]-part[:-1,:-1,:2]
        return dx[...,0]*dy[...,1]-dx[...,1]*dy[...,0]
    front_flipped=float(np.mean(orientation(grid[:half-1])<0))
    rear_unflipped=float(np.mean(orientation(grid[half+2:])>0))
    metrics=dict(median_pair_xy_error_m=float(np.median(errors)),
        p90_pair_xy_error_m=float(np.quantile(errors,.9)),
        median_edge_xy_error_m=float(np.median(errors[0])),
        top_layer_fraction=layer_order['top_layer_fraction'],
        material_pair_vertical_separation_fraction=float(np.mean(separation>.001)),
        projected_front_overlap_fraction=layer_order['projected_front_overlap_fraction'],
        front_flipped_fraction=front_flipped,rear_unflipped_fraction=rear_unflipped,
        x_span_m=xspan,y_span_m=yspan,in_target_fraction=float(np.mean(target)),
        min_z_m=float(flat.min()),p95_z_m=float(np.quantile(flat,.95)))
    checks=dict(material_edges_aligned=metrics['median_edge_xy_error_m']<.015 and metrics['p90_pair_xy_error_m']<.025,
        material_halves_overlap=metrics['median_pair_xy_error_m']<.015,
        two_layers_in_correct_order=metrics['top_layer_fraction']>=.9,
        front_half_turned_over=front_flipped>=.9 and rear_unflipped>=.9,
        footprint_preserved=xspan>=width*.85 and yspan>=width*.375,
        inside_target=metrics['in_target_fraction']>=.98,
        supported_and_flat=metrics['min_z_m']>top-.003 and metrics['p95_z_m']<top+.015)
    return dict(shape_success=all(checks.values()),checks=checks,metrics=metrics,layer_order=layer_order,
        scope='original single centerline-fold material shape after release')
