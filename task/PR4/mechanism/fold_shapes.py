"""Read-only shape checks for diagonal and consecutive two-fold cloth designs.

Material correspondence, actual face orientation, projected layer order and
fixed target geometry are checked separately. This never proves a grasp or
replaces review of the release and withdrawal recording.
"""
import numpy as np
from .geometry import surface_layer_order
from . import fold_layout


def _finish(points,pair_errors,edge_errors,layers,orientation,inside,x_min,y_min,top,scope):
    p=np.asarray(points);errors=np.concatenate([np.ravel(x) for x in pair_errors])
    xspan,yspan=np.quantile(p[:,:2],.99,axis=0)-np.quantile(p[:,:2],.01,axis=0)
    m=dict(median_pair_xy_error_m=float(np.median(errors)),p90_pair_xy_error_m=float(np.quantile(errors,.9)),
        median_edge_xy_error_m=float(np.median(np.concatenate([np.ravel(x) for x in edge_errors]))),
        top_layer_fraction=min(x['top_layer_fraction'] for x in layers),
        projected_front_overlap_fraction=min(x['projected_front_overlap_fraction'] for x in layers),
        orientation_fractions=orientation,x_span_m=float(xspan),y_span_m=float(yspan),
        in_target_fraction=float(np.mean(inside)),min_z_m=float(p[:,2].min()),p95_z_m=float(np.quantile(p[:,2],.95)))
    m['per_material_part_pair_errors']=[dict(median_m=float(np.median(x)),p90_m=float(np.quantile(x,.9))) for x in pair_errors]
    m['per_material_part_edge_median_m']=[float(np.median(x)) for x in edge_errors]
    checks=dict(material_edges_aligned=all(x<.015 for x in m['per_material_part_edge_median_m']) and all(x['p90_m']<.025 for x in m['per_material_part_pair_errors']),
        material_halves_overlap=all(x['median_m']<.015 for x in m['per_material_part_pair_errors']),
        layers_in_correct_order=m['top_layer_fraction']>=.9,
        material_parts_turned_over=min(orientation)>=.9,
        footprint_preserved=xspan>=x_min and yspan>=y_min,inside_target=m['in_target_fraction']>=.98,
        supported_and_flat=m['min_z_m']>top-.003 and m['p95_z_m']<top+.015)
    checks={name:bool(value) for name,value in checks.items()}
    return dict(shape_success=all(checks.values()),checks=checks,metrics=m,layer_order=layers,scope=scope)


def diagonal(points,center,width,top,n,triangles):
    p=np.asarray(points);triangles=np.asarray(triangles);layout=fold_layout.diagonal(n)
    row,col=np.indices((n,n));distance=(row+col-(n-1)).ravel()
    moving=layout['moving_ids'];partners=layout['partner_ids']
    errors=np.linalg.norm(p[moving,:2]-p[partners,:2],axis=1)
    edges=((moving//n==0)|(moving%n==0))
    back_tri=triangles[(distance[triangles]>2).all(1)]
    layers=[surface_layer_order(p[distance<-3],p[back_tri])]
    faces=p[triangles];normal=np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0])[:,2]
    orientation=[float(np.mean(normal[(distance[triangles]<-2).all(1)]<0)),
        float(np.mean(normal[(distance[triangles]>2).all(1)]>0))]
    radius=width/np.sqrt(2);x=p[:,0]-center[0];y=p[:,1]-center[1]
    inside=(y>=-.010)&(abs(x)+y<=radius+.010)
    result=_finish(p,[errors],[errors[edges]],layers,orientation,inside,2*radius*.85,radius*.75,top,
        'single diagonal fold; immutable reflected material pairs and actual triangular layer surfaces')
    corner_error=float(np.linalg.norm(p[0,:2]-p[-1,:2]))
    result['metrics']['opposite_corner_xy_error_m']=corner_error
    result['checks']['opposite_corners_aligned']=corner_error<.015
    result['shape_success']=all(result['checks'].values())
    return result


def double(points,center,width,top,n,direction='rear_hems_backward'):
    from .geometry import spatial_layer_order
    p=np.asarray(points);layers=fold_layout.quarter_layers(n,direction);parts=[p[x] for x in layers]
    pairs=[np.linalg.norm((x-parts[0])[...,:2],axis=2) for x in parts[1:]]
    edge_errors=[np.r_[e[0],e[-1],e[:,0],e[:,-1]] for e in pairs]
    orders=[spatial_layer_order(parts[i+1][2:-2],parts[i][2:-2]) for i in range(3)]
    orientation=[]
    # Reversing a quarter's row indexing changes its mesh orientation. Check
    # in original material row order rather than the correspondence array.
    for ids,sign in zip(layers,[1,-1,1,-1]):
        rows=np.sort(ids[:,0]//n);part=p.reshape(n,n,3)[rows[2:-2]]
        a=part[:-1,1:,:2]-part[:-1,:-1,:2];b=part[1:,:-1,:2]-part[:-1,:-1,:2]
        orientation.append(float(np.mean(sign*(a[...,0]*b[...,1]-a[...,1]*b[...,0])>0)))
    x=p[:,0]-center[0];y=p[:,1]-center[1]
    low=width/4 if direction=='folded_edge_forward' else 0.
    inside=(abs(x)<=width/2+.010)&(y>=low-.010)&(y<=low+width/4+.010)
    return _finish(p,pairs,edge_errors,orders,orientation,inside,width*.85,width*.1875,top,
        'forward half-fold then '+direction+'; all four material quarters, adjacent layer order and fixed quarter target')
