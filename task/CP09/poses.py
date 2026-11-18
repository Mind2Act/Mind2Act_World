"""World TCP from the actual link6 frame; source X5Kinematics.fk line32."""
import numpy as np
from scipy.spatial.transform import Rotation

SOURCE_TCP_IN_LINK6_M = (.145, 0., 0.)


def measured_tcp(position, orientation_wxyz):
    p=np.asarray(position,dtype=float);q=np.asarray(orientation_wxyz,dtype=float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:
        raise ValueError('CP09 measured flange frame invalid')
    return (p+Rotation.from_quat(q[[1,2,3,0]]).apply(SOURCE_TCP_IN_LINK6_M)).tolist()


def flange_readers(stage, world, *, view_type=None, non_root=None):
    """Rigid tensor pose views with every SDK authoring side effect disabled.

    Isaac5.1 RigidPrim._apply_rigid_body_apis only reads an existing API when
    prepare_contact_sensors and track_contact_forces are both False. Non-root
    articulation links also skip SDK post-reset pose/velocity writes.
    """
    from pxr import UsdPhysics
    if view_type is None:
        from isaacsim.core.prims import RigidPrim
        view_type=RigidPrim
    if non_root is None:
        from isaacsim.core.utils.prims import is_prim_non_root_articulation_link
        non_root=is_prim_non_root_articulation_link
    views={}
    for side in ('Left','Right'):
        path='/World/Robot'+side+'/Asset/link6'
        prim=stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not non_root(prim_path=path):
            raise ValueError('CP09 flange reader requires an existing non-root rigid articulation link')
        views[side]=world.scene.add(view_type(prim_paths_expr=path,name='CP09Flange'+side,
            reset_xform_properties=False,prepare_contact_sensors=False,
            track_contact_forces=False,disable_stablization=False,
            contact_filter_prim_paths_expr=[]))
    return views
