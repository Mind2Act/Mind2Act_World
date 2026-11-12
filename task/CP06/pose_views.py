"""Existing non-root robot frames through explicit read-only Isaac5.1 views."""

def flanges(stage,world,*,view_type=None,non_root=None):
    from pxr import UsdPhysics
    if view_type is None:
        from isaacsim.core.prims import RigidPrim
        view_type=RigidPrim
    if non_root is None:
        from isaacsim.core.utils.prims import is_prim_non_root_articulation_link
        non_root=is_prim_non_root_articulation_link
    views={}
    for side in ('Left','Right'):
        path='/World/Robot'+side+'/Asset/link6';prim=stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or not non_root(prim_path=path):
            raise ValueError('CP06 read-only flange requires an existing non-root rigid link')
        views[side]=world.scene.add(view_type(prim_paths_expr=path,name='CP06Flange'+side,
            reset_xform_properties=False,prepare_contact_sensors=False,track_contact_forces=False,
            disable_stablization=False,contact_filter_prim_paths_expr=[]))
    return views
