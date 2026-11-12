"""Apply the conveyor parent transform used by selected CP01 training replays."""

import math


def apply_training_conveyor_layout(stage, setup):
    """Normalize the whole authored conveyor once, including rails and legs.

    Matching only OriginalBelt or the invisible collision surface is insufficient.
    The selected training replay normalizes the authored 0.30 m belt centered at
    y=0.145 through its parent; inner asset transforms must remain unchanged.
    Repeated calls replace the same parent ops rather than multiplying them.
    """
    from pxr import Gf, UsdGeom

    width = float(setup["belt_width_m"])
    center_y = float(setup["belt_y_m"])
    if not math.isfinite(width) or width <= 0 or not math.isfinite(center_y):
        raise ValueError("Invalid CP01 conveyor width or center")
    conveyor = stage.GetPrimAtPath("/World/CP01/Conveyor")
    if not conveyor:
        raise ValueError("Missing CP01 conveyor visual asset")
    conveyor.SetTypeName("Xform")
    xf = UsdGeom.Xformable(conveyor)
    xf.ClearXformOpOrder()
    scale = width / 0.30
    translation = (0.0, center_y - 0.145 * scale, 0.0)
    translate_attr = conveyor.GetAttribute("xformOp:translate:collectionLayout")
    translate_op = (UsdGeom.XformOp(translate_attr) if translate_attr else
                    xf.AddTranslateOp(opSuffix="collectionLayout"))
    scale_attr = conveyor.GetAttribute("xformOp:scale:collectionLayout")
    scale_op = (UsdGeom.XformOp(scale_attr) if scale_attr else
                xf.AddScaleOp(opSuffix="collectionLayout"))
    translate_op.Set(Gf.Vec3d(*translation))
    scale_op.Set(Gf.Vec3f(1.0, scale, 1.0))
    xf.SetXformOpOrder([translate_op, scale_op])
    center = UsdGeom.XformCache().GetLocalToWorldTransform(conveyor).Transform(
        Gf.Vec3d(0.0, 0.145, 0.858)
    )
    if abs(float(center[1]) - center_y) > 1e-6:
        raise ValueError("CP01 conveyor visual/physical center mismatch")
    return {
        "prim": str(conveyor.GetPath()),
        "belt_width_m": width,
        "center_y_m": center_y,
        "visual_center_y_m": float(center[1]),
        "translate_collectionLayout_m": list(translation),
        "scale_collectionLayout": [1.0, scale, 1.0],
    }
