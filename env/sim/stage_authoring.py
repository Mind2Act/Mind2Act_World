"""Author episode edits over immutable station sublayers."""


def materialize_local_layer_stack(stage):
    """Make inherited station specs replaceable in the episode root, in memory.

    RemovePrim only erases specs in the edit target. Consolidate local sublayers
    before replacing whole subtrees so frozen station opinions cannot survive
    underneath a new asset reference. References/payloads remain composition
    arcs; neither their contents nor any source layer on disk are rewritten.
    """
    from pxr import UsdUtils
    root = stage.GetRootLayer()
    if root.subLayerPaths:
        root.TransferContent(UsdUtils.FlattenLayerStack(stage))


def copy_local_prim(stage, source, destination):
    """Copy the current local layer-stack opinions, preserving composition arcs.

    A thin episode root does not contain inherited specs. Flatten only its local
    layer stack into an anonymous snapshot, then copy the requested subtree into
    the episode root. Referenced assets and the shared projection stay untouched.
    Recompute on each copy so current runtime overrides/prototypes are included.
    """
    from pxr import Sdf, UsdUtils
    source, destination = Sdf.Path(source), Sdf.Path(destination)
    snapshot = UsdUtils.FlattenLayerStack(stage)
    if not snapshot.GetPrimAtPath(source):
        raise ValueError('No local station prim to copy: ' + str(source))
    if not Sdf.CopySpec(snapshot, source, stage.GetRootLayer(), destination):
        raise RuntimeError('Cannot copy station prim: ' + str(source))


def remove_composed_prim(stage, path):
    """Remove owned specs and suppress any weaker inherited prim in this episode."""
    from pxr import Usd
    with Usd.EditContext(stage, stage.GetRootLayer()):
        stage.RemovePrim(path)
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsActive():
            stage.OverridePrim(path).SetActive(False)
