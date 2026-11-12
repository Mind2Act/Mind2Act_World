"""Actual static rack footprint and actual mallet center; no target inset."""
import numpy as np


class RackGeometry:
    def __init__(self,stage,sides):
        from pxr import Usd,UsdGeom,UsdPhysics
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
        self.bounds={};self.centers={}
        for side in sides:
            root='/World/CP12/Tools/'+side;body=root+'/Mallet'
            boxes=[]
            for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
                if str(p.GetPath()).startswith(body+'/') or str(p.GetPath())==body:
                    continue
                if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False:
                    b=cache.ComputeWorldBound(p).ComputeAlignedRange()
                    if not b.IsEmpty():boxes.append((np.asarray(b.GetMin()),np.asarray(b.GetMax())))
            if not boxes:raise ValueError('CP05 actual rack colliders missing')
            self.bounds[side]=(np.min([b[0] for b in boxes],axis=0),np.max([b[1] for b in boxes],axis=0))
            b=cache.ComputeUntransformedBound(stage.GetPrimAtPath(body)).ComputeAlignedRange()
            if b.IsEmpty():raise ValueError('CP05 actual mallet geometry missing')
            self.centers[side]=np.asarray(b.GetMidpoint())

    def contains(self,side,position,quaternion):
        from scipy.spatial.transform import Rotation
        q=np.asarray(quaternion)
        center=np.asarray(position)+Rotation.from_quat(q[[1,2,3,0]]).apply(self.centers[side])
        lo,hi=self.bounds[side]
        return bool(np.all(center[:2]>=lo[:2]) and np.all(center[:2]<=hi[:2]))
