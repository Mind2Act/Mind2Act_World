"""Read actual authored container faces; no scoring padding or body writes."""
import numpy as np


class Containers:
    def __init__(self, stage):
        from pxr import Usd, UsdGeom
        self.cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy'])
        self.xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        root = '/World/Cooking/'
        self.bottom = self.bounds(stage.GetPrimAtPath(root+'PotCollision/Bottom'))
        self.walls = []
        center = (self.bottom[0]+self.bottom[1])/2
        for prim in stage.GetPrimAtPath(root+'PotCollision').GetChildren():
            if not prim.GetName().startswith('Wall'):
                continue
            mesh = UsdGeom.Mesh(prim)
            if not mesh:
                raise ValueError('CP04 actual pot wall must expose its mesh faces')
            transform = self.xforms.GetLocalToWorldTransform(prim)
            points = np.asarray([transform.Transform(tuple(p)) for p in mesh.GetPointsAttr().Get()])
            wall_center = points.mean(axis=0)
            counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
            offset = 0
            for count in counts:
                face = points[list(indices[offset:offset+count])]; offset += count
                normal = np.cross(face[1]-face[0], face[2]-face[0])
                length = np.linalg.norm(normal)
                if length == 0:
                    continue
                normal /= length
                midpoint = face.mean(axis=0)
                if normal@(midpoint-wall_center) < 0: normal = -normal
                inward = center[:2]-midpoint[:2]
                if np.linalg.norm(normal[:2]) == 0 or np.linalg.norm(inward) == 0:
                    continue
                alignment = normal[:2]@inward/(np.linalg.norm(normal[:2])*np.linalg.norm(inward))
                if alignment > .9:
                    self.walls.append((points[:,2].min(),points[:,2].max(),normal,float(normal@face[0])))
        if not self.walls:
            raise ValueError('CP04 actual pot cavity walls missing')
        self.top = max(hi for lo,hi,_,_ in self.walls)
        self.tray_lower = [self.bounds(stage.GetPrimAtPath(root+'CapTray/LeftRim'))[1][0],
                           self.bounds(stage.GetPrimAtPath(root+'CapTray/FrontRim'))[1][1]]
        self.tray_upper = [self.bounds(stage.GetPrimAtPath(root+'CapTray/RightRim'))[0][0],
                           self.bounds(stage.GetPrimAtPath(root+'CapTray/BackRim'))[0][1]]

    def bounds(self, prim):
        if not prim:
            raise ValueError('CP04 container collider missing')
        box = self.cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return np.asarray(box.GetMin()), np.asarray(box.GetMax())

    def pot_contains(self, point, *, rim=False):
        point = np.asarray(point, float).copy()
        if rim: point[2] = self.top
        z = point[2]
        if not self.bottom[1][2] <= z <= self.top:
            return False
        walls = [(normal,d) for lo,hi,normal,d in self.walls if lo <= z <= hi]
        if not walls:
            raise ValueError('CP04 pot wall height is not covered by authored geometry')
        return all(normal@point >= d for normal,d in walls)


def support_reaches(path, target, edges):
    """Follow measured upward support, including physically stacked objects."""
    pending, seen = [path], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for body, other in edges:
            if body == current or body.startswith(current+'/'):
                if other == target or other.startswith(target+'/'):
                    return True
                pending.append(other)
    return False


class DrawerSlots:
    """Authored drawer cavity, partitioned by the original jar slot centers."""
    def __init__(self,stage,slots,drawer_origin):
        from pxr import Usd,UsdGeom
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy'])
        def box(name):
            prim=stage.GetPrimAtPath('/World/Cooking/MovingDrawer/'+name)
            if not prim:raise ValueError('CP04 actual drawer boundary missing: '+name)
            b=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            return np.asarray(b.GetMin()),np.asarray(b.GetMax())
        self.lower=np.array([box('Side0')[1][0],box('Front')[1][1]])
        self.upper=np.array([box('Side2')[0][0],box('Back')[0][1]])
        self.origin=np.asarray(drawer_origin)
        self.slots={n:np.asarray(p)[:2] for n,p in slots.items()}

    def contains(self,name,center,drawer_position):
        point=np.asarray(center)[:2]-(np.asarray(drawer_position)-self.origin)[:2]
        if not (np.all(point>=self.lower) and np.all(point<=self.upper)):
            return False
        distances={n:np.linalg.norm(point-p) for n,p in self.slots.items()}
        return bool(distances[name]==min(distances.values()))


def support_directions(collider0,collider1,normal,impulse,separation):
    """Measured upward ownership; a speculative force-free gap is not support."""
    normal=np.asarray(normal,float);impulse=np.asarray(impulse,float)
    if normal.shape!=(3,) or impulse.shape!=(3,) or not np.isfinite([*normal,*impulse,separation]).all():
        raise ValueError('CP04 invalid measured support contact')
    if np.linalg.norm(impulse)==0 and separation>0:return set()
    if normal[2]>0:return {(collider0,collider1)}
    if normal[2]<0:return {(collider1,collider0)}
    return set()
