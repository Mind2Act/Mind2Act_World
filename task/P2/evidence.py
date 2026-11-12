"""Task-local native measurement subscription and sealed streaming evidence."""
import hashlib
import json
from pathlib import Path


class Rows:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('xb')
        self.count,self.size=0,0
        self.digest=hashlib.sha256()

    def append(self,row):
        if row['physics_tick'] != self.count+1:
            raise ValueError('P2 native measurement gap')
        raw=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
        self.stream.write(raw);self.digest.update(raw)
        self.count+=1;self.size+=len(raw)

    def receipt(self):
        self.stream.flush()
        return dict(schema='p2.native-measurements.v1',path=str(self.path),sha256=self.digest.hexdigest(),
                    rows=self.count,bytes=self.size)

    def close(self):
        self.stream.close()


def read_rows(receipt):
    if receipt['schema'] != 'p2.native-measurements.v1':
        raise ValueError('P2 native measurement schema mismatch')
    digest,size,count=hashlib.sha256(),0,0
    with Path(receipt['path']).open('rb') as stream:
        for raw in stream:
            digest.update(raw);size+=len(raw);count+=1
            yield json.loads(raw)
    if (digest.hexdigest(),size,count)!=(receipt['sha256'],receipt['bytes'],receipt['rows']):
        raise ValueError('P2 native measurement SHA/size/count mismatch')


class Manifolds:
    def __init__(self):
        self.pairs={}
        self.subscription=None

    def subscribe(self):
        from pxr import PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface
        from omni.physx.bindings._physx import ContactEventType

        def report(headers,data):
            for h in headers:
                ordered=tuple(str(PhysicsSchemaTools.intToSdfPath(p)) for p in (h.collider0,h.collider1))
                pair=tuple(sorted(ordered))
                if not any(p.startswith('/World/Task/') for p in pair):
                    continue
                if h.type==ContactEventType.CONTACT_LOST:
                    self.pairs.pop(pair,None)
                    continue
                points=data[h.contact_data_offset:h.contact_data_offset+h.num_contact_data]
                if points:
                    sign=1 if pair[0]==ordered[0] else -1
                    self.pairs[pair]=[dict(position=list(map(float,p.position)),
                        normal=[sign*float(v) for v in p.normal],impulse=list(map(float,p.impulse)),
                        separation=float(p.separation)) for p in points]
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(report)

    def snapshot(self):
        return [dict(pair=list(pair),active=True,points=points) for pair,points in sorted(self.pairs.items())]

    def close(self):
        if self.subscription is not None:
            self.subscription.unsubscribe()
            self.subscription=None


def measure(tick,bodies,robots,contacts,world,stage):
    from pxr import Usd,UsdGeom
    poses={}
    for name,body in bodies.items():
        p,q=body.get_world_pose()
        poses[name]=dict(position=p.tolist(),quaternion_wxyz=q.tolist(),
            linear_velocity_m_s=body.get_linear_velocity().tolist(),
            angular_velocity_rad_s=body.get_angular_velocity().tolist())
    base=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Task/Stove/PreplacedTier'))
    return dict(physics_tick=tick,simulation_time_s=float(world.current_time),poses=poses,
        preplaced_xyz=list(map(float,base.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())),
        q={side:r.get_joint_positions().tolist() for side,r in robots.items()},
        qd={side:r.get_joint_velocities().tolist() for side,r in robots.items()},contacts=contacts.snapshot())
