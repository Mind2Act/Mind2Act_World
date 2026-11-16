"""Raw native FEM points and measurements, written once per physical tick."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from mind2act_eval.contracts import NativeEvidence,to_json

COUNT=6561
POINT_BYTES=COUNT*3*4


def vector(value,n):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('PR4 missing/nonfinite native vector')
    return a


def rotation(q):
    q=vector(q,4)
    if abs(np.linalg.norm(q)-1)>1e-4:raise ValueError('PR4 nonunit native quaternion')
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def contact_envelopes(row,points,hulls,hem):
    """Original signed collision-plane envelope, recomputed from raw poses."""
    out={}
    for key,body in row['finger_poses'].items():
        local=(points[hem]-body['position'])@rotation(body['quaternion'])
        distance=np.min([np.max(local@np.asarray(h)[:,:3].T+np.asarray(h)[:,3],axis=1) for h in hulls[key]],axis=0)
        out[key]=dict(body,hem_contact_ids=hem[(distance>=-.001)&(distance<=.002)].tolist(),
                      min_signed_plane_distance_m=float(distance.min()))
    return out


def validate_row(row,tick):
    required={'tick','time_s','joints','tcp','finger_poses','rigid_contact_impulses_ns','render_physics_error_m'}
    if set(row) not in (required,required|{'point_source'}) or row['tick']!=tick or type(row['tick']) is not int:
        raise ValueError('PR4 requires each consecutive native tick without phase labels')
    vector([row['time_s']],1)
    if row['time_s']<0 or set(row['joints'])!={'Left','Right'} or set(row['tcp'])!={'Left','Right'}:
        raise ValueError('PR4 missing native arms/clock')
    for s in ('Left','Right'):vector(row['joints'][s],8);vector(row['tcp'][s],3)
    if set(row['finger_poses'])!={s+'/link'+str(j) for s in ('Left','Right') for j in (7,8)}:
        raise ValueError('PR4 missing native finger bodies')
    for body in row['finger_poses'].values():
        if set(body)!={'position','quaternion'}:raise ValueError('PR4 invalid native finger schema')
        vector(body['position'],3);rotation(body['quaternion'])
    source=row.get('point_source')
    if source is not None and source!='native_surface_simulation_mesh_usd':
        raise ValueError('PR4 unsupported native point source')
    if row['render_physics_error_m'] is None:
        if source!='native_surface_simulation_mesh_usd':
            raise ValueError('PR4 unavailable tensor comparison requires the native surface mesh source')
    else:
        vector([row['render_physics_error_m']],1)
        if row['render_physics_error_m'] < 0:
            raise ValueError('PR4 invalid negative diagnostic render/FEM error')
    seen=set()
    for c in row['rigid_contact_impulses_ns']:
        if set(c)!={'pair','impulse'} or len(c['pair'])!=2:raise ValueError('PR4 invalid rigid contact measurement')
        pair=tuple(c['pair'])
        if pair!=tuple(sorted(pair)) or pair in seen or any(not isinstance(p,str) or not p.startswith('/World/') for p in pair):
            raise ValueError('PR4 invalid/duplicate rigid contact pair')
        seen.add(pair);vector(c['impulse'],3)


class Recorder:
    def __init__(self,instance_id,case,initial,settled,triangles,hulls,output_dir,support_geometry=None, *, strict=False):
        self.instance_id,self.case=instance_id,to_json(case)
        self.initial=np.asarray(initial,dtype='<f4');self.settled=np.asarray(settled,dtype='<f4')
        self.triangles=np.asarray(triangles,dtype='<i4');self.hulls=to_json(hulls)
        if self.initial.shape!=(COUNT,3) or self.settled.shape!=(COUNT,3) or not np.isfinite(self.initial).all() or not np.isfinite(self.settled).all():
            raise ValueError('PR4 missing original/native settled cloth')
        if strict and hashlib.sha256(self.initial.tobytes()).hexdigest()!=case['initial_points_sha256']:
            raise ValueError('PR4 original cloth mesh hash differs')
        if self.triangles.shape!=(12800,3) or self.triangles.min()<0 or self.triangles.max()>=COUNT:
            raise ValueError('PR4 missing original material topology')
        if set(self.hulls)!={s+'/link'+str(j) for s in ('Left','Right') for j in (7,8)}:
            raise ValueError('PR4 missing original finger collision hulls')
        self.directory=Path(output_dir).resolve();self.directory.mkdir(parents=True,exist_ok=True)
        with (self.directory/'initial.npz').open('xb') as stream:
            np.savez_compressed(stream,initial=self.initial,settled=self.settled,triangles=self.triangles)
        self.static_sha=hashlib.sha256((self.directory/'initial.npz').read_bytes()).hexdigest()
        self.support_receipt=None
        if support_geometry is not None:
            from .mechanism.support import validate_support
            support=to_json(support_geometry);validate_support(support,case['origin'])
            path=self.directory/'support-geometry.json'
            data=json.dumps(support,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            with path.open('xb') as stream:stream.write(data)
            self.support_receipt=dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        self.json_stream=self.point_stream=None
        try:
            self.json_stream=(self.directory/'raw-trace.jsonl').open('xb')
            self.point_stream=(self.directory/'points.f32').open('xb')
        except BaseException:
            self.close();raise
        self.rows,self.previous_time=0,None
        self.raw_sha,self.points_sha=hashlib.sha256(),hashlib.sha256()
        self.raw_bytes=0

    def record(self,row,points):
        validate_row(row,self.rows)
        p=np.asarray(points,dtype='<f4')
        if p.shape!=(COUNT,3) or not np.isfinite(p).all():raise ValueError('PR4 missing native FEM points')
        if self.previous_time is not None and abs(row['time_s']-self.previous_time-1/120)>1e-6:
            raise ValueError('PR4 native clock gap')
        data=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
        binary=p.tobytes();self.json_stream.write(data);self.point_stream.write(binary)
        self.raw_sha.update(data);self.points_sha.update(binary);self.raw_bytes+=len(data)
        self.rows+=1;self.previous_time=row['time_s']

    def evidence(self,tick):
        if tick!=self.rows-1 or self.point_stream is None:raise ValueError('PR4 native audit/tick mismatch')
        self.json_stream.flush();self.point_stream.flush()
        values=dict(case=self.case,hulls=self.hulls,
            initial=dict(path=str(self.directory/'initial.npz'),sha256=self.static_sha),
            raw_trace=dict(path=str(self.directory/'raw-trace.jsonl'),sha256=self.raw_sha.hexdigest(),bytes=self.raw_bytes,frames=self.rows),
            points=dict(path=str(self.directory/'points.f32'),sha256=self.points_sha.hexdigest(),bytes=self.rows*POINT_BYTES,frames=self.rows))
        if self.support_receipt is not None:values['support_geometry']=self.support_receipt
        return NativeEvidence('PR4',self.instance_id,tick,values)

    def close(self):
        for name in ('json_stream','point_stream'):
            stream=getattr(self,name,None)
            if stream is not None:stream.close();setattr(self,name,None)


def checked_path(receipt, *, strict=False):
    path=Path(receipt['path'])
    if not path.is_absolute() or ('bytes' in receipt and path.stat().st_size!=receipt['bytes']):
        raise ValueError('PR4 audit file length mismatch')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    if strict and digest.hexdigest()!=receipt['sha256']:raise ValueError('PR4 audit SHA mismatch')
    return path


def reconstruct(evidence, *, strict=False):
    """Read the last actual FEM mesh against fixed target and source support.

    All recorded clocks/geometry remain validated. No historical motion,
    release interval, hand count, or material-region gate affects the score.
    """
    from .instances import validate_case
    if evidence.task_id != 'PR4':
        raise ValueError('PR4 scorer received another task')
    v = evidence.values
    from .mechanism.support import validate_support, support_witness
    if 'support_geometry' not in v:
        raise ValueError('PR4 missing native support geometry proof; legacy evidence requires source verification')
    case = to_json(v['case'])
    support_geometry=validate_support(json.loads(checked_path(v['support_geometry'], strict=strict).read_bytes()),case['origin'])
    validate_case(case, case['level'])
    n = v['points']['frames']
    if (type(n) is not int or n < 1 or v['raw_trace']['frames'] != n
            or evidence.tick != n-1 or v['points']['bytes'] != n*POINT_BYTES):
        raise ValueError('PR4 audit frame/tick binding mismatch')
    with np.load(checked_path(v['initial'], strict=strict), allow_pickle=False) as source:
        if set(source.files) != {'initial', 'settled', 'triangles'}:
            raise ValueError('PR4 initial audit schema mismatch')
        initial, settled, triangles = (source[k].copy() for k in ('initial', 'settled', 'triangles'))
    if (initial.shape != (COUNT, 3) or settled.shape != (COUNT, 3)
            or not np.isfinite(initial).all() or not np.isfinite(settled).all()
            or triangles.shape != (12800, 3) or triangles.min() < 0 or triangles.max() >= COUNT
            or hashlib.sha256(initial.astype('<f4').tobytes()).hexdigest() != case['initial_points_sha256']):
        raise ValueError('PR4 initial mesh/topology binding mismatch')
    hulls = to_json(v['hulls'])
    expected = {s+'/link'+str(j) for s in ('Left', 'Right') for j in (7, 8)}
    if set(hulls) != expected:
        raise ValueError('PR4 missing measured finger collision hulls')
    for parts in hulls.values():
        if not parts:
            raise ValueError('PR4 empty finger collision hull')
        for planes in parts:
            a = np.asarray(planes, dtype=float)
            if a.ndim != 2 or a.shape[1] != 4 or len(a) < 4 or not np.isfinite(a).all():
                raise ValueError('PR4 invalid finger collision planes')
    points = np.memmap(checked_path(v['points'], strict=strict), dtype='<f4', mode='r', shape=(n, COUNT, 3))
    # Keep memory bounded for long native traces. Final geometry remains the
    # actual measured material mesh; no hull/replacement surface stands in.
    for start in range(0, n, 64):
        chunk = points[start:start+64]
        if not np.isfinite(chunk).all():
            raise ValueError('PR4 nonfinite native FEM point')
    previous_time = None
    with checked_path(v['raw_trace'], strict=strict).open() as stream:
        count = 0
        for i, line in enumerate(stream):
            row = json.loads(line)
            validate_row(row, i)
            if previous_time is not None and abs(row['time_s']-previous_time-1/120) > 1e-6:
                raise ValueError('PR4 raw audit native clock gap')
            previous_time = row['time_s']
            count += 1
    if count != n:
        raise ValueError('PR4 raw audit length differs from FEM points')
    from .mechanism.terminal_shape import terminal_shape
    origin = case['origin']
    shape = terminal_shape(points[-1], triangles, case['level'], origin['fold_target_center'], origin['width'])
    support=support_witness(points[-1],support_geometry)
    checks = dict(target_shape=bool(shape['shape_success']),target_support=support['supported'])
    return dict(success=all(checks.values()), checks=checks, shape=shape, support=support,
                hands_and_grasp_regions_scored=False, expert_phases_consumed=False,
                measurement_frames=n, read_tick=evidence.tick)
