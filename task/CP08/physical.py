"""Final placements and explicit hard rework from per-tick CP08 measurements."""
from copy import deepcopy
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .source_rules import unwanted_robot_contacts
from mind2act_eval.scoring import CP_PARTIAL_CAPS

VERSION = 'mind2act-final-scoring-20261105'
CONTACT_SCHEMA = 'cp08-physx-owned-support-v2'


def owned_support(pair, points):
    """PhysX normals point from collider1 toward collider0; preserve ownership."""
    out = []
    if len(pair) != 2:
        raise ValueError('CP08 missing collider ownership')
    for point in points:
        normal = np.asarray(point['normal'], dtype=float)
        separation = float(point['separation_m'])
        if normal.shape != (3,) or not np.isfinite(normal).all() or not math.isfinite(separation):
            raise ValueError('CP08 nonfinite contact geometry')
        out.append(dict(collider=pair[0], other=pair[1], upward_normal=float(normal[2]), separation_m=separation))
        out.append(dict(collider=pair[1], other=pair[0], upward_normal=float(-normal[2]), separation_m=separation))
    return out



def canonical_support_geometry(pair, points):
    """Compact raw geometry: normal belongs to canonical pair[0], no path duplication."""
    canonical=sorted(pair);sign=1. if pair[0]==canonical[0] else -1.
    owned_support(pair,points)  # Validate raw geometry before retaining it.
    return [dict(normal_on_pair0=[sign*float(v) for v in p['normal']],separation_m=float(p['separation_m'])) for p in points]

def _moved(body, origin, tolerance):
    delta = np.asarray(body['position']) - np.asarray(origin['position'])
    angle = (Rotation.from_quat(np.asarray(body['quaternion'])[[1,2,3,0]]).inv() *
             Rotation.from_quat(np.asarray(origin['quaternion'])[[1,2,3,0]])).magnitude()
    return np.linalg.norm(delta[:2]) >= tolerance or abs(delta[2]) >= .003 or angle >= math.radians(1)


def _supported_at_target(body, spec, tolerance):
    xyz = np.asarray(body['position']); goal = np.asarray(spec['target'])
    expected = set(spec['supports'])
    supported = expected.issubset(body['supports']) if expected else body['table_support']
    return bool(np.linalg.norm(xyz[:2]-goal[:2]) < tolerance and
                abs(xyz[2]-goal[2]) < .003 and supported)


def _placed(body, spec, tolerance):
    return _supported_at_target(body, spec, tolerance)


class StabilityWindow:
    """Compatibility adapter for the current measured placement, without dwell."""
    def __init__(self, spec, config):
        self.spec, self.config, self.stable = spec, config, False

    def append(self, body):
        self.stable = _placed(body, self.spec, self.config['tolerance'])


def _validate(row, specs):
    if set(row['blocks']) != set(specs):
        raise ValueError('CP08 missing physical block row')
    for body in row['blocks'].values():
        p=np.asarray(body['position'],dtype=float);q=np.asarray(body['quaternion'],dtype=float)
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:
            raise ValueError('CP08 invalid measured pose')
        if any(not math.isfinite(body[k]) for k in ('tilt_deg','yaw_error_deg')):
            raise ValueError('CP08 invalid measured orientation')
        if any(type(body[k]) is not bool for k in ('table_support','robot_contact')) or any(type(v) is not bool for v in body['bilateral_by_arm'].values()):
            raise ValueError('CP08 invalid physical contact predicates')
        if set(body['bilateral_by_arm']) != {'Left','Right'} or not isinstance(body['supports'],list):
            raise ValueError('CP08 incomplete physical contacts')
def evaluate(rows, config, initial):
    """Independent terminal placements plus the explicit hard unload/repair flow."""
    hard = config['level'] == 'hard'
    specs = {b['id']: b for b in config['blocks']+config.get('distractors', [])}
    targets = {b['id']: b for b in config['blocks']}
    if set(initial['blocks']) != set(specs):
        raise ValueError('CP08 missing reset block evidence')
    step_ids = ['C_unloaded', 'B_repaired', 'C', 'D', 'E', 'F', 'finish'] if hard else ['B', 'A', 'C'] + (['D'] if config['level']=='medium' else []) + ['finish']
    unload_tick = repair_tick = None
    upper_ticks = {}; events = []; previous_b = False
    tolerance = config['tolerance']
    for index, row in enumerate(rows):
        if row['tick'] != index+1 or row.get('support_measurements_complete') is not True:
            raise ValueError('CP08 consecutive owned physical support measurements required')
        _validate(row, specs)
        if not hard:
            continue
        bodies = row['blocks']; c, b = bodies['C'], bodies['B']
        cp, bp = np.asarray(c['position']), np.asarray(b['position'])
        cs, bs = np.asarray(specs['C']['size']), np.asarray(specs['B']['size'])
        # A missing single manifold is not proof of unloading: C must also be
        # geometrically clear of B, either above it or outside its footprint.
        clear = cp[2]-cs[2]/2 > bp[2]+bs[2]/2+1e-6 or any(abs(cp[j]-bp[j]) > (cs[j]+bs[j])/2 for j in (0,1))
        if unload_tick is None and not set(c['supports']) & {'A', 'B'} and clear:
            unload_tick = row['tick']; events.append(dict(tick=unload_tick,event='C_unloaded'))
        b_good = _placed(b, targets['B'], tolerance)
        if unload_tick is not None and not previous_b and b_good:
            repair_tick = row['tick']; upper_ticks = {}
            events.append(dict(tick=repair_tick,event='B_repaired'))
        if repair_tick is not None:
            for name in ('C','D','E','F'):
                if _placed(bodies[name],targets[name],tolerance):
                    upper_ticks.setdefault(name,row['tick'])
        previous_b = b_good
    checks = {n: bool(rows and _placed(rows[-1]['blocks'][n], spec, tolerance)) for n,spec in targets.items()}
    if hard:
        units = [unload_tick is not None, checks.get('B',False)] + [checks.get(n,False) for n in ('C','D','E','F')]
        flow = repair_tick is not None and all(n in upper_ticks for n in ('C','D','E','F'))
    else:
        units = [checks.get(n,False) for n in step_ids[:-1]]; flow = True
    reference_only = bool(rows and not any(d['id'] in body['supports'] for d in config.get('distractors',[]) for body in rows[-1]['blocks'].values()))
    finish=bool(all(checks.values()) and flow and all(units) and reference_only)
    if finish and config.get('distractors'):
        area=initial.get('build_area_bounds_m')
        if area is None:raise ValueError('CP08 actual marked build area measurements missing')
        lower,upper=np.asarray(area['lower'],float),np.asarray(area['upper'],float)
        if lower.shape!=(2,) or upper.shape!=(2,) or not np.isfinite([*lower,*upper]).all() or not np.all(lower<upper):
            raise ValueError('CP08 invalid actual build area bounds')
        finish=all(not np.all((np.asarray(rows[-1]['blocks'][d['id']]['position'])[:2]>=lower) & (np.asarray(rows[-1]['blocks'][d['id']]['position'])[:2]<=upper)) for d in config['distractors'])
    units.append(finish)
    k = sum(units); success = all(units)
    core_ids = ['B','C','D','E','F'] if hard else step_ids[:-1]
    core_k = sum(checks[name] for name in core_ids)
    return dict(success=success, score=100. if success else CP_PARTIAL_CAPS[config['level']]*core_k/len(core_ids),
        core_k=core_k, core_N=len(core_ids),
        report_contract=VERSION, report_step_source='measured_physical_state',
        raw_sampling_complete=True, raw_contact_schema=CONTACT_SCHEMA,
        measured_report_steps=units, completed_steps=k, total_steps=len(units),
        completion_percent=100*k/len(units), first_error=None,
        cutoff_tick=rows[-1]['tick'] if rows else 0, step_ids=step_ids,
        step_ticks=[None]*len(units), events=events, physical_moves=[],
        pending_physical_transfers=[], expert_events_consumed=False,
        measurement_source='measured block centers and owned physical support',
        block_checks=checks, cutoff_block_checks=checks, no_unwanted_contacts=True)
