"""Map a baseline per-episode spec (`jobs/M/b-mainline-20261025-v1/specs/<tier>/episode_NNNN.json`) to an InstanceSpec.

Planner/expert data (`motion_jobs`, `source_plan`, collision boxes, grasp and motion revisions) is not copied.
"""
import json
import math
from functools import lru_cache
from importlib import resources

from mind2act_eval.contracts import ClockSpec, InstanceSpec

MANIFEST_KEYS = ('case', 'bodies', 'carriers', 'stations', 'submission_stations', 'neutral', 'cover_park',
                 'target_body_id', 'target_cover_id', 'robot_home_joint_degrees', 'environment_transport',
                 'cover_transport_lift_m', 'static_collision_roots', 'replay_ready_prim', 'station')
CASE_KEYS = ('difficulty', 'swaps', 'expected_path', 'initial_target_station', 'exchange_transport',
             'cover_phase_seconds', 'stable_seconds', 'allow_flat_label_contact', 'submission_stations')


@lru_cache(maxsize=None)
def layout() -> dict:
    return json.loads(resources.files('mind2act_eval.tasks.M1').joinpath('config', 'layout.json').read_text(encoding='utf-8'))


def require_baseline_clock(clock) -> None:
    """Choreography, hold and ledger windows are 120 Hz tick counts; reject any other physics rate."""
    if clock.physics_hz != 120:
        raise ValueError('M1 timing is bound to the 120 Hz baseline physics clock')


def spec_from_row(row: dict, *, asset_version: str, protocol_version: str = None,
                  clock: ClockSpec = ClockSpec(120, 30, 30)) -> InstanceSpec:
    lay = layout()
    require_baseline_clock(clock)
    cfg = row['config']
    manifest = cfg['native_inputs']['manifest']
    if row['task_id'] != 'M' or row['split'] != 'train' or manifest['case'] != cfg['source_manifest_case']:
        raise ValueError('not a consistent M/M1 train instance')
    if row['case_id'] != cfg['source_job']['id']:
        raise ValueError('M job identity differs')
    if len(manifest['case']['expected_path']) != row['binding']['expected_units']:
        raise ValueError('M expected path length differs from the binding')
    kept = {k: manifest[k] for k in MANIFEST_KEYS if k in manifest}
    kept['case'] = {k: manifest['case'][k] for k in CASE_KEYS if k in manifest['case']}
    config = dict(manifest=kept, instruction=row['instruction'],
                  robot_rest=[math.radians(d) for d in lay['home_joints_deg']] + [lay['rest_finger_m']] * 2)
    return InstanceSpec(
        task_id='M1', instance_id=row['case_id'], difficulty=row['difficulty'], source_id=lay['source_id'],
        config_version=lay['config_version'], asset_version=asset_version,
        protocol_version=protocol_version or lay['protocol_version'], clock=clock, config=config,
        assets={'station': f"M1/{row['case_id']}/scene.usda", 'x5': 'robot/x5/ARX.usd'})


def plain(value):
    """Frozen specs/evidence hold tuples and mappingproxy; the mechanism wants plain lists and dicts."""
    from collections.abc import Mapping
    from mind2act_eval.contracts import FrozenHistory
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, FrozenHistory)):
        return [plain(v) for v in value]
    return value
