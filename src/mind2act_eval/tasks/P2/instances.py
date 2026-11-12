"""Build a public P2 spec from an exact selected source record and resolver IDs."""
import math

from mind2act_eval.contracts import ClockSpec, InstanceSpec
from .binding import SOURCE_ID, VERSION, DISMANTLING_RULE, EVALUATION_MODE, digest, instruction, validate_source


def spec_from_source(row, *, assets, asset_version, scene_sha256, task_scene_sha256,
                     food_assets_sha256, instance_id=None, source_world_time=None,
                     cameras=None, clock=ClockSpec(120, 30, 30)):
    validate_source(row)
    if clock.physics_hz != 120 or clock.control_hz not in (30, 120) or clock.rgb_hz != 30:
        raise ValueError('P2 source clock requires120 physics,30/120 control and30 RGB')
    if not {'station', 'task_scene', 'x5'} <= assets.keys():
        raise ValueError('P2 requires source station/task_scene/shared X5 resolver IDs')
    foods = set(row['target_foods_bottom_to_top'] + row['decoy_foods'])
    if not foods <= food_assets_sha256.keys() or any('food_' + food not in assets for food in foods):
        raise ValueError('P2 requires every original food asset')
    config = dict(source_instance=row, source_record_sha256=digest(row),
                  instruction=instruction(row['difficulty']), input_protocol_version=VERSION,
                  dismantling_rule_version=DISMANTLING_RULE,
                  evaluation_mode=EVALUATION_MODE,
                  scene_sha256=scene_sha256, task_scene_sha256=task_scene_sha256,
                  food_assets_sha256=food_assets_sha256,
                  robot_rest=[math.radians(v) for v in [0, 65, 100, -65, 0, 0]] + [.044, .044])
    if source_world_time is not None:
        config['source_world_time'] = source_world_time
    if cameras is not None:
        config['cameras'] = cameras
    return InstanceSpec('P2', instance_id or row['instance_id'], row['difficulty'], SOURCE_ID,
                        VERSION, asset_version, 'v1', seed=row['seed'], clock=clock,
                        assets=assets, config=config)
