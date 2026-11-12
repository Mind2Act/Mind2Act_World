from copy import deepcopy
import hashlib
from types import SimpleNamespace as NS

import pytest
from mind2act_eval.contracts import to_json
from mind2act_eval.tasks.P2.binding import DISMANTLING_RULE, digest, instruction, validate
from mind2act_eval.tasks.P2.instances import spec_from_source
from .test_p2_measurements import source


@pytest.mark.parametrize('tier', ['easy', 'medium', 'hard'])
def test_typed_source_spec_binds_seed_assets_clock_and_ruling(tmp_path, tier):
    row = source(tier)
    row['seed'] = 2049
    before = deepcopy(row)
    path = tmp_path/'asset.usda'
    path.write_text('#usda 1.0\n')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    foods = set(row['target_foods_bottom_to_top'] + row['decoy_foods'])
    assets = {key: 'asset.usda' for key in ['station', 'task_scene', 'x5', *['food_'+f for f in foods]]}
    spec = spec_from_source(row, assets=assets, asset_version='synthetic', scene_sha256=sha,
        task_scene_sha256=sha, food_assets_sha256={f: sha for f in foods}, source_world_time=8.025)
    assert spec.seed == row['seed'] and spec.instance_id == row['instance_id']
    assert spec.clock.ticks_per_action == 4 and spec.clock.physics_hz == 120
    assert spec.config['dismantling_rule_version'] == DISMANTLING_RULE
    assert spec.config['evaluation_mode'] == 'formal'
    assert spec.config['instruction'] == instruction(tier)
    assert spec.config['source_record_sha256'] == digest(row)
    assert row == before
    assert validate(spec, NS(assets=NS(resolve=lambda key: path)))[0] == row
    changed = to_json(spec)
    changed['config'].pop('dismantling_rule_version')
    from mind2act_eval.contracts import InstanceSpec
    with pytest.raises(ValueError, match='explicit'):
        validate(InstanceSpec.from_dict(changed), NS(assets=NS(resolve=lambda key: path)))
