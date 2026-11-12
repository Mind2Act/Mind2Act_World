"""Both native scene and independent scorer must use their validated source clock."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from mind2act_eval.contracts import InstanceSpec, NativeEvidence
from mind2act_eval import original_test_inputs
from mind2act_eval.tasks.CP08.audit import Audit
from mind2act_eval.tasks.CP08.instances import instructions, selected
from mind2act_eval.tasks.CP08.projection import cue_visible
from mind2act_eval.tasks.CP08.scene import StructureScene
from mind2act_eval.tasks.CP08.scorer import score
from test_cp08_projection_audit import projection
from test_cp08_scorer import raw


@pytest.mark.parametrize('level', ['easy', 'medium', 'hard'])
@pytest.mark.parametrize('original', [True, False], ids=['source-state0', 'selected-H5-state72'])
@pytest.mark.parametrize('consumer', ['scene', 'scorer'])
def test_scene_and_scorer_agree_at_both_cue_boundaries(tmp_path, monkeypatch, level, original, consumer):
    from mind2act_eval.tasks.CP08 import instances
    row = selected(level)
    config = dict(source_case=deepcopy(row['source_case']), scene_sha256=row['scene_sha256'],
        physics_budget=row['physics_budget'], source_spec_sha256=row['source_spec_sha256'],
        source_projection=projection(level), instruction=instructions(level)['en'])
    identity, seed = row['instance_id'], 0
    if original:
        identity, seed = 'cpu-original-cp08-'+level, 23
        config['source_case']['split'] = 'eval'
        config['source_projection'] = dict(source_physics_hz=120, source_state_tick=0,
            source_action_tick=4, source_cue_hide_tick=360)
        config['test_source'] = dict(schema=original_test_inputs.SCHEMA, instance_set='test50-r5',
            split='test', logical_slot=identity, source_seed=seed,
            source_configuration_sha256=config['source_spec_sha256'],
            source_request_sha256=original_test_inputs.REQUEST_SHA256,
            snapshot_sha256=original_test_inputs.SNAPSHOT_SHA256)
        # Fake only private source I/O. Projection validation, scene, raw audit,
        # Replay, contact normalization and independent scoring remain real.
        monkeypatch.setattr(instances, 'original_case', lambda *a: {})
        monkeypatch.setattr(original_test_inputs, 'audit_source', lambda *a: dict(test_source=config['test_source']))
        monkeypatch.setattr(original_test_inputs, 'read_audit', lambda *a: dict(task_id='CP08',
            split='eval', config=dict(source_case=config['source_case'])))
    spec = InstanceSpec('CP08', identity, level, 'fr3-replay-new15-20261031',
        'cp08-source-projection-v1', 'cpu-only', 'v1', seed=seed, config=config)

    end = 361 if original else 289
    if consumer == 'scorer':
        audit = Audit(tmp_path/'raw.jsonl')
        for tick in range(end+1):
            sample = raw(config['source_case'], tick)
            sample['cue_visible'] = cue_visible(tick, config['source_projection'])
            audit.append(sample)
        values = dict(source_case=config['source_case'], source_projection=config['source_projection'],
                      audit=audit.receipt())
        if original: values['test_source'] = config['test_source']
        audit.close()
        assert score(NativeEvidence('CP08', identity, end, values)).value == 0
        return

    class Sensors:
        errors = []
        visible = True
        def read(self, tick):
            sample = raw(config['source_case'], tick)
            sample['cue_visible'] = self.visible
            return sample
        def set_cue(self, visible): self.visible = visible
        def close(self): pass

    sensors = Sensors()
    scene = StructureScene(spec, SimpleNamespace(output_dir=tmp_path, assets=None), sensors)
    try:
        for tick in range(1, end+1):
            scene.before_step(tick)
            scene.after_step(tick)
            assert scene.replay.rows[-1]['cue_visible'] == cue_visible(tick, config['source_projection'])
        result = score(scene.evidence(end))
        assert result.value == 0 and not result.components['task_success']
        assert scene.replay.rows[-2]['cue_visible'] is True
        assert scene.replay.rows[-1]['cue_visible'] is False
    finally:
        scene.close()


@pytest.mark.parametrize('offset', [0, 72])
@pytest.mark.parametrize('after_hide', [False, True])
def test_replay_still_rejects_wrong_exposure_at_each_boundary(offset, after_hide):
    from mind2act_eval.tasks.CP08.replay import Replay
    config = selected('easy')['source_case']
    replay = Replay(config, source_state_tick=offset)
    end = 360-offset+int(after_hide)
    for tick in range(1, end):
        sample = raw(config, tick)
        sample['cue_visible'] = tick+offset <= 360
        replay.sample(sample)
    sample = raw(config, end)
    sample['cue_visible'] = after_hide
    with pytest.raises(ValueError, match='screen exposure differs'):
        replay.sample(sample)
