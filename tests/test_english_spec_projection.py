"""English runnable specs retain the exact source and replay identities."""
from dataclasses import replace
import pytest

from mind2act_eval.contracts import to_json
from mind2act_eval.tasks.CP09.instances import config, spec_for_selected, validate
from mind2act_eval.robot.task_collision_overlay import (
    DECISION, ENGLISH_DECISION, SOURCE_ENGLISH_DECISION, canonical_geometry_decision,
    english_geometry_decision, is_geometry_decision)


def test_cp09_english_decision_preserves_selected_physics_binding():
    source = spec_for_selected('easy', instance_id=config('selected')['easy']['instance_id'],
        station_asset='station.usda', x5_asset='x5.usd', station_proof_asset='proof.json',
        case_asset='case.json', station_relocation_asset='relocation.json', asset_version='v1')
    runtime = replace(source, config=dict(to_json(source.config), geometry_decision=(
        'Preserve the acquisition finger pads and geometry; '
        'do not remove them or replace them with generic geometry.')))
    assert validate(runtime) == validate(source)
    assert {k: v for k, v in runtime.config.items() if k != 'geometry_decision'} == {
        k: v for k, v in source.config.items() if k != 'geometry_decision'}


def test_exact_decision_projection_round_trips_without_a_caller_mask():
    assert english_geometry_decision(DECISION) == ENGLISH_DECISION
    assert canonical_geometry_decision(ENGLISH_DECISION, DECISION) == DECISION
    assert SOURCE_ENGLISH_DECISION != ENGLISH_DECISION
    assert is_geometry_decision(SOURCE_ENGLISH_DECISION)
    with pytest.raises(ValueError):
        canonical_geometry_decision(SOURCE_ENGLISH_DECISION, DECISION)


def test_training_spec_with_unreviewed_assets_cannot_self_authorize_as_an_english_derivative():
    from mind2act_eval.english_specs import project_spec, approved_source_config_sha256
    source = spec_for_selected('easy', instance_id=config('selected')['easy']['instance_id'],
        station_asset='station.usda', x5_asset='x5.usd', station_proof_asset='proof.json',
        case_asset='case.json', station_relocation_asset='relocation.json', asset_version='v1')
    with pytest.raises(ValueError, match='exact code-owned'):
        project_spec(source)
    assert approved_source_config_sha256(source) is None


@pytest.mark.parametrize('text', [ENGLISH_DECISION + ' ', 'keep generic pads', '', None, 123])
def test_unknown_decisions_cannot_authorize_source_geometry(text):
    assert not is_geometry_decision(text)
    with pytest.raises(ValueError):
        english_geometry_decision(text)
    with pytest.raises(ValueError):
        canonical_geometry_decision(text, DECISION)


@pytest.mark.parametrize('field,value', [('seed', 1), ('source_id', 'different-source'),
                                        ('config_version', 'different-version')])
def test_english_label_does_not_waive_the_selected_physical_source_identity(field, value):
    source = spec_for_selected('easy', instance_id=config('selected')['easy']['instance_id'],
        station_asset='station.usda', x5_asset='x5.usd', station_proof_asset='proof.json',
        case_asset='case.json', station_relocation_asset='relocation.json', asset_version='v1')
    runtime = replace(source, config=dict(to_json(source.config), geometry_decision=ENGLISH_DECISION))
    with pytest.raises(ValueError):
        validate(replace(runtime, **{field: value}))


@pytest.mark.parametrize('field,value', [('physics_budget', 1), ('station_config_sha256', '0' * 64),
                                        ('source_case_sha256', '0' * 64), ('robot_rest', [0.] * 8)])
def test_english_label_does_not_waive_the_selected_physical_configuration(field, value):
    source = spec_for_selected('easy', instance_id=config('selected')['easy']['instance_id'],
        station_asset='station.usda', x5_asset='x5.usd', station_proof_asset='proof.json',
        case_asset='case.json', station_relocation_asset='relocation.json', asset_version='v1')
    with pytest.raises(ValueError):
        validate(replace(source, config=dict(to_json(source.config),
            geometry_decision=ENGLISH_DECISION, **{field: value})))
