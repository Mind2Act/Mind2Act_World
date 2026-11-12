import pytest
from mind2act_eval import cli
from mind2act_eval.release_contract import ReleaseContractError
from mind2act_eval.runtime_release import validate_release_inputs
from test_instance_catalog import catalog_release, save_catalog
from test_runtime_release import release


def prepare(catalog_release):
    spec, args, descriptor, catalog = catalog_release
    row = next(r for r in catalog['instances'] if r['case_id'] == spec.instance_id)
    row['limits'] = dict(physics_ticks=20160, wall_s=7200, retries=0)
    save_catalog(args, descriptor, catalog)
    args.physics_ticks, args.wall_s, args.retries = 20160, 21960, 0
    return spec, args


def test_longer_infrastructure_wall_keeps_original_simulation_budget(catalog_release):
    spec, args = prepare(catalog_release)
    assert validate_release_inputs(spec, args).resolve_case(spec.instance_id)['limits']['physics_ticks'] == 20160


@pytest.mark.parametrize('field,value', [('physics_ticks', 20161), ('retries', 1), ('wall_s', 7199), ('wall_s', float('nan')), ('wall_s', float('inf'))])
def test_longer_wall_cannot_change_physics_retry_or_shorten_original(catalog_release, field, value):
    spec, args = prepare(catalog_release)
    setattr(args, field, value)
    with pytest.raises(ReleaseContractError, match='original release budgets'):
        validate_release_inputs(spec, args)
