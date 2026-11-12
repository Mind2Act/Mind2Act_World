"""Public CP07 recipe counts, selected by authenticated source cohort only.

The held-out request index contains source identities, not a reference plan.
Only public source counts and explicit current binary-scoring metadata cross
the adapter boundary. Historical step counts are labelled source_scoring_steps.
No raw source, private binding, solution or target geometry is opened.
"""
from collections.abc import Mapping

from mind2act_eval.original_test_inputs import checked_record, request_records
from mind2act_eval.scoring import SCORING_VERSION

TRAIN_LAYOUT_PROTOCOL = 'mind2act-common_input-coding-cp07-training-v3'
TEST_PROTOCOL = 'mind2act-common_input-coding-cp07-original8-v2'
LEGACY_PROTOCOL = 'mind2act-common_input-coding-v1'
PROFILE_SCHEMA = 'cp07.public-recipe.v1'
TEST_SOURCE = 'cp07-original8-train-v2-cpu'
TRAINING_SOURCE = 'cp07-bridge-car-canonical-fff77c74'
TEST_STOCK = {'easy': 8, 'medium': 11, 'hard': 14}
TRAINING_COUNTS = {'easy': (8, 8), 'medium': (13, 18), 'hard': (13, 28)}


def public_profile(cohort, difficulty):
    if difficulty not in TEST_STOCK:
        raise ValueError('Unknown CP07 public recipe difficulty')
    if cohort == 'original-test-eight':
        required, stock = 8, TEST_STOCK[difficulty]
    elif cohort in ('diagnostic-training', 'training-layout-test'):
        required, stock = TRAINING_COUNTS[difficulty]
    else:
        raise ValueError('Unknown CP07 public recipe cohort')
    return dict(schema=PROFILE_SCHEMA, cohort=cohort, required_pieces=required,
                stock_pieces=stock, source_scoring_steps=required + 2,
                score_kind='binary', scoring_version=SCORING_VERSION)


def validate_parameters(parameters, difficulty, protocol):
    if not isinstance(parameters, Mapping) or set(parameters) != {'cp07_recipe_profile'}:
        raise ValueError('An explicit source-qualified CP07 public recipe profile is required')
    profile = parameters['cp07_recipe_profile']
    if (not isinstance(profile, Mapping)
            or any(type(profile.get(k)) is not int for k in
                   ('required_pieces', 'stock_pieces', 'source_scoring_steps'))):
        raise ValueError('CP07 public recipe counts must be exact integers')
    expected = public_profile(profile.get('cohort'), difficulty)
    if dict(profile) != expected:
        raise ValueError('CP07 public recipe profile differs from its source cohort')
    required_protocol = TEST_PROTOCOL if profile['cohort'] == 'original-test-eight' else LEGACY_PROTOCOL
    if profile['cohort'] == 'training-layout-test':
        required_protocol = TRAIN_LAYOUT_PROTOCOL
    if protocol != required_protocol:
        raise ValueError('CP07 public recipe cohort requires its explicit input protocol')
    return expected


def project_parameters(spec, assets):
    """Runner-owned projection; authenticate before returning public counts."""
    if spec.task_id != 'CP07':
        raise ValueError('CP07 public recipe projection requires CP07')
    mode = spec.config.get('cp07_mode')
    if mode == 'train_layout_test':
        from mind2act_eval.cp07_layout_identity import validate_identity
        validate_identity(spec)
        cohort = 'training-layout-test'
    elif mode == 'source_bound_test' and spec.source_id == TEST_SOURCE:
        # Existing source authority checks task/tier/seed/config/index against
        # the pinned original request. A plausible case ID alone grants nothing.
        checked_record(spec, request_records(assets.resolve('test_source_request')))
        cohort = 'original-test-eight'
    elif (mode == 'diagnostic_training' and spec.source_id == TRAINING_SOURCE
          and 'test_source' not in spec.config):
        cohort = 'diagnostic-training'
    else:
        raise ValueError('Unknown or mixed CP07 public recipe source profile')
    return {'cp07_recipe_profile': public_profile(cohort, spec.difficulty)}
